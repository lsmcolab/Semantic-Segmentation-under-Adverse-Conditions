import torch
from torch import nn
from torch.nn import functional as F
from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]

class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        # X01
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

        self._init_weight()

    def forward(self, feature):
        #low_level_feature = 4, 48, 192, 192
        low_level_feature = self.project( feature['low_level'] )

        # 4, 256, 48, 48
        output_feature,weather_pred,time_pred = tuple(self.aspp(feature['out']))
        # 4, 256, 192, 192
        #upsample by 4
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        # inside self.classifier it is 4, 304(256+48), 192, 192
        # After self.classifier it is 4, 19(n classes), 192, 192
        return [self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) ),weather_pred,time_pred]
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []

# 1by1 conv layer in the ASPP Block
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

# 6 12 18 branches in ASPP Block
        rate1, rate2, rate3 = tuple(atrous_rates)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, 128,  3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)#kerim weather

        self.fc1 = nn.Linear(294912, 120)#kerim weather
        self.fc2 = nn.Linear(120, 84)#kerim weather
        self.fc3_weather = nn.Linear(84, 4)#kerim weather
        self.fc3_time = nn.Linear(84, 2)  # kerim weather
        self.sofmax = nn.Softmax()#kerim weather



        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))


# you have 5 branches + 1mine
        self.convs = nn.ModuleList(modules)

# ??
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        weather_pred = None
        time_pred = None
        # we only activate this model for training
        if self.training:
            weather_embeddings = self.relu(self.bn1(self.conv1(x)))
            weather_embeddings = self.relu(self.bn2(self.conv2(weather_embeddings)))
            weather_embeddings = torch.flatten(weather_embeddings, 1)
            weather_embeddings = F.relu(self.fc1(weather_embeddings))
            weather_embeddings = F.relu(self.fc2(weather_embeddings))
            weather_pred = self.fc3_weather(weather_embeddings)

            time_embeddings = self.relu(self.bn1(self.conv1(x)))
            time_embeddings = self.relu(self.bn2(self.conv2(time_embeddings)))
            time_embeddings = torch.flatten(time_embeddings, 1)
            time_embeddings = F.relu(self.fc1(time_embeddings))
            time_embeddings = F.relu(self.fc2(time_embeddings))
            time_pred = self.fc3_time(time_embeddings)

        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        # shape is 4(batchsize), 1536(256*6), 48, 48
        # after self.project(res) 4, 256, 48, 48
        return [self.project(res), weather_pred, time_pred]
