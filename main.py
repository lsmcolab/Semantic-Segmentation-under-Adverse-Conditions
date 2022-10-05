# The model is weather and time aware!
# This models is trained on: AWSS + CS 1:1
# low level features i.e module.backbone.low_level_features (Atrous Conv.) are frozen when training on CS
# Multi-task learning two losses Segmentation loss and weather_time loss, propagated separtely.
# weather awareness just of the Atrous Convolution
# ------------------------
# Please note that our code is based on DeepLabV3+ pytorch implementation.
# --------------------------
import torch
import torch.nn as nn
import numpy as np
import random
import os
from tqdm import tqdm
import network
import utils
import argparse
from torch.utils import data
from datasets import Cityscapes, ACDC, AWSS
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from utils.visualizer import Visualizer
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=True,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--test_batch_size", type=int, default=4,
                        help='batch size for testing (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts,tr_ds_name=None):
    """ Dataset And Augmentation
    """

    if opts.dataset == 'cityscapes' or tr_ds_name=="cityscapes":
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        # should be just used in the training phase
        val_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])


        train_dst = Cityscapes(root=opts.data_root_cs, split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root_cs,   split='val', transform=val_transform)
        tst_dst = Cityscapes(root=opts.data_root_cs,   split='test', transform=val_transform)

    if opts.dataset == 'ACDC' or tr_ds_name=="ACDC":
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

# should be just used in the training phase
        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            # et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        train_dst = []
        val_dst = ACDC(root=opts.data_root_acdc, split='val', transform=val_transform)
        if opts.ACDC_test_class is not None:
            tst_dst = ACDC(root=opts.data_root_acdc,  split='test', transform=val_transform,test_class=opts.ACDC_test_class)
        else:
            tst_dst = []

    if opts.dataset == "AWSS" or tr_ds_name=="AWSS":#
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.1673, 0.1685, 0.1948],
                            std=[0.0801, 0.0775, 0.0805]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.1673, 0.1685, 0.1948],
                            std=[0.0801, 0.0775, 0.0805]),
        ])

        train_dst = AWSS(root=opts.data_root_awss, split='train', transform=train_transform)
        val_dst = AWSS(root=opts.data_root_awss, split='val', transform=val_transform)
        tst_dst = []

    return train_dst, val_dst, tst_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels,names,weather_ids,time_ids, data_domain) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs,_,_ = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    # target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    # target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    # Image.fromarray(image).save('results/%d_image.png' % img_id)
                    # Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save(f'results/{names[i]}')

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    # plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main(ACDC_test_class = None,n_itrs=10000000,MODE=12345):

    opts = get_argparser().parse_args()
    opts.ACDC_test_class = ACDC_test_class
    opts.finetune = False
    opts.pretrained_model = None
    # MODE = 0#train on cityscapes and AWSS
    # MODE = 11#test on cityscapes
    # MODE = 21#test on acdc
    opts.data_root_cs = "/home/kerim/DataSets/SemanticSegmentation/cityscapes"#Update as necessary
    opts.data_root_acdc = "/home/kerim/DataSets/SemanticSegmentation/ACDC"#Update as necessary
    opts.data_root_awss = "/home/kerim/Silver_Project/AWSS"#Update as necessary
    opts.total_itrs = n_itrs
    opts.test_class = None
    opts.val_batch_size = 8
    # -----------------------------------------------------------
    if MODE==0:#train on cityscapes and AWSS
        opts.test_only = False
        opts.save_val_results = False
        opts.dataset = 'cityscapes_AWSS'#"cityscapes" and "AWSS"

    elif MODE==11:#test pretrained cityscapes and finetuned on AWSS test on *cityscapes*
        opts.test_only = True
        opts.save_val_results = True
        opts.dataset = "cityscapes"
        opts.ckpt = "checkpoints/D01_deeplabv3plus_mobilenet_cityscapes_AWSS_os16.pth"

    elif MODE==21:#test pretrained on cityscapes fine-tuned on AWSS test on *acdc*
        opts.test_only = True
        opts.save_val_results = True
        opts.dataset = "ACDC"
        opts.ckpt = "checkpoints/D01_deeplabv3plus_mobilenet_cityscapes_AWSS_os16.pth"
    # --------------------------------------------------------------
    opts.model = "deeplabv3plus_mobilenet"
    opts.enable_vis = True
    opts.vis_port = 28300
    opts.gpu_id = '0'
    opts.lr = 0.1
    opts.crop_size = 768
    opts.batch_size = 4
    opts.output_stride = 16
    opts.crop_val = True


    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],   std=[0.229, 0.224, 0.225])
    elif opts.dataset.lower() == 'acdc':
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],   std=[0.229, 0.224, 0.225])
        opts.num_classes = 19
    elif opts.dataset.lower() == 'awss':
        opts.num_classes = 19
        denorm = utils.Denormalize(mean=[0.1987, 0.1846, 0.1884], std=[0.1084, 0.0950, 0.0902])
    else:
        opts.num_classes = 19
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    #get cs training and validation datasets
    if opts.test_only:
        _, _, tst_dst = get_dataset(opts)
    else:#training
        # get cs training dataset
        train_dst_cs, val_dst_cs, _ = get_dataset(opts,'cityscapes')
        # get awss training dataset
        train_dst_awss, _, _ = get_dataset(opts,'AWSS')
        # get acdc training dataset
        _, val_dst_acdc, _ = get_dataset(opts,'ACDC')

        train_loader_cs = data.DataLoader(
            train_dst_cs, batch_size=opts.batch_size, shuffle=True, num_workers=8,
            drop_last=True)  # drop_last=True to ignore single-image batches.

        train_loader_awss = data.DataLoader(
            train_dst_awss, batch_size=opts.batch_size, shuffle=True, num_workers=8,
            drop_last=True)  # drop_last=True to ignore single-image batches.


        val_loader_cs = data.DataLoader(
            val_dst_cs,#cs
            batch_size=opts.val_batch_size, shuffle=True, num_workers=8)

        val_loader_acdc = data.DataLoader(
            val_dst_acdc,#acdc
            batch_size=opts.val_batch_size, shuffle=True, num_workers=8)

    if opts.test_only:#Testing
        test_loader = data.DataLoader(
            tst_dst, batch_size=opts.test_batch_size, shuffle=True, num_workers=8)
        print(f"Dataset: {opts.dataset}, Test set: {len(tst_dst)}")
    else:#Training
        print(f"Dataset: CS+AWSS, Train set cs: {len(train_dst_cs)}, Train set awss: {len(train_dst_awss)},"
              f" Val set cs: {len(val_dst_cs)},Val set acdc: {len(val_dst_cs)}")




    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score_cs": best_score_cs,
            "best_score_acdc": best_score_acdc,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score_cs = 0.0
    best_score_acdc = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if not opts.finetune:
        if opts.ckpt is not None and os.path.isfile(opts.ckpt):
            # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
            checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(model)
            model.to(device)
            if opts.continue_training:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                scheduler.load_state_dict(checkpoint["scheduler_state"])
                cur_itrs = checkpoint["cur_itrs"]
                best_score = checkpoint['best_score']
                print("Training state restored from %s" % opts.ckpt)
            print("Model restored from %s" % opts.ckpt)
            del checkpoint  # free memory
        else:
            print("[!] Retrain")
            model = nn.DataParallel(model)
            model.to(device)
    elif opts.finetune:
        checkpoint = torch.load(opts.pretrained_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_itrs = checkpoint["cur_itrs"]
        best_score_cs = checkpoint['best_score_cs']
        best_score_acdc = checkpoint['best_score_acdc']
        print(f"Fine-tuning model {opts.pretrained_model} on dataset {opts.dataset}")
        # Freeze all but last layer
        for name, param in model.named_parameters():
            print(name)
            if not 'module.classifier.aspp.convs' in name:#module.classifier.classifier.
                param.requires_grad = False

    # ==========   Train Loop   ==========#


    # if opts.test_only:
    #     print("[!] testing")
    #     model.eval()
    #     test_score, ret_samples = validate(
    #         opts=opts, model=model, loader=test_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
    #     print(metrics.to_str(test_score))
    #     return

    if opts.test_only:#testing

        vis_sample_id = None
        # if MODE==11:
        #     vis_sample_id = vis_sample_id_cs
        # elif MODE==21:
        #     vis_sample_id = vis_sample_id_acdc
        print("[!] testing")
        model.eval()
        test_score, ret_samples = validate(
            opts=opts, model=model, loader=test_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(test_score))
        IoU_scores = np.array(list(test_score['Class IoU'].values()))[np.array([0, 1, 2, 5, 6, 7, 8, 10, 11, 13])]
        print(IoU_scores)
        return
    else:#training
        vis_sample_id_cs = np.random.randint(0, len(val_loader_cs), opts.vis_num_samples,
                                             np.int32) if opts.enable_vis else None  # sample idxs for visualization

        vis_sample_id_acdc = np.random.randint(0, len(val_loader_acdc), opts.vis_num_samples,
                                               np.int32) if opts.enable_vis else None  # sample idxs for visualization

    # model = Weather_Classifier().cuda()
    # criterion = nn.CrossEntropyLoss()
    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1

        dataloader_iterator_cs = iter(train_loader_cs)
        dataloader_iterator_awss = iter(train_loader_awss)
        for i, (_, _, _, _, _) in enumerate(train_loader_awss):

            # print(i)
            # try:
            if i%2==0:
                #AWSS
                (images, labels, weather_ids, time_ids,data_domain) = next(dataloader_iterator_awss)

                for name, param in model.named_parameters():
                    # print(name)
                    if 'module.backbone.low_level_features.' in name:
                        param.requires_grad = True

            else:
                #CS
                (images, labels, _, weather_ids, time_ids,data_domain) = next(dataloader_iterator_cs)

                for name, param in model.named_parameters():
                    # print(name)
                    if 'module.backbone.low_level_features.' in name:
                        param.requires_grad = False


            cur_itrs += 1
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            weather_ids = weather_ids.cuda()
            time_ids = time_ids.cuda()

            optimizer.zero_grad()

            outputs,weather_preds,time_preds = tuple(model(images))

            loss_segmentation = criterion(outputs, labels)
            loss_segmentation.backward(retain_graph=True)

            loss_weather = criterion(weather_preds,weather_ids)
            loss_weather = loss_weather*0.00001

            loss_time = criterion(time_preds,time_ids)
            loss_time = loss_time*0.00001


            loss_weather.backward(retain_graph=True)
            loss_time.backward()


            optimizer.step()
            np_loss = loss_segmentation.detach().cpu().numpy()

            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if  (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score_cs, ret_samples_cs = validate(
                    opts=opts, model=model, loader=val_loader_cs, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id_cs)

                val_score_acdc, ret_samples_acdc = validate(
                    opts=opts, model=model, loader=val_loader_acdc, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id_acdc)
                print(metrics.to_str(val_score_cs),metrics.to_str(val_score_acdc))
                if val_score_cs['Mean IoU'] > best_score_cs and val_score_acdc['Mean IoU'] > best_score_acdc:  # save best model
                    best_score_cs = val_score_cs['Mean IoU']
                    best_score_acdc = val_score_acdc['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))


                if vis is not None:  # visualize validation score and samples
                    # vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU CS", cur_itrs, val_score_cs['Mean IoU'])
                    vis.vis_table("[Val] Class IoU CS", val_score_cs['Class IoU'])

                    vis.vis_scalar("[Val] Mean IoU ACDC", cur_itrs, val_score_acdc['Mean IoU'])
                    vis.vis_table("[Val] Class IoU ACDC", val_score_acdc['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples_cs):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst_cs.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst_cs.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)


                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return

if __name__ == '__main__':

    ACDC_classes = ['rain','fog','snow','night']
    opts = []
    # MODE = 0#train on cityscapes and AWSS
    MODE = 11#test on cityscapes
    # MODE = 21  # test on acdc
    if MODE == 21:
        for ACDC_test_class in ACDC_classes:
            main(ACDC_test_class=ACDC_test_class,MODE=MODE)
        exit()
    main(MODE=MODE)

# Expected Output
# =================
# Cityscapes
# ------------
# Overall Acc: 0.939875
# Mean Acc: 0.825837
# FreqW Acc: 0.890178
# Mean IoU: 0.746920
# Per-class IoU: [0.95335767 0.72550871 0.88743945 0.50785172 0.44274712 0.60158601 0.87598127 0.87561615 0.70870125 0.89040882]
# -------------------------------------------------------------------
# ACDC
# ------------
# ACDC (Rain)
# Overall Acc: 0.877963
# Mean Acc: 0.667648
# FreqW Acc: 0.791779
# Mean IoU: 0.566379
# Per-class IoU: [0.76403344 0.36642211 0.72458654 0.31330346 0.32441966 0.40964191 0.81764442 0.9160704  0.40519023 0.62248053]
# ---
# ACDC (Fog)
# Overall Acc: 0.899750
# Mean Acc: 0.697483
# FreqW Acc: 0.826015
# Mean IoU: 0.599675
# Per-class IoU: [0.89968568 0.61364265 0.72526159 0.30592753 0.36195622 0.40860551 0.82659963 0.91106372 0.34895574 0.59505032]
# ---
# ACDC (Snow)
# Overall Acc: 0.813452
# Mean Acc: 0.597240
# FreqW Acc: 0.690171
# Mean IoU: 0.502845
# Per-class IoU: [0.72744211 0.27268236 0.63459407 0.28343183 0.2544351  0.42002753 0.75314912 0.75577437 0.34386931 0.58304499]
# ---
# ACDC (Night)
# Overall Acc: 0.589835
# Mean Acc: 0.365503
# FreqW Acc: 0.423166
# Mean IoU: 0.271261
# Per-class IoU: [0.75924663 0.34913799 0.43090425 0.08629936 0.10791564 0.08757231 0.37398767 0.04645397 0.18224199 0.28884528]