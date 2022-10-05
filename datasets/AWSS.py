import os
import numpy as np
import random
import collections
import glob
import xml.etree.ElementTree as ET
import torch
import torchvision
from torch.utils import data

from PIL import Image
import cv2


class AWSS(data.Dataset):
    CityscapesClass = collections.namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                                 'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)

    def rgb2mask(self,rgb, rgb_to_index):
        rgb = rgb.numpy()

        # merge classes
        # --rider + person => person
        old_class = (255, 0, 0)
        new_class = (220, 20, 60)
        rgb[np.all(rgb == old_class, axis=-1)] = new_class

        # --vegetation + terrain => vegetation
        old_class = (152, 251, 152)
        new_class = (107, 142, 35)
        rgb[np.all(rgb == old_class, axis=-1)] = new_class

        # --pole + polegroup => pole
        # already combined same color!

        # --car + track + bus + motorcycle + bicycle => car
        old_classes = [(0, 0, 70),(0, 60, 100),(0, 0, 230),(119, 11, 32)]
        new_class = (0, 0, 142)
        for old_class in old_classes:
            rgb[np.all(rgb == old_class, axis=-1)] = new_class


        mask = np.ones((rgb.shape[0], rgb.shape[1]), dtype='uint8')*255
        for k, v in rgb_to_index.items():
           mask[np.all(rgb == v, axis=-1)] = k

        return mask

    # only for synthetic data see get_images_paths_ACDC for real data
    def get_images_paths(self,image_dir, percentage=100, nTrain_samples=None, params=None):
        images = []
        targets = []
        timeOfDay = 'None'
        weather = 'None'
        for filename in glob.iglob(image_dir + '/**/RGB_Shaky_Fr_*.png', recursive=True):
            if filename.find("RGB_Shaky_Fr_") != -1:
                # ------------------------------
                img = filename.replace('/home/kerim/Silver_Project/AWSS/', '')
                temp = img.split('/')
                seq_name, frame_name = temp[0], temp[3]
                xml_path = '/home/kerim/Silver_Project/AWSS/' + seq_name + '/Annotations/Textual/TextualInfo.xml'

                root_node = ET.parse(xml_path).getroot()

                for tag in root_node.findall('StaticInfo/TimeOfDay'):
                    timeOfDay = tag.text.lower()

                for tag in root_node.findall('StaticInfo/WeatherCondition'):
                    weather = tag.text.lower()
            # --------------------------------------------
            if params["include_class_synth_train"] in timeOfDay or params["include_class_synth_train"] in weather:
                images.append(filename)
                img = filename.replace(image_dir+'/', '')
                temp = img.split('/')
                seq_name, frame_name = temp[0], temp[3]
                targets.append(image_dir+'/' + seq_name + '/Annotations/SemanticSeg/' + frame_name.replace("RGB_Shaky",
                                                                                                         "SemSeg"))
        return images, targets

    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
        """
            params

                root : str
                    Path to the data folder
        """

        self.transform = transform
        self.root = root
        self.crop_size = [768,768]
        self.images = []
        self.targets = []

        self.images_root = self.root

        self.rgb_to_index = {
            255: [0, 0, 0],  # unlabeled
            0: [128, 64, 128],  # road
            1: [244, 35, 232],  # sidewalk
            2: [70, 70, 70],  # building
            5: [153, 153, 153],  # pole
            6: [250, 170, 30],  # traffic light
            7: [220, 220, 0],  # traffic sign
            8: [107, 142, 35],  # vegetation
            10: [70, 130, 180],  # sky
            11: [220, 20, 60],  # person
            13: [0, 0, 142]  # car

             }
        params = {}
        params['include_class_synth_train'] = ''
        self.images,self.targets = self.get_images_paths(image_dir= self.images_root,params=params)

# shuffle train/valid/test data
        random.Random(4).shuffle(self.images)
        random.Random(4).shuffle(self.targets)

        if split=='train':
            self.images = self.images[:int(len(self.images)*0.95)]#55
            self.targets = self.targets[:int(len(self.targets) * 0.95)]#55
        elif split=='val':
            self.images = self.images[int(len(self.images)*0.95):]
            self.targets = self.targets[int(len(self.targets) * 0.95):]
        #SHOULD BE REMOVED JUST FOR TEST!!!
        elif split=='test':
            self.images = []
            self.targets = []

        FLAG = True#for HRNet
        if split =="test" and FLAG:
            print("".join("{}	{}\n".format(x.replace("/home/kerim/Silver_Project/AWSS/",''),
                                             y.replace("/home/kerim/Silver_Project/AWSS/",'')) for x, y in zip(self.images, self.targets)))

            pass
    def __len__(self):
        return len(self.images)


    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        return cls.train_id_to_color[target]

    def __getitem__(self, index):

        image = Image.open(self.images[index]).convert('RGB')
        name = self.images[index]

        label = Image.open(self.targets[index])
        label_name = self.targets[index]

        if self.transform:
            image, label = self.transform(image, label)

        label = self.rgb2mask(label, self.rgb_to_index)


# get the weather and time labels
        # ------------------------------
        img = self.images[index].replace('/home/kerim/Silver_Project/AWSS/', '')
        temp = img.split('/')
        seq_name, frame_name = temp[0], temp[3]
        xml_path = '/home/kerim/Silver_Project/AWSS/' + seq_name + '/Annotations/Textual/TextualInfo.xml'

        root_node = ET.parse(xml_path).getroot()

        for tag in root_node.findall('StaticInfo/TimeOfDay'):
            timeOfDay = tag.text.lower()

        for tag in root_node.findall('StaticInfo/WeatherCondition'):
            weather = tag.text.lower()

# normal = 0, rain = 1, fog=2, snow=3
# daytime = 0, night= 1

        if 'normal' in weather:
            weather_id = 0
        elif 'rain' in weather:
            weather_id = 1
        elif 'fog' in weather:
            weather_id = 2
        elif 'snow' in weather:
            weather_id = 3

        if 'day' in timeOfDay:
            time_id = 0
        elif 'night' in timeOfDay:
            time_id = 1


        # print(xml_path,time_id)
        # weather_id = -50
        print("AWSS" + '*' * 20)
        data_domain = 0#0:synthetic, 1:real
        return image, label,weather_id, time_id, data_domain
