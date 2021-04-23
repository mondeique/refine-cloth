import os.path
import random

import numpy
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import util.util as util
import cv2


class RefineClothTestDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def make_data_bundles(self, base_image_path):
        path_bundles = []
        for base_path in base_image_path:
            # product_path = os.path.join(base_image_path, base_path)
            components = base_path.split('/')
            print(components)
            # components = [root,clothes,hist,pXXX,cXXX_cXXX.jpg (source_real.jpg)]
            path_bundles.append({
                'matched_image': base_path,
                'real_image': os.path.join(self.root, 'test_clothes', 'base', components[-2], components[-1][-8:-4] + '.png'),
                'real_image_mask': os.path.join(self.root, 'test_clothes', 'mask', components[-2], components[-1][-8:-4] + '_mask.png'),
                'source_image': os.path.join(self.root, 'test_clothes', 'base', components[-2], components[-1][0:4] + '.png'),
                'source_image_mask': os.path.join(self.dir_clothes, 'mask', components[-2], components[-1][0:4] + '_mask.png')
            })

        return path_bundles

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.root = opt.dataroot
        self.batch_size = opt.batch_size
        self.dir_clothes = os.path.join(self.root, 'test_clothes')
        # self.dir_images = os.path.join(self.root, 'images')
        self.dir_clothes_images = os.path.join(self.root, 'test_clothes/hist/')
        self.base_images_path = sorted(make_dataset(self.dir_clothes_images))

        self.train_data_bundle_paths = self.make_data_bundles(self.base_images_path)

        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        train_path = self.train_data_bundle_paths[index]

        # base_image = Image.open(train_path('base_image')).convert('RGB')
        # base_image_mask = Image.open(train_path('base_image_mask')).convert('L')
        # base_cloth = Image.open(train_path('base_cloth')).convert('RGB')
        # base_cloth_mask = Image.open(train_path('base_cloth_mask')).convert('L')
        # input_cloth = Image.open(train_path('input_cloth')).convert('RGB')
        # input_cloth_mask = Image.open(train_path('input_cloth_mask')).convert('L')
        #
        # image_list = [base_image, base_image_mask, base_cloth, base_cloth_mask, input_cloth, input_cloth_mask]
        resized_image_dict = {}
        for key, image in train_path.items():
            if 'mask' in key:
                image = Image.open(image).convert("L")
                new_image = util.expand2square(image, 255)
            else:
                image = Image.open(image).convert("RGB")
                if 'image' in key:
                    new_image = util.expand2square(image, 0)
                else:
                    new_image = util.expand2square(image, 0)
            new_image = new_image.resize((self.opt.loadSize, self.opt.loadSize), Image.LANCZOS)
            new_image = transforms.ToTensor()(new_image)
            resized_image_dict[key] = new_image

        return resized_image_dict

    def __len__(self):
        return len(self.train_data_bundle_paths)

    def name(self):
        return 'RefineClothTestDataset'
