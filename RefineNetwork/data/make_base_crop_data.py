import numpy as np
import os
import cv2
from PIL import Image


raw_data_path = '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/images/'
base_raw_data_path = os.path.join(raw_data_path, 'base')
mask_raw_data_path = os.path.join(raw_data_path, 'mask')

for product in sorted(os.listdir(base_raw_data_path)):
    product_path = os.path.join(base_raw_data_path, product)
    for color in sorted(os.listdir(product_path)):
        color_path = os.path.join(product_path, color)
        for image in sorted(os.listdir(color_path)):
            image_path = os.path.join(color_path, image)
            image_name_split = image.split('.')[0]
            mask_path = image_path.split('base')[0] + 'mask/' + image_path.split('/')[-3] + '/' + \
                        image_path.split('/')[-2] + '/' + image_name_split + '_mask.png'
            print(image_path)
            print(mask_path)
            image = cv2.imread(image_path).astype(np.float32) / 255
            mask = cv2.imread(mask_path).astype(np.float32)
            os.makedirs(f'./dataset/images/base_crop/{product}/{color}', exist_ok=True)
            # multiply mask and base image
            cropped_image = cv2.multiply(image, mask)
            print(cropped_image)
            cv2.imwrite(f'./dataset/images/base_crop/{product}/{color}/{image_name_split}_mask.png', cropped_image)
