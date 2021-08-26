import numpy as np
import os
import cv2
from PIL import Image


def get_mask(image):
    image_array = np.asarray(image)
    b = (image_array == 0)
    c = b.astype(int)
    c[c != 1] = 0
    c[c == 1] = 255
    return c

raw_data_path = '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/clothes/'
base_raw_data_path = os.path.join(raw_data_path, 'base')
mask_raw_data_path = os.path.join(raw_data_path, 'mask')

#for product in sorted(os.listdir(base_raw_data_path)):
#    product_path = os.path.join(base_raw_data_path, product)
#    for color in sorted(os.listdir(product_path)):
#        color_path = os.path.join(product_path, color)
#        color_name_split = color.split('.')[0]
#        image = Image.open(os.path.join(color_path)).convert('L')
#        os.makedirs(f'./dataset/clothes/mask/{product}', exist_ok=True)
#        image_mask = get_mask(image)
#        cv2.imwrite(f'./dataset/clothes/mask/{product}/{color_name_split}_mask.png', image_mask)

for product in sorted(os.listdir(mask_raw_data_path)):
    product_path = os.path.join(mask_raw_data_path, product)
    for color in sorted(os.listdir(product_path)):
        color_path = os.path.join(product_path, color)
        print(color_path)
        color_name_split = color.split('_')[0]
        color_split_1 = color_path.split('/')[-2]
        for base_color in sorted(os.listdir(os.path.join(base_raw_data_path, color_split_1))):
            color_split = color_split_1 + '/' + base_color
            base_color_path = os.path.join(base_raw_data_path, color_split)
            print(base_color_path)
            os.makedirs(f'./dataset/clothes/hist/{product}', exist_ok=True)
            os.system(f"python /home/ubuntu/Desktop/data-conversion/match_histogram_skimage.py -s {color_path} -r "
                      f"{base_color_path} -o '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/clothes/hist/{product}/'")
