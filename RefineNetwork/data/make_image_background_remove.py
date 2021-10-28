import numpy as np
import os
import cv2
from PIL import Image


# case 1 : cloth image remover

# raw_data_path = '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/clothes/'
# base_raw_data_path = os.path.join(raw_data_path, 'base')

# for product in sorted(os.listdir(base_raw_data_path)):
#    product_path = os.path.join(base_raw_data_path, product)
#    for color in sorted(os.listdir(product_path)):
#        color_path = os.path.join(product_path, color)
#        print(color_path)
#        new_color_path = color_path + '.png'
#        os.system(f"python /home/ubuntu/Desktop/data-conversion/image-background-remove-tool/main.py -i {color_path} "
#                  f"-o {new_color_path}")
#        os.remove(color_path)


# case 2 : real image remover

raw_data_path = '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/images/'
base_raw_data_path = os.path.join(raw_data_path, 'hist_2')

for product in sorted(os.listdir(base_raw_data_path)):
    product_path = os.path.join(base_raw_data_path, product)
    for color in sorted(os.listdir(product_path)):
        color_path = os.path.join(product_path, color)
        for image in sorted(os.listdir(color_path)):
            image_path = os.path.join(color_path, image)
            print(image_path)
            os.system(f"python /home/ubuntu/Desktop/data-conversion/image-background-remove-tool/main.py -i {image_path} "
                      f"-o {image_path}")
