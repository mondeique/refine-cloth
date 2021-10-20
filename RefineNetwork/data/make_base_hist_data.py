import numpy as np
import os


raw_data_path = '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/'
image_raw_data_path = os.path.join(raw_data_path, 'images/base_crop/')
cloth_raw_data_path = os.path.join(raw_data_path, 'clothes/base/')

for product in sorted(os.listdir(image_raw_data_path)):
    product_path = os.path.join(image_raw_data_path, product)
    for color in sorted(os.listdir(product_path)):
        color_path = os.path.join(product_path, color)
        for image in sorted(os.listdir(color_path)):
            image_path = os.path.join(color_path, image)
            image_name_split = image_path.split('/')[-1].split('_')[0]
            for base_color in sorted(os.listdir(os.path.join(cloth_raw_data_path, product))):
                base_color_path = os.path.join(cloth_raw_data_path, product, base_color)
                os.makedirs(f'./dataset/images/hist/{product}/{color}/', exist_ok=True)
                os.system(f"python /home/ubuntu/Desktop/data-conversion/match_histogram_skimage.py -s {image_path} -r "
                          f"{base_color_path} -o '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/images/hist/{product}/{color}/'")
