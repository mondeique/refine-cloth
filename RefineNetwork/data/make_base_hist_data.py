import numpy as np
import os
import cv2

from colorthief import ColorThief


def simple_hist(image_path, base_color_path, final_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    result = np.zeros((image.shape[0], image.shape[1], 3))
    color_thief = ColorThief(base_color_path)
    dominant_color = color_thief.get_color(quality=1)
    final_color = tuple(reversed(dominant_color))
    result[image == 255] = final_color
    cv2.imwrite(final_path, result)

    return result


## 단순 input clothes 의 color 를 hex 값으로 가져와서 mask 부분에 있는 옷을 입혀버리자.

raw_data_path = '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/'
image_raw_data_path = os.path.join(raw_data_path, 'images/mask/')
cloth_raw_data_path = os.path.join(raw_data_path, 'clothes/base/')

for product in sorted(os.listdir(image_raw_data_path)):
    print(f"NOW IS {product}")
    product_path = os.path.join(image_raw_data_path, product)
    for color in sorted(os.listdir(product_path)):
        color_path = os.path.join(product_path, color)
        for image in sorted(os.listdir(color_path)):
            image_path = os.path.join(color_path, image)
            image_name_split = image_path.split('/')[-1].split('_')[0]
            for base_color in sorted(os.listdir(os.path.join(cloth_raw_data_path, product))):
                base_color_path = os.path.join(cloth_raw_data_path, product, base_color)
                os.makedirs(f'./dataset/images/hist/{product}/{color}/', exist_ok=True)
                final_path = f'./dataset/images/hist/{product}/{color}/' + image_name_split + '_' + base_color
                simple_hist(image_path, base_color_path, final_path)

