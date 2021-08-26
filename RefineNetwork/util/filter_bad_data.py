import os
import shutil

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy


def make_folder(target_root, product, color):
    try:
        if not os.path.exists(os.path.join(target_root, 'images', 'segmentation', product, color)):
            os.makedirs(os.path.join(target_root, 'images', 'segmentation', product, color))
            os.makedirs(os.path.join(target_root, 'clothes', 'base', product), exist_ok=True)
            os.makedirs(os.path.join(target_root, 'clothes', 'mask', product), exist_ok=True)
            os.makedirs(os.path.join(target_root, 'images', 'base', product, color), exist_ok=True)
            os.makedirs(os.path.join(target_root, 'images', 'mask', product, color), exist_ok=True)
    except OSError:
        print('Already Created')

data_origin_root = '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset'
dataroot = '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/images/segmentation'
target_root = '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/junk_data'
products = sorted(os.listdir(dataroot))
plt.show()
for i, product in enumerate(products):
    colors = sorted(os.listdir(os.path.join(dataroot, product)))
    for j, color in enumerate(colors):
        items = sorted(os.listdir(os.path.join(dataroot, product, color)))
        for k, item in enumerate(items):
            base = mpimg.imread(os.path.join(data_origin_root,'images','base', product, color, item[:-3] + 'jpg'))
            mask = mpimg.imread(os.path.join(data_origin_root,'images','mask', product, color, item[:-4] + '_mask.png'))
            mask = numpy.reshape(mask, (mask.shape[0], mask.shape[1], 1))
            img = numpy.multiply(base, mask)
            fig = plt.figure(figsize=(10,10))
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow((img).astype(numpy.uint8))
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.imshow((base).astype(numpy.uint8))
            plt.show()
            print(f'product {product}  진행 현황 {i}/{len(products)} \nproduct {color}  진행 현황 {j}/{len(colors)} \n product {item}  진행 현황 {k}/{len(items)} ')
            opinion = input('좋은 데이터라면 그냥 enter키를, 지우려면 x 을 입력후 enter키를 입력해주세요')
            if opinion == 'x':
                make_folder(target_root, product, color)
                shutil.move(os.path.join(dataroot, product, color, item), os.path.join(target_root, 'images', 'segmentation', product, color, item))
                # shutil.move(os.path.join(data_origin_root, 'clothes', 'base', product, color), os.path.join(target_root, 'clothes', 'base', product, color))
                # shutil.move(os.path.join(data_origin_root, 'clothes', 'mask', product, color + '_mask.png'), os.path.join(target_root, 'clothes', 'mask', product, color + '_mask.png'))
                shutil.move(os.path.join(data_origin_root, 'images', 'base', product, color, item[:-3] + 'jpg'), os.path.join(target_root, 'images', 'base', product, color, item[:-3] + 'jpg'))
                shutil.move(os.path.join(data_origin_root, 'images', 'mask', product, color, item[:-4] + '_mask.png'), os.path.join(target_root, 'images', 'mask', product, color, item[:-4] + '_mask.png'))
            else:
                print('제대로 입력해줘잉')

            plt.close()

