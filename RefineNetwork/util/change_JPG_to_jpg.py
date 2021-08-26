import os

data_origin_root = '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/raw_data/'
for product in sorted(os.listdir(data_origin_root)):
    product_path = os.path.join(data_origin_root, product)
    for color in sorted(os.listdir(product_path)):
        color_path = os.path.join(product_path, color)
        for name in sorted(os.listdir(color_path)):
            if 'png' in name or 'jpg' in name or 'JPG' in name:
                if 'JPG' in name:
                    print(name)
                    new_name = name.split('.')[0] + '.jpg'
                    print(new_name)
                    os.rename(os.path.join(color_path, name), os.path.join(color_path, new_name))
            else:
                images_path = os.path.join(color_path, name)
                for poses in sorted(os.listdir(images_path)):
                    if 'JPG' in poses:
                        print(poses)
                        new_poses = poses.split('.')[0] + '.jpg'
                        print(new_poses)
                        os.rename(os.path.join(images_path, poses), os.path.join(images_path, new_poses))