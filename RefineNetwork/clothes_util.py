import os

raw_data_path = '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/clothes/base/p001'
for color in sorted(os.listdir(raw_data_path)):
    color_path = os.path.join(raw_data_path, color)
    png_name = color_path.split('.')[0] + ".png"
    color_name = color_path.split('/')[-1].split('.')[0]
    mask_name = "/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/clothes/mask/p001/" + color_name + "_mask.png"

    os.system(
        f"python /home/ubuntu/Desktop/data-conversion/image-background-remove-tool/main.py -i {color_path} -o {png_name}")
    os.system(f"python /home/ubuntu/Desktop/data-conversion/get_mask.py -s {png_name} -r {mask_name}")


