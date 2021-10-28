# background remove source image
cd ../image-background-remove-tool
python main.py -i ../RefineNetwork/data/dataset/clothes/base/input_clothes.jpg -o ../RefineNetwork/data/dataset/clothes/base/

# get mask of source image
cd ../
python get_mask.py -s './RefineNetwork/data/dataset/clothes/base/input_clothes.png' -r './RefineNetwork/data/dataset/clothes/mask/input_clothes_mask.png'

# model image segmentation & mask
cd RefineNetwork/data
python render_data_AndreAI.py

# model image 상의부분만 추출한 base crop
python make_base_crop_data.py

# model source image 의 색상 입힌 hist data
python make_base_hist_data.py

cd ../
python test.py
