# activate virtualenv
source activate img2img-translation

# background remove source image
cd ../image-background-remove-tool
python main.py -i /home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/clothes/base/p001/c001.jpg -o /home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/clothes/base/p001/c001.png

# get mask of source image
cd ../
python get_mask.py -s '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/clothes/base/p001/c001.png' -r '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/clothes/mask/p001/c001_mask.png'

# model image segmentation & mask
cd RefineNetwork/data
python render_data_AndreAI.py

# model image 상의부분만 추출한 base crop
python make_base_crop_data.py

# model source image 의 색상 입힌 hist data
python make_base_hist_data.py

cd ../
python test.py
