#!/bin/bash

# activate virtualenv
source activate img2img-translation

#
python clothes_util.py
# background remove source image
#cd ../image-background-remove-tool
#LOCATION=/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/clothes/base/p001/
#FILE=$(find $LOCATION -type f)
#for i in $FILE;
#do
#  NEW_LOCATION=${i}.png
#  python main.py -i ${i} -o ${NEW_LOCATION}
#done
#  NEW_LOCATION=${i}+".png"
#  python main.py -i ${i} -o ${NEW_LOCATION}

## get mask of source image
#cd ../
#python get_mask.py -s '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/clothes/base/p001/c001.png' -r '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/clothes/mask/p001/c001_mask.png'

# model image segmentation & mask
cd /home/ubuntu/Desktop/data-conversion/RefineNetwork/data
python render_data_AndreAI.py

# model image 상의부분만 추출한 base crop
python make_base_crop_data.py

# model source image 의 색상 입힌 hist data
python make_base_hist_data.py

cd ../
python test.py
