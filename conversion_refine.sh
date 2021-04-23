
# background remove source image
cd image-background-remove-tool
python main.py -i ../image_1.jpg -o ../RefineNetwork/data/dataset/test_clothes/base/test/test.png

# get mask of source image
cd ../
python get_mask.py -s './RefineNetwork/data/dataset/test_clothes/base/test/test.png' -r './RefineNetwork/data/dataset/test_clothes/mask/test/test_mask.png'

# background remove reference image
cd image-background-remove-tool
python main.py -i ../image_2.jpg -o ../RefineNetwork/data/dataset/test_clothes/base/test/refe.png

# get mask of reference image
cd ../
python get_mask.py -s './RefineNetwork/data/dataset/test_clothes/base/test/refe.png' -r './RefineNetwork/data/dataset/test_clothes/mask/test/refe_mask.png'

# Main Color injection to image_mask
python match_histogram_skimage.py -s './RefineNetwork/data/dataset/test_clothes/mask/test/test_mask.png' -r './RefineNetwork/data/dataset/test_clothes/base/test/refe.png' -o './RefineNetwork/data/dataset/test_clothes/hist/test/test_refe.jpg'
# Refine Network (shadow, line)
cd RefineNetwork
python test.py

