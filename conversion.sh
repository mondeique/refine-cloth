# background remove source image
cd image-background-remove-tool
python main.py -i ../image_1.jpg -o ../result/

# get mask of source image
cd ../
python get_mask.py -s './result/image_1.png' -r './result/image_mask.png'

# background remove reference image
cd image-background-remove-tool
python main.py -i ../image_2.jpg -o ../result/

# Match Histogram
cd ../
python match_histogram_skimage.py -s './result/image_1.png' -r './result/image_2.png'

# Make Final Result

python make_result.py -m './result/image_mask.png' -r './result/matched.jpg' -o './result/final.jpg'