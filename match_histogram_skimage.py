from skimage import exposure
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="path to the input source image")
ap.add_argument("-r", "--reference", required=True, help="path to the input reference image")
ap.add_argument("-o", "--output_path", required=True, help="path to the matched result")
args = vars(ap.parse_args())

# load the source and reference images
print("[INFO] loading source and reference images...")
src = cv2.imread(args["source"])
ref = cv2.imread(args["reference"])

product_path = args["source"].split('/')[-2]
color_path = args["source"].split('/')[-1]
color_path = color_path.split('_')[-2]
base_color_path = args["reference"].split('/')[-1]

# determine if we are performing multichannel histogram matching
# and then perform histogram matching itself
print("[INFO] performing histogram matching...")
multi = True if src.shape[-1] > 1 else False
matched = exposure.match_histograms(src, ref, multichannel=multi)

# base_path = '/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/clothes/hist/'
final_path = args["output_path"]
print(final_path)
# show the output images
# cv2.imwrite('./result/src.jpg', src)
# cv2.imwrite('./result/ref.jpg', ref)
cv2.imwrite(final_path, matched)