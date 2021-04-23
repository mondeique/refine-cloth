from PIL import Image
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mask", required=True, help="path to the source mask image")
ap.add_argument("-r", "--reference", required=True, help="path to the input reference image")
ap.add_argument("-o", "--output", required=True, help="path to the output image path")
args = vars(ap.parse_args())


def make_final(mask, input):
    mask_array = np.asarray(mask)
    image_array = np.asarray(mask)
    b = (image_array == 0)
    c = b.astype(int)
    c[c != 1] = 0
    c[c == 1] = 1
    input_array = np.asarray(input)
    input_array = np.multiply(input_array, c)
    final_array = np.add(mask_array, input_array)
    return final_array


mask = Image.open(args["mask"]).convert('L')
width, height = mask.size
mask = np.reshape(mask, (height, width, 1))
ref = Image.open(args["reference"])
output = make_final(mask, ref)
output = Image.fromarray(output.astype(np.uint8))
output.save(args["output"])

