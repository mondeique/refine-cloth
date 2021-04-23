from PIL import Image
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="path to the input source image")
ap.add_argument("-r", "--reference", required=True, help="path to the input reference image")
args = vars(ap.parse_args())


def get_mask(image):
    image_array = np.asarray(image)
    b = (image_array == 0)
    c = b.astype(int)
    c[c != 1] = 0
    c[c == 1] = 255
    return c


image = Image.open(args["source"]).convert('L')
image_mask = get_mask(image)
cv2.imwrite(args["reference"], image_mask)
