import os
import shutil
from PIL import Image
import cv2
import numpy as np

image_path = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/tess-test/"
images = os.listdir(image_path)

output_path = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/preprocessing-outputs/tess-test/"

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.makedirs(output_path)

for image in images:
    if ".jpg" not in image:
        continue
    try:
        img = np.array(Image.open(image_path + image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,51,14)
        cv2.imwrite(output_path + image, img)
    except:
        print("Error processing image: " + image)
    