import os
import shutil
from PIL import Image
import cv2
from cv2 import minAreaRect
import numpy as np
from regex import R
import sys, os
from tqdm import tqdm

image_path = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/tess-test/"
images = os.listdir(image_path)

output_path = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/preprocessing-outputs/tess-test/unrotated/"

def getangle(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
    minAreaRect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(minAreaRect)
    box = np.int0(box)
    angle = cv2.minAreaRect(largest_contour)[-1]
    #angle = - angle
    if angle < .09:
        angle = 0.0
    if angle > 5.0:
        angle /= 100
    if angle < -45:
        angle = -(90 + angle)
    
    #print(angle)
    return angle

def rotateimg(img):
    h,w = img.shape[:2]
    center = (w//2,h//2)
    moment = cv2.getRotationMatrix2D(center, getangle(img), 1.0)
    rotated = cv2.warpAffine(img, moment, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.makedirs(output_path)

for image in tqdm(images):
    if ".jpg" not in image:
        continue
    try:
        img = cv2.imread(image_path + image, 0)
        img = np.array(img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # performed inline cv2.imread
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,51,14)
        img = rotateimg(img)
        cv2.imwrite(output_path + image, img)
    except Exception as e:
        print("Error processing image: " + image)
        type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(type, fname, tb.tb_lineno, e)
    