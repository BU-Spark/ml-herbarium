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
    # heavier preprocessing to blur, and threshold images (not to be saved) for reliable angle(skew) measurements
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # find the curves along the images with the same color (boundaries)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
    # traces the rectangle border of each image with a rectangle
    minAreaRect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(minAreaRect)
    # bitmap
    box = np.int0(box)
    angle = cv2.minAreaRect(largest_contour)[-1]
    #angle = - angle
    # dealing with strange errors, 95% accuracy 
    if angle < .09:
        angle = 0.0
    if angle > 5.0:
        angle /= 100
    if angle < -45:
        angle = -(90 + angle)
    
    #print(angle)
    return angle

def rotateimg(img):
    # gets the size of the entire image
    h,w = img.shape[:2]
    center = (w//2,h//2)
    # better than moments based deskewing because there is less error
    moment = cv2.getRotationMatrix2D(center, getangle(img), 1.0) # 1.0 bc do not need to grayscale twice
    # bicubic interpolation bc smoother than bilinear/K-nearest neighbors, interpolates with four kernels, each w/2 and h/2
    rotated = cv2.warpAffine(img, moment, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) 
    return rotated

# if not os.path.exists(output_path):
#     os.makedirs(output_path)
# else:
#     shutil.rmtree(output_path)
#     os.makedirs(output_path)

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
    
import os
import cv2
def check_images( s_dir, ext_list):
    bad_images=[]
    bad_ext=[]
    s_list= os.listdir(s_dir)
    for f in s_list:               
        f_path=os.path.join (s_dir,f)
        index=f.rfind('.')
        ext=f[index+1:].lower()
        if ext not in ext_list:
            print('file ', f_path, ' has an invalid extension ', ext)
            bad_ext.append(f_path)
        if os.path.isfile(f_path):
            try:
                img=cv2.imread(f_path)
                shape=img.shape
                image_contents = tf.io.read_file(f_path)
                image = tf.image.decode_jpeg(image_contents, channels=3)
            except Exception as e:
                print('file ', f_path, ' is not a valid image file')
                print(e)
                bad_images.append(f_path)
        else:
            print('*** fatal error, you a sub directory ', f, ' in class directory ', f)
#         else:
#             print ('*** WARNING*** you have files in ', s_dir, ' it should only contain sub directories')
    return bad_images, bad_ext

source_dir = r'/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/drago_testdata/'
good_exts=['jpg', 'png', 'jpeg'] # list of acceptable extensions
bad_file_list, bad_ext_list=check_images(source_dir, good_exts)
if len(bad_file_list) !=0:
    print('improper image files are listed below')
    for i in range (len(bad_file_list)):
        print (bad_file_list[i])
else:
    print(' no improper image files were found')