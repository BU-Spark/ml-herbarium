"""
@file pp_ocr_deployment.py
@author Zhengqi Dong(dong760@bu.edu)
@brief
@date: 2022-11-29
@version 0.1

@copyright Copyright (c) 2022 n

How to run: 
$ python deployment.py -s /projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/20220425-160006-matching/912082001.jpg -r /usr4/dl523/dong760/CS549_Herbarium_Project/ml-herbarium/PaddleOCR/output
"""
import os
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from PIL import Image
import time
import argparse

# Multiprocessing stuff
import multiprocessing as mp
NUM_CORES = min(mp.cpu_count(), 50)

# Import PaddleOCR class from paddleocr
from paddleocr import PaddleOCR, draw_ocr

from utils import *

def get_img(f_path, package="pillow"):
    """This function is use when the dataloader is used as a Dictionary, e.g., my_dataloader[imgID]. So, for given imgID. Retrun the image if success; Return False, otherwise.
    Args:
        @imgID: str representation for image id.
        @package: what package to use? e..g, 'pillow', or 'cv2', usuauly people say cv2 is faster, but harder to use, so use 'pillow' in default you not sure
    Return :
        @img: return an image in nparr if in success, or False otherwise, with 3 dimension, (H, W, C), where C is the number of channel, normally 3 for color image
    """
    try:  # Generally pillow is faster!
        if package=='cv2':
            # With cv2
            img =cv2.imread(f_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif package == "pillow":
            # With Image or PIllow package
#             img = Image.open(f_path)
#             img.verify() # Check that the image is valid
            with Image.open(f_path) as img:
                img.load()
            img = np.asarray(img.convert("RGB"))
        else:
            print(f"Error, not implemented! Abort process")
            os.exit(0)
        # Read more here for other methods of loading images, https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
        return img
    except (IOError, SyntaxError) as e:
        print('file ', f_path, ' is not a valid image file')
        print(e)
        return False


def display_OCR_result_with_img_path(img_path, save_dir=None, display_result=False):
    """
    Args:
        @img_dict: Dict<int: imgId, nparr: img>, can be used for extracting image 
        @imgID: str, a str format id that must exist in img_dict, for the image to display
    Example:
        >>> timestr = time.strftime("%Y%m%d%H%M%S_")
        >>> save_dir = os.path.join(PROJECT_DIR+"output/")+timestr+".jpg"
        # With batch dict
        >>> display_OCR_result_with_imgID(batch_dict, '1019531437', display_result=True)
        # With Single image
        >>> imgID = str(imgID)
        >>> save_dir= '/usr4/dl523/dong760/CS549_Herbarium_Project/ml-herbarium/PaddleOCR/output'
        >>> temp_img_dict = {}
        >>> temp_img_dict[imgID] = data_loader[imgID]
        >>> display_OCR_result_with_imgID(temp_img_dict, imgID, save_dir=save_dir, display_result=False)
    """
    # Get img as nparr
    img = get_img(f_path=img_path)
    
    # Use PaddleOCR to make prediction
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu = False, use_mp=True, show_log=False)
    result = ocr.ocr(img, cls=True)
    result = np.squeeze(np.array(result), axis=0) # Remove redundant dim and transform to nparr, e.g., [1, 19, 4, 2] --> [19, 4, 2]

    # draw result
    from PIL import Image
    print(result)
    print(result.shape)
    # image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(img,   # image(Image|array): RGB image
                       boxes, # boxes(list): boxes with shape(N, 4, 2)
                       txts,  # txts(list): the texts
                       scores, # confidence
                       drop_score=0.1, # drop_score(float): only scores greater than drop_threshold will be visualized
                       font_path='/usr4/dl523/dong760/CS549_Herbarium_Project/ml-herbarium/PaddleOCR/doc/fonts/simfang.ttf') # font_path: the path of font which is used to draw text
    im_show = Image.fromarray(im_show)
    if save_dir:
        timestr = time.strftime("%Y%m%d%H%M%S_")
        save_dir = os.path.join(save_dir, timestr+"_pred_result.jpg")
        im_show.save(save_dir)  # Save the result
    if display_result:
        plt.figure("results_img", figsize=(30,30))
        plt.imshow(im_show)
        plt.show()


def display_img_with_img_path(img_path):
    """The function will simple used to display an image, for a given imgID where the corresponding image must exist in img_dict, a dictionary.
    Args:
        @img_dict: Dict<int: imgId, nparr: img>, can be used for extracting image 
        @imgID: str, a str format id that must exist in img_dict, for the image to display
    """
    img = get_img(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)

def _parse_args():
    """
    Command-line arguments to the system. 

    @return: 
        the parsed args in bundle.
    """
    parser = argparse.ArgumentParser(description=f"{__name__}")
    parser.add_argument('-s', '--src_path', type=str, default="src/", required=True, help="Where is the source path of image?")
    parser.add_argument('-r', '--save_dir', required=True, help="Where you want the result to be saved to?")
    parser.add_argument('-r', '--save_dir', required=True, help="Where you want the result to be saved to?")
    args = parser.parse_args()
    return args


SRC_DIR = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/20220425-160006-matching/912082001.jpg"
save_dir = '/usr4/dl523/dong760/CS549_Herbarium_Project/ml-herbarium/PaddleOCR/output'
if __name__ == '__main__':
    #### gets arguments from command line 
    args = _parse_args()
    print(f"args: {args}")
    display_OCR_result_with_img_path(img_path=args.src_path, save_dir=args.save_dir)
    # display_OCR_result_with_img_path(img_path=SRC_DIR, save_dir=save_dir)
