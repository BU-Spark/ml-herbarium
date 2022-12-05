"""
@file pp_ocr_deployment.py
@author Zhengqi Dong(dong760@bu.edu)
@brief
@date: 2022-11-29
@version 0.1

@copyright Copyright (c) 2022 n

How to run: 
$ python pp_ocr_deployment.py -d /projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/20220425-160006-matching/ -f 912082001.jpg -r /usr4/dl523/dong760/CS549_Herbarium_Project/ml-herbarium/PaddleOCR/output
"""
import os
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

from string_grouper import match_strings, match_most_similar # You have to install 0.5.0 version, 'pip install string_grouper=='

# Multiprocessing stuff
import multiprocessing as mp
NUM_CORES = min(mp.cpu_count(), 50)

# Import PaddleOCR class from paddleocr
from paddleocr import PaddleOCR, draw_ocr



# Some printout texting color
def prRed(skk): print("\033[91m{}\033[00m" .format(skk))
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def batch_evaluation(ocr, data_loader, gt_dict, corpus_set, min_similarity=0.1):
    """Perform batch evaluation
    Args:
        @ocr: a PaddleOCR object, which take cares all detection and recoginition work
        @data_loader: a My_DataLoader object, which haddle images loading and processing
        @gt_dict: a dictionary, Dict<key: imgID, val: image data>
        @corpus_set: a list of three item, [taxon_corpus_set, geography_corpus_set, collector_corpus_set]
        @min_similarity: float, the minimum similarity you want to use for string matching
    Action:
        Performs evaluation on all images in the data_loader.imgIds_list
    """

    total_pred = {'taxon': 0, 'geography': 0, 'collector': 0}
    correct_pred = {'taxon': 0, 'geography': 0, 'collector': 0}
    pred_threshold = 0.8
    min_similarity = 0.1
    str_lst = ['taxon', 'geography', 'collector']
    print(f"taxon_corpus_set: {pd.Series(list(corpus_set[0])).shape}, geography_corpus_set: {pd.Series(list(corpus_set[1])).shape}, collector_corpus_set: {pd.Series(list(corpus_set[2])).shape}")
        
    print(f"Corrected Prediction Format: {color.GREEN} Pred_category(corr_count/total_count): gt_result || predicted_result || imgID {color.END}")
    count = 0
    for imgId in data_loader.imgIds_list:
        if not gt_dict.get(imgId):
            print(f"Warning: Bad image, {imgId} doesn't exist in Ground Truth Label!")
            continue
        start = time.time()
        # Load and fit the image
        img = data_loader[imgId]
        if not isinstance(img, (np.ndarray, list, str, bytes)) or len(img.shape)!=3 or img.shape[-1]!=3:
            continue
        pred_results = ocr.ocr(img)
        
        # Process the predicted result
        pred_results = pred_results[0]
        boxes = [line[0] for line in pred_results] # a list of list, e.g., [[696.0, 26.0], [844.0, 26.0], [844.0, 84.0], [696.0, 84.0]]
        txts = [line[1][0] for line in pred_results] # a list of str, e.g., 'Field'
        scores = [line[1][1] for line in pred_results] # a list of float, e.g., 0.9960

        # If predicted result is none, skip to next one
        if len(txts) == 0:
            print(f"len(txts):, {len(txts)}, pd.Series(txts).shape: {pd.Series(txts).shape}")
            print(f"Warning! No predicted text matched for image {imgId}")
            continue

        # Evaluate the performance
        for i in range(3):
            match_df = match_strings(pd.Series( list(corpus_set[i])), pd.Series(txts), min_similarity = min_similarity, max_n_matches = 1)
            # match_df = matches_above_x(match_df, threshold)
            # Get the one with highest similarity as predicted result
            if not match_df.empty:
                # display(match_df)
                # print(match_df.loc[match_df['similarity'].idxmax()])
                predicted_txt    = match_df.loc[match_df['similarity'].idxmax()]['left_side']
                if predicted_txt in corpus_set[i]:
                    total_pred[str_lst[i]] += 1
                    # gt = gt_series.loc[imgId][str_lst[i]]
                    gt = gt_dict[imgId][str_lst[i]]
                    if predicted_txt == gt:
                        correct_pred[str_lst[i]] += 1
                        print(f"{str_lst[i]+'('+str(correct_pred[str_lst[i]])+'/'+str(total_pred[str_lst[i]])+'):': <15} {gt} || {predicted_txt} || {imgId}")
                    else:
                        print(f"{str_lst[i]: <15}{gt} || {predicted_txt} || {imgId} ")
            else:
                print(f"Ground Truth Not Found for:{imgId}")
        end = time.time()
        print(f"imgId: {imgId}, elapse: {round((end-start)*1e-6, 3)} sec")
        count +=1
        
    print('\n\n********************************Report Statistics')
    acc_lst = [round(correct_pred[i]/total_pred[i], 4) for i in str_lst]
    print(acc_lst)
    print(f"Total Images had been processed for evaluation: {count}")
    print(f"Total Prediction: taxon: {total_pred['taxon']}, geography: {total_pred['geography']}, collector: {total_pred['collector']}")
    print(f"Correct Prediction: taxon: {correct_pred['taxon']}, geography: {correct_pred['geography']}, collector: {correct_pred['collector']}")
    print(f"ACC: taxon: {acc_lst[0]}, geography: {acc_lst[1]}, collector: {acc_lst[2]}")
    print(f"Average Acc: {sum(correct_pred.values())/sum(total_pred.values())}")


def check_images(s_dir, ext_list):
    """A function to check all invalid image
    Args:
        @s_dir: the director of image folder
        @ext_list: a list of good image format in str.
    Example:
        >>> source_dir = r'/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/>>> scraped-data/drago_testdata/image/'
        >>> good_exts=['jpg', 'png', 'jpeg'] # list of acceptable extensions
        >>> bad_file_list, bad_ext_list=check_images(source_dir, good_exts)
        >>> if len(bad_file_list) !=0:
        >>>     print('improper image files are listed below')
        >>>     for i in range (len(bad_file_list)):
        >>>         print (bad_file_list[i])
        >>> else:
        >>>     print(' no improper image files were found')
    """
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
    return bad_images, bad_ext

from paddleocr import PaddleOCR, draw_ocr
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
            # img = Image.open(f_path)
            # img.verify() # Check that the image is valid
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

def display_OCR_result_with_img_path(img_dir, img_name, save_dir=None, display_result=False):
    """
    Args:
        @img_dir: str, the directory that contains this image
        @img_name: str, the name of image file
        @save_dir: str, where the result shuold be saved to?
        @display_result: bool, whether you want to display the result at the end
    """
    # Get img as nparr
    img_dict = {}
    img_dict[img_name] = get_img(f_path=os.path.join(img_dir, img_name))
    display_OCR_result_with_imgID(img_dict, img_name, save_dir, display_result)

def display_OCR_result_with_imgID(img_dict, imgID, save_dir=None, display_result=False):
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
    if imgID not in img_dict.keys():
        print(f"Error, {imgID} doesn't exist in {img_dict.keys()}")
    # Get img as nparr
    img = img_dict[imgID]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Use PaddleOCR to make prediction
    ocr = PaddleOCR(use_angle_cls=True, 
                det_algorithm="DB",          # ['DB', "EAST"]
                rec_algorithm="SVTR_LCNet",  # ['CRNN', 'SVTR_LCNet'], ....Rosetta、CRNN、STAR-Net、RARE
                # det_model_dir=".pretrain_models/",
                # rec_model_dir="",
                use_gpu=False, 
                lang='en',
                show_log=False) # need to run only once to download and load model into memory
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

def display_img_with_imgID(imgID, img_dict):
    """The function will simple used to display an image, for a given imgID where the corresponding image must exist in img_dict, a dictionary.
    Args:
        @img_dict: Dict<int: imgId, nparr: img>, can be used for extracting image 
        @imgID: str, a str format id that must exist in img_dict, for the image to display
    """
    img = img_dict[imgID]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)

def _parse_args():
    """
    Command-line arguments to the system. 

    Return: 
        @args: the parsed args in bundle.

    Example:
        $ python pp_ocr_deployment.py -d /projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/20220425-160006-matching/ -f 912082001.jpg -r /usr4/dl523/dong760/CS549_Herbarium_Project/ml-herbarium/PaddleOCR/output
    """
    parser = argparse.ArgumentParser(description=f"{__name__}")
    parser.add_argument('-d', '--img_dir', type=str, default="src/", required=True, help="What's the directory that this image is stored?")
    parser.add_argument('-f', '--img_name', type=str, default="src/", required=True, help="What the name of this image file?")
    parser.add_argument('-r', '--save_dir', required=True, help="Where you want the result to be saved to?")
    parser.add_argument('--display_result', type=str, default='n', help="Do you want to display the result for this image? [y/n]")
    args = parser.parse_args()
    return args


# SRC_DIR = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/20220425-160006-matching/912082001.jpg"
# save_dir = '/usr4/dl523/dong760/CS549_Herbarium_Project/ml-herbarium/PaddleOCR/output'
from utils import *
if __name__ == '__main__':
    #### gets arguments from command line 
    args = _parse_args()
    print(f"args: {args}")
    if args.display_result.lower() in ('yes', 'true', 't', 'y', '1'):
        display_OCR_result_with_img_path(img_dir=args.img_dir, img_name=args.img_name, save_dir=args.save_dir, display_result=True)
    elif args.display_result.lower() in ('no', 'false', 'f', 'n', '0'):
        display_OCR_result_with_img_path(img_dir=args.img_dir, img_name=args.img_name, save_dir=args.save_dir, display_result=False)
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

