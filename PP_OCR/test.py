import os
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from PIL import Image

# Multiprocessing stuff
import multiprocessing as mp
NUM_CORES = min(mp.cpu_count(), 50)

# Import PaddleOCR class from paddleocr
from paddleocr import PaddleOCR, draw_ocr


PROJECT_DIR = "/usr4/dl523/dong760/CS549_Herbarium_Project/ml-herbarium/PaddleOCR/"
# Modify the default directory where Aistudio code runs to /home/aistudio/
os.chdir(PROJECT_DIR)
DATASET_PATH = img_dir = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/drago_testdata/" # 20220616-061434/ or 20220425-160006/ or drago_testdata/"

img_dict = {}
def addImg(fIdx):
    """Adding image to the img_dict, a Dict<int: imgId, nparr: img>, based on the given img_ID
    """
    img_dict[fIdx]=cv2.imread(os.path.join(DATASET_PATH, fIdx+".jpg"))
    # 	# By default, cv2.imread() use BGR, if you want to read in grayscale use, cv2.imread('image_1.png', 1)
    # 	temp_img = cv2.imread(os.path.join(DATASET_PATH, fIdx+".jpg"))
    # 	# convert image form BGR to RGB color for matplotlib
    # 	img_dict[fIdx] = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
    return img_dict

def getOrigImgs(multi_proc=True):
    print("Getting original images...")
    print("Starting multiprocessing...")
    file_list = sorted(os.listdir(DATASET_PATH))
    imgIds_list = [img[:-4] for img in file_list if ".jpg" in img]
    if multi_proc:
        pool = mp.Pool(NUM_CORES)
        for item in tqdm(pool.imap(addImg, imgIds_list), total=len(imgIds_list)):
            img_dict.update(item)
        pool.close()
        pool.join()
    else:
        for fIdx in tqdm(imgIds_list):
            img_dict[fIdx]=cv2.imread(os.path.join(DATASET_PATH, fIdx+".jpg"))
    print("\nOriginal images obtained.\n")


def display_OCR_result_with_imgID(imgID):
    """
    Args:
        @imgID: str, a str format id for the image to display
    Example:
        >>> timestr = time.strftime("%Y%m%d%H%M%S_")
        >>> save_path = os.path.join(PROJECT_DIR+"output/")+timestr+".jpg"
        >>> display_OCR_result_with_imgID('1228540653', save_path)
    """
    # Get img as nparr
    img = img_dict[imgID]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Use PaddleOCR to make prediction
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu = False, use_mp=True)
    result = ocr.ocr(img, cls=True)
    result = np.squeeze(np.array(result), axis=0) # Remove redundant dim and transform to nparr, e.g., [1, 19, 4, 2] --> [19, 4, 2]

    # draw result
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
                       font_path='./doc/fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    if save_path:
        im_show.save(save_path)  # Save the result

# Loading images into a Dict<int: imgId, nparr: img>

getOrigImgs()
img_list = list(img_dict.items())	# a list of Tuple<int: imgId, nparr: img>


# Making Prediction
ocr = PaddleOCR(use_angle_cls=True, 
                det_algorithm="DB",          # ['DB', "EAST"]
                rec_algorithm="SVTR_LCNet",  # ['CRNN', 'SVTR_LCNet'], ....Rosetta、CRNN、STAR-Net、RARE
                # det_model_dir=".pretrain_models/",
                # rec_model_dir="",
                use_gpu=False, 
                lang='en') # need to run only once to download and load model into memory

# Making prediction
imgID_list = list(img_dict.keys())
predition_dict = {} # key: imgID, val: (predicted_bboes, predicted_res)
def inference_all_imgs(img_list):
    """
    Args:
        @img_list: a list of <str: imgId, nparr: img_arr>
    """
    predicted_result = []
    for imgId, img in img_list:
        # 3. Perform prediction
        predition_dict[imgId] = ocr.ocr(img) 
        # return a list of [boxes, (pred_text, conf)], e.g., [[[696.0, 26.0], [844.0, 26.0], [844.0, 84.0], [696.0, 84.0]], ('Field', 0.9904948472976685)]
inference_all_imgs(img_list)


# Building corpus set: Use set for O(1) look up, and remove duplicated item
taxon_corpus_set     = set(pd.read_csv(DATASET_PATH +'/taxon_corpus.txt', delimiter = "\t", names=["Taxon"]).squeeze())
geography_corpus_set = set(pd.read_csv(DATASET_PATH +'/geography_corpus.txt', delimiter = "\t", names=["Geography"]).squeeze())
collector_corpus_set = set(pd.read_csv(DATASET_PATH +'/collector_corpus.txt', delimiter = "\t", names=["Collector"]).squeeze())

# Building gt_dict
gt_dict = {}
str_lst = ['taxon', 'geography', 'collector']
Taxon_truth = { line.split(":")[0] : line.split(": ")[1].strip() for line in open(DATASET_PATH + '/taxon_gt.txt') }
Geography_truth = { line.split(":")[0] : line.split(": ")[1].strip() for line in open(DATASET_PATH + '/geography_gt.txt') }
Collector_truth = { line.split(":")[0] : line.split(": ")[1].strip() for line in open(DATASET_PATH + '/collector_gt.txt') }
for imgId in predition_dict.keys():
    gt_dict[imgId] = {str_lst[0]: Taxon_truth[imgId], str_lst[1]: Geography_truth[imgId], str_lst[2]: Collector_truth[imgId]}

    
# Let's first save those temporary gt_result and predicted_result into a pickle file
import pickle

PP_OCR_output_dir = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/PP_OCR_data/"
# Save groud truth result
gt_name = 'gt_result.pkl'
dbfile = open(PP_OCR_output_dir+gt_name, 'ab')
pickle.dump(gt_dict, dbfile)
dbfile.close()
df = pd.DataFrame.from_dict(gt_dict, orient='index').rename(columns={0: 'taxon',1: 'geography',2: 'collector'})
df.to_pickle(PP_OCR_output_dir+gt_name)
# Save predicted result
predicted_name = 'predicted_result.pkl'
df = pd.DataFrame.from_dict(predition_dict, orient='index')
df.to_pickle(PP_OCR_output_dir+predicted_name)



# Perform string matching
from string_grouper import match_strings, match_most_similar

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
    
    
total_pred = {'taxon': 0, 'geography': 0, 'collector': 0}
correct_pred = {'taxon': 0, 'geography': 0, 'collector': 0}
pred_threshold = 0.8
min_similarity = 0.1
corpus_set = [taxon_corpus_set, geography_corpus_set, collector_corpus_set]
str_lst = ['taxon', 'geography', 'collector']

    
print(f"Corrected Prediction Format: {color.GREEN} Pred_category(corr_count/total_count): gt_result || predicted_result || imgID {color.END}")
for imgId, pred_results in predition_dict.items():
    # image = Image.open(img_path).convert('RGB')
    pred_results = pred_results[0]
    boxes = [line[0] for line in pred_results] # a list of list, e.g., [[696.0, 26.0], [844.0, 26.0], [844.0, 84.0], [696.0, 84.0]]
    txts = [line[1][0] for line in pred_results] # a list of str, e.g., 'Field'
    scores = [line[1][1] for line in pred_results] # a list of float, e.g., 0.9960
    
    for i in range(3):
        match_df = match_strings(pd.Series(list(corpus_set[i])), pd.Series(txts), min_similarity = min_similarity, max_n_matches = 1)
        if not match_df.empty: 
            # Get the one with highest similarity as predicted result
            predicted_txt    = match_df.loc[match_df['similarity'].idxmax()]['left_side']
            if predicted_txt in corpus_set[i]:
                total_pred[str_lst[i]] += 1
                gt = gt_dict[imgId][str_lst[i]]
                if predicted_txt == gt:
                    correct_pred[str_lst[i]] += 1
                    print(f"{color.GREEN}{str_lst[i]+'('+str(correct_pred[str_lst[i]])+'/'+str(total_pred[str_lst[i]])+'):': <15} {gt} || {predicted_txt} || {imgId}  {color.END}")
                else:
                    print(f"{str_lst[i]: <15}{gt} || {predicted_txt} || {imgId} ")
        else:
            print(f"{color.RED}Ground Truth Not Found for:{imgId}{color.END}")
    
print('\n\n********************************Report Statistics')
acc_lst = [round(correct_pred[i]/total_pred[i], 4) for i in str_lst]
print(acc_lst)
print(f"{color.BOLD}Total Images had been processed for evaluation: {len(predition_dict.keys())}{color.END}")
print(f"{color.BOLD}Total Prediction: taxon: {total_pred['taxon']}, geography: {total_pred['geography']}, collector: {total_pred['collector']}{color.END}")
print(f"{color.BOLD}Correct Prediction: taxon: {correct_pred['taxon']}, geography: {correct_pred['geography']}, collector: {correct_pred['collector']}{color.END}")
print(f"{color.BOLD}ACC: taxon: {acc_lst[0]}, geography: {acc_lst[1]}, collector: {acc_lst[2]}{color.END}")
print(f"{color.BOLD}Average Acc: {sum(correct_pred.values())/sum(total_pred.values())}{color.END}")
