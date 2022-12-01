import os
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from PIL import Image
import time

# Multiprocessing stuff
import multiprocessing as mp
NUM_CORES = min(mp.cpu_count(), 50)

# Import PaddleOCR class from paddleocr
from paddleocr import PaddleOCR, draw_ocr

from utils import *


# Dataloader for working with gpu's
batch_size= 16
# SRC_DIR = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/drago_testdata/images/test/"
SRC_DIR = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/20220425-160006-matching"
data_loader = My_DataLoader(SRC_DIR, batch_size)
# batch_dict, label_list = data_loader.get_next_batch()
# print(batch_dict, label_list)


# Making Prediction
## Read here for model inference, https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/inference_ppocr_en.md, and other pretrained model, https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_overview_en.md
ocr = PaddleOCR(use_angle_cls=True, 
                det_algorithm="DB",          # ['DB', "EAST"]
                rec_algorithm="SVTR_LCNet",  # ['CRNN', 'SVTR_LCNet'], ....Rosetta、CRNN、STAR-Net、RARE
                # det_model_dir=".pretrain_models/",
                # rec_model_dir="",
                use_gpu=False, 
                lang='en',
                show_log=False) # need to run only once to download and load model into memory

import pandas as pd
from string_grouper import match_strings, match_most_similar # You have to install 0.5.0 version, 'pip install string_grouper=='
# Building corpus set
# Use set for O(1) look up, and remove duplicated item
TAXON_CORPUS_PATH = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/drago_testdata/full_corpus/corpus_taxon.txt"
GEOGRAPH_CORPUS_PATH = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/drago_testdata/full_corpus/corpus_geography.txt"
COLLECTOR_CORPUS_PATH = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/drago_testdata/corpus/collector_corpus.txt"
taxon_corpus_set     = set(pd.read_csv(TAXON_CORPUS_PATH, delimiter = "\t", names=["Taxon"]).squeeze())
geography_corpus_set = set(pd.read_csv(GEOGRAPH_CORPUS_PATH, delimiter = "\t", names=["Geography"]).squeeze())
collector_corpus_set = set(pd.read_csv(COLLECTOR_CORPUS_PATH, delimiter = "\t", names=["Collector"]).squeeze())

# Building gt_dict
gt_dict = {}
str_lst = ['taxon', 'geography', 'collector']
# GT_LABEL_DIR = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/drago_testdata/gt_labels"
GT_LABEL_DIR = SRC_DIR
Taxon_truth = { line.split(":")[0] : line.split(": ")[1].strip() for line in open(GT_LABEL_DIR + '/taxon_gt.txt') }
Geography_truth = { line.split(":")[0] : line.split(": ")[1].strip() for line in open(GT_LABEL_DIR + '/geography_gt.txt') }
Collector_truth = { line.split(":")[0] : line.split(": ")[1].strip() for line in open(GT_LABEL_DIR + '/collector_gt.txt') }
# comparison_file = {"Taxon":Taxon_truth,"Geography":Geography_truth,"Collector":Collector_truth}
for imgId in data_loader.imgIds_list:
    gt_dict[imgId] = {str_lst[0]: Taxon_truth[imgId], str_lst[1]: Geography_truth[imgId], str_lst[2]: Collector_truth[imgId]}
    
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
    
# 
    
total_pred = {'taxon': 0, 'geography': 0, 'collector': 0}
correct_pred = {'taxon': 0, 'geography': 0, 'collector': 0}
pred_threshold = 0.8
min_similarity = 0.1
corpus_set = [taxon_corpus_set, geography_corpus_set, collector_corpus_set]
str_lst = ['taxon', 'geography', 'collector']
print(f"taxon_corpus_set: {pd.Series(list(corpus_set[0])).shape}, geography_corpus_set: {pd.Series(list(corpus_set[1])).shape}, collector_corpus_set: {pd.Series(list(corpus_set[2])).shape}")
    
print(f"Corrected Prediction Format: {color.GREEN} Pred_category(corr_count/total_count): gt_result || predicted_result || imgID {color.END}")
count = 0
for imgId in data_loader.imgIds_list:
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
        match_df = match_strings(pd.Series(list(corpus_set[i])), pd.Series(txts), min_similarity = min_similarity, max_n_matches = 1)
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



