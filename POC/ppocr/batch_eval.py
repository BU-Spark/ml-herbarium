# Import PaddleOCR class from paddleocr
from paddleocr import PaddleOCR, draw_ocr
from utils import My_DataLoader
import pp_ocr_deployment

# Dataloader for working with gpu's
batch_size= 4
# SRC_DIR = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/drago_testdata/images/"
SRC_DIR = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/20220621-052943"
# SRC_DIR = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/TROCR_Training/goodfiles"
# SRC_DIR = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/20220425-160006-matching"

# Read image in batch
data_loader = My_DataLoader(SRC_DIR, batch_size)
# print(data_loader.imgIds_list)

# Get next batch of image
# batch_dict, label_list = data_loader.get_next_batch()
# print(f" All imgIds read from given path: {label_list}")
# print(f"First image shape: {batch_dict[label_list[0]].shape}")

# Get all gt label
gt_dict = data_loader.get_gt_dict()
# print(gt_dict)

# Get corpuse
all_corpus_set = data_loader.get_all_corpus_set()

# Start OCR detection and recognition
with_gpu = False
model_dir = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/PP_OCR_data/pretrain_models"
ocr = PaddleOCR(use_angle_cls=True, 
            cls_model_dir="./cls/en_ppocr_mobile_v2.0_cls_infer.tar",
            det_algorithm="DB",          # ['DB', "EAST"]
            rec_algorithm="CRNN",  # ['CRNN', 'SVTR_LCNet'], ....Rosetta、CRNN、STAR-Net、RARE
            # det_model_dir=model_dir, # det_mv3_db_v2.0_train, det_r50_drrg_ctw_train.tar
            # rec_model_dir=model_dir,
            use_gpu=with_gpu, 
            lang='en',
            show_log=False) # need to run only once to download and load model into memory
pp_ocr_deployment.batch_evaluation(ocr, data_loader, gt_dict, all_corpus_set, save_result=True)



