#!/bin/bash -l

#$ -N tesstrain       # Give job a name
#$ -j y               # Merge the error and output streams into a single file

module load python3/3.8.10
module load leptonica/1.82.0
module load libicu/71.1
module load tesseract/4.1.3

source /usr4/ugrad/en/ml-herbarium/.env/bin/activate

cd /usr4/ugrad/en/tesstrain
make training MODEL_NAME=IAM_lines_standalone DATA_DIR=/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training GROUND_TRUTH_DIR=/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/training/IAM/gt/lines PSM=7 TESSDATA=$HOME/ml-herbarium/transcription/handwriting_tesseract_training/tessdata
make training MODEL_NAME=IAM_words_standalone DATA_DIR=/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training GROUND_TRUTH_DIR=/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/training/IAM/gt/words PSM=8 TESSDATA=$HOME/ml-herbarium/transcription/handwriting_tesseract_training/tessdata
make training MODEL_NAME=IAM_sentences_standalone DATA_DIR=/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training GROUND_TRUTH_DIR=/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/training/IAM/gt/sentences PSM=7 TESSDATA=$HOME/ml-herbarium/transcription/handwriting_tesseract_training/tessdata