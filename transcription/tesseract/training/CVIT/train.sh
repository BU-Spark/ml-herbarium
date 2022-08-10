#!/bin/bash -l

#$ -l h_rt=48:00:00     # Request runtime (up to 48 hours)
#$ -l cpu_type=X5675    # Request CPU type 
#$ -N "CVIT Standalone Train"      # Give job a name
#$ -j y                 # Merge the error and output streams into a single file
#$ -m e                 # Send email at end of job
#$ -m a                 # Send email at abort of job
#$ -m b                 # Send email at begining of job
#$ -M "en@bu.edu"

module load python3/3.10.5
module load leptonica/1.82.0
module load libicu/71.1
module load tesseract/4.1.3

cd /usr4/ugrad/en/tesstrain

make training MODEL_NAME=CVIT DATA_DIR=/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/training/CVIT GROUND_TRUTH_DIR=/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/training/CVIT/Images_90K_Normalized/2 PSM=8 TESSDATA=$HOME/ml-herbarium/transcription/tesseract/tessdata
for folder in {3..88172}
do
    echo "Processing folder $folder"
    make training MODEL_NAME=CVIT START_MODEL=CVIT DATA_DIR=/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/training/CVIT GROUND_TRUTH_DIR=/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/training/CVIT/Images_90K_Normalized/$folder PSM=8 TESSDATA=/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/training/CVIT/CVIT_eng
done
