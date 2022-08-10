#!/bin/bash -l

#$ -N "CVIT GT"       # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m e               # Send email at end of job
#$ -m a               # Send email at abort of job
#$ -m b                         # Send email at begining of job
# -M "your@email.here" #Be sure to add a money sign after the hashtag

module load python3/3.8.10

python3 /projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/training/CVIT/generate_gt.py
