#!/bin/bash -l
#$ -l gpus=1
#$ -j y
#$ -m ea
#$ -l gpu_c=6.0

module load python3/3.8.10
python3 /projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/MNIST/pre-processing.py