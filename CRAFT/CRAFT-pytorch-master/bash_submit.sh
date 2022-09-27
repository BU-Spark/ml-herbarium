#!/bin/bash -l
#$ -l gpus=1
#$ -j y
#$ -m ea
#$ -l gpu_c=6.0

module load python3/3.8.10
module load mxnet/1.7.0
module load pytorch/1.10.2

python test.py --trained_model=craft_mlt_25k.pth --test_folder=/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/20220425-160006/

###if you want to run on cpu, run the below command line###
#python test.py --trained_model=craft_mlt_25k.pth --test_folder=../../in_data/images/..cuda=False
