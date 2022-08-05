#!/bin/bash -l

#$ -N "Download (contd.) CVIT Dataset"       # Give job a name
#$ -j y               # Merge the error and output streams into a single file

wget -c http://cdn.iiit.ac.in/cdn/ocr.iiit.ac.in/data/dataset/iiit-hws/iiit-hws.tar.gz -P /projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/training/CVIT