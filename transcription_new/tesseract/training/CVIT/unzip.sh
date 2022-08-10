#!/bin/bash -l

#$ -N "Unzip CVIT Dataset"       # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m e               # Send email at end of job
#$ -m a               # Send email at abort of job
# -M "your@email.here" #Be sure to add a money sign after the hashtag

tar -xf /projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/training/CVIT/iiit-hws.tar.gz