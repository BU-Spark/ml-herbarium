#!/bin/bash -l

#$ -N "transcribe_labels.py"       # Give job a name
#$ -j y               # Merge the error and output streams into a single file

module load python3/3.10.5
source /usr4/ugrad/en/ml-herbarium/.env/bin/activate
python3 /usr4/ugrad/en/ml-herbarium/transcription/transcribe_labels.py "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/tess-test" -d