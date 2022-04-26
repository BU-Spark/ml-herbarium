#!/bin/bash

module load python3/3.8.10
source /usr4/ugrad/en/ml-herbarium/.env/bin/activate
python3 /usr4/ugrad/en/ml-herbarium/transcription/transcribe_labels.py "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/20220425-160006-tesseract" -d