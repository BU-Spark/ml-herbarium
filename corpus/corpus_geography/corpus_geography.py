### imports
import wget
import zipfile
import os
import pandas as pd
import shutil


### globals
dataset_url = 'https://api.gbif.org/v1/occurrence/download/request/0230970-210914110416597.zip'
output_dir = '/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-angeline1/ml-herbarium/corpus/corpus_geography/'
zip_file = 'asia_and_oceania.zip'
txt = 'occurrence.txt'
txt_dir = 'asia_and_oceania/occurrence.txt'
csv = 'occurrence.csv'
corpus_geography = 'corpus_geography.txt'


### Download Dataset
# only download the dataset if it is not in the directory (it takes a looooong time to download)
# if os.path.exists(output_dir + zip_file):
#     os.remove(output_dir + zip_file)
# wget.download(dataset_url, out=output_dir + zip_file)


### Extract files from .zip
if os.path.exists(output_dir + 'asia_and_oceania'):
    shutil.rmtree(output_dir + 'asia_and_oceania')
with zipfile.ZipFile(output_dir + zip_file,'r') as zip_ref:
    zip_ref.extractall(output_dir + 'asia_and_oceania') 


### Convert .txt to .csv
read_file = pd.read_csv(output_dir + txt_dir, sep='\t')
read_file.to_csv(output_dir + csv)
df = pd.read_csv(csv, low_memory=False)


higherGeography = df['higherGeography'] 
higherGeography = higherGeography.unique().tolist()
geo_no_semi = []
for i in range(len(higherGeography)): 
    geo_no_semi.append(higherGeography[i].split(';')[:-1])


textfile = open(output_dir + corpus_geography, 'w')
for i in range(len(geo_no_semi)):
    textfile.write(', '.join([str(item) for item in geo_no_semi[i][1:]]) + '\n')
textfile.close()
