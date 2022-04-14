### imports
import wget
import zipfile
import os
import pandas as pd
import shutil


### globals
dataset_url = 'https://hosted-datasets.gbif.org/datasets/ipni.zip'
output_dir = '/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-angeline1/ml-herbarium/corpus/corpus_taxon/'
csv_dir = 'ipni2/ipni.csv'
csv = 'ipni.csv'
zip_file = 'ipni.zip'
corpus_taxon = 'corpus_taxon.txt'


### Download Dataset
if os.path.exists(output_dir + zip_file):
    os.remove(output_dir + zip_file)
wget.download(dataset_url, out=output_dir + zip_file)


### Extract .csv from .zip
if os.path.exists(output_dir + 'ipni2'):
    shutil.rmtree(output_dir + 'ipni2')
with zipfile.ZipFile(output_dir + zip_file,'r') as zip_ref:
    zip_ref.extract(member=csv_dir, path=output_dir)
#zf = zipfile.ZipFile(output_dir + zip_file)
df = pd.read_csv(csv_dir, header=None)


### Export Taxon Name to corpus-taxon.txt
with open(output_dir + corpus_taxon, "w") as output_file:
    [output_file.write(row + '\n') for row in df[1]]
    output_file.close()

with open(output_dir + corpus_taxon, 'r') as fp:
    x = len(fp.readlines())
    print('\n' + 'Total lines:', x) 

