### imports
import wget
import zipfile
import os
import pandas as pd
import shutil
import collections


### globals
dataset_url = 'https://hosted-datasets.gbif.org/datasets/ipni.zip'
output_dir = '/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-angeline1/ml-herbarium/corpus/corpus_taxon/'
csv_dir = 'ipni2/ipni.csv'
csv = 'ipni.csv'
zip_file = 'ipni.zip'
corpus_taxon = 'corpus_taxon.txt'
duplicates = 'duplicates_taxon.txt'


### Download Dataset
if os.path.exists(output_dir + zip_file):
    os.remove(output_dir + zip_file)
wget.download(dataset_url, out=output_dir + zip_file)


### Extract .csv from .zip
if os.path.exists(output_dir + 'ipni2'):
    shutil.rmtree(output_dir + 'ipni2')
with zipfile.ZipFile(output_dir + zip_file,'r') as zip_ref:
    zip_ref.extract(member=csv_dir, path=output_dir)
# zf = zipfile.ZipFile(output_dir + zip_file)
df = pd.read_csv(csv_dir, header=None)


### Export Taxon Name to corpus-taxon.txt
# we need to take care of the duplicates in the dataset 
list_taxon = df[1].to_list()
list_taxon_no_dup = list(set(list_taxon))

# print out the duplicates on duplicates_taxon.txt
dup = [item for item, count in collections.Counter(list_taxon).items() if count > 1]
with open(output_dir + duplicates, "w") as output_file:
    for i in range(len(dup)):
        output_file.write(dup[i] + '\n')
    output_file.close()
print('\n' + 'Finish printing duplicates.')

# print out the taxon names without duplicates 
with open(output_dir + corpus_taxon, "w") as output_file:
    for i in range(len(list_taxon_no_dup)):
        output_file.write(list_taxon_no_dup[i] + '\n')
    output_file.close()
print('Finish printing taxon names.')

