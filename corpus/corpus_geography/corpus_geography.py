### imports
import wget
import zipfile
import os
import pandas as pd
import shutil


### globals
dataset_asia_url = 'https://api.gbif.org/v1/occurrence/download/request/0230970-210914110416597.zip'
dataset_northamerica_url = 'https://api.gbif.org/v1/occurrence/download/request/0232563-210914110416597.zip'
# else: europe, south america, antarctica and africa
dataset_else_url = 'https://api.gbif.org/v1/occurrence/download/request/0232644-210914110416597.zip'

output_dir = '/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-angeline1/ml-herbarium/corpus/corpus_geography/'

zip_file_asia = 'asia_and_oceania.zip'
zip_file_northamerica = 'north_america.zip'
zip_file_else = 'else.zip'

txt_asia_dir = 'asia_and_oceania/occurrence.txt'
txt_northamerica_dir = 'north_america/occurrence.txt'
txt_else_dir = 'else/occurrence.txt'

csv_asia = 'occurrence_asia.csv'
csv_northamerica = 'occurrence_north_america.csv'
csv_else = 'occurrence_else.csv'

corpus_geography = 'corpus_geography.txt'


def download_dataset(dataset_url, zip_file):
    if os.path.exists(output_dir + zip_file):
        os.remove(output_dir + zip_file)
    wget.download(dataset_url, out=output_dir + zip_file)


def process_geo(zip_file, txt_dir, csv):
    # Extract files from .zip¶
    if os.path.exists(output_dir + zip_file[:-4]):
        shutil.rmtree(output_dir + zip_file[:-4])
    with zipfile.ZipFile(output_dir + zip_file,'r') as zip_ref:
        zip_ref.extractall(output_dir + zip_file[:-4])
        
    # convert text file to csv format 
    read_file = pd.read_csv(output_dir + txt_dir, sep='\\t', engine='python')
    read_file.to_csv(output_dir + csv)
    df = pd.read_csv(csv, low_memory=False)
    
    # take higher geography column from the csv file 
    higherGeography = df['higherGeography'] 
    higherGeography = higherGeography.unique().tolist() # avoid duplicates
    geo_no_semi = []
    for i in range(len(higherGeography)): 
        geo_no_semi.append(higherGeography[i].split(';')[:-1])
    total_cities.append(geo_no_semi)


def write_geo(total_cities):
     # write them to a text file
    textfile = open(output_dir + corpus_geography, 'w')
    for i in range(len(total_cities)):
        for j in range(len(total_cities[i])):
            textfile.write(', '.join([str(item) for item in total_cities[i][j][1:]]) + '\n')
    textfile.close()


total = [[dataset_asia_url, zip_file_asia, txt_asia_dir, csv_asia], [dataset_northamerica_url, zip_file_northamerica,txt_northamerica_dir, csv_northamerica], [dataset_else_url, zip_file_else, txt_else_dir, csv_else]]
total_cities = []
for i in range(len(total)):
    dataset_url = total[i][0]
    zip_file = total[i][1]
    txt_dir = total[i][2]
    csv = total[i][3]
#     download_dataset(dataset_url, zip_file)
    process_geo(zip_file, txt_dir, csv)
    print('done' , i)

write_geo(total_cities)
print('done writing')


with open(output_dir + corpus_geography, 'r') as fp:
    x = len(fp.readlines())
    print('Total lines:', x)

