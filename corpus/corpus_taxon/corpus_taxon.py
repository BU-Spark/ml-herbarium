### imports
import wget
import zipfile
import os
import pandas as pd
import shutil
import collections
import pickle 


### globals
dataset_url = 'https://hosted-datasets.gbif.org/datasets/ipni.zip'
dataset_dir = '/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/corpus_taxon/'
output_dir = '/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-angeline1/ml-herbarium/corpus/corpus_taxon/'
csv_dir = 'ipni2/ipni.csv'
csv = 'ipni.csv'
zip_file = 'ipni.zip'
corpus_taxon = 'corpus_taxon.txt'
duplicates = 'duplicates_taxon.txt'


### Download Dataset
# if os.path.exists(dataset_dir + zip_file):
#     os.remove(dataset_dir + zip_file)
# wget.download(dataset_url, out=dataset_dir + zip_file)


### Extract .csv from .zip
if os.path.exists(dataset_dir + 'ipni2'):
    shutil.rmtree(dataset_dir + 'ipni2')
with zipfile.ZipFile(dataset_dir + zip_file,'r') as zip_ref:
    zip_ref.extract(member=csv_dir, path=output_dir)
# zf = zipfile.ZipFile(dataset_dir + zip_file)
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
    with open(output_dir + "single.txt", "w") as file:
        dict_possible_species = {}
        dict_possible_genus = {}
        for i in range(len(list_taxon_no_dup)):
            # get rid of single-word taxons
            if ' ' in list_taxon_no_dup[i]:
                output_file.write(list_taxon_no_dup[i] + '\n')
                genus = list_taxon_no_dup[i].split()[0]
                species = "".join(list_taxon_no_dup[i].split()[1:])
            
                if genus not in dict_possible_species:
                    dict_possible_species[genus] = [species]
                else:
                    dict_possible_species[genus] += [species]

                if species not in dict_possible_genus:
                    dict_possible_genus[species] = [genus]
                else:
                    dict_possible_genus[species] += [genus]
            else:
                file.write(list_taxon_no_dup[i] + '\n')
        with open(dataset_dir + 'output/possible_species.pkl', 'wb') as f:
            pickle.dump(dict_possible_species, f) 
        with open(dataset_dir + 'output/possible_genus.pkl', 'wb') as filee:
            pickle.dump(dict_possible_genus, filee)     
file.close()
output_file.close()
print('Finish printing taxon names.')

