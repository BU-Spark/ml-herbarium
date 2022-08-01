### imports
import wget
import zipfile
import os
import pandas as pd
import collections
import pickle 


### globals
dataset_url = 'https://hosted-datasets.gbif.org/datasets/ipni.zip'
dataset_dir = '/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/corpus_taxon/'
output_dir = '/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-angeline1/ml-herbarium/corpus/corpus_taxon/'
tsv = "Name.tsv"
zip_file = 'ipni.zip'
corpus_taxon = 'corpus_taxon.txt'
duplicates = 'duplicates_taxon.txt'


### Download Dataset
# [WARNING] only run this if the dataset is too old 
# if os.path.exists(dataset_dir + zip_file):
#     os.remove(dataset_dir + zip_file)
# wget.download(dataset_url, out=dataset_dir + zip_file)


### Extract .tsv from .zip
if os.path.exists(dataset_dir + tsv):
    os.remove(dataset_dir + tsv)
with zipfile.ZipFile(dataset_dir + zip_file,'r') as zip_ref:
    zip_ref.extract(member=tsv, path=dataset_dir)
df = pd.read_csv(dataset_dir + tsv, sep="\t", dtype="object")


### Export Taxon Name to corpus-taxon.txt
# we need to take care of the duplicates in the dataset 
list_taxon = []
for i in range(len(df)):
    if df["col:rank"][i] == "spec.":
        list_taxon.append(df["col:scientificName"][i])
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
    dict_possible_species = {}
    dict_possible_genus = {}
    for i in range(len(list_taxon_no_dup)):
        output_file.write(list_taxon_no_dup[i] + '\n')
        genus = list_taxon_no_dup[i].split()[0].lower()
        species = "".join(list_taxon_no_dup[i].split()[1:]).lower()
        # generate a dictionary of possible species for each genus
        if genus not in dict_possible_species:
            dict_possible_species[genus] = [species]
        else:
            # do not include duplicated species 
            if species not in dict_possible_species[genus]:
                dict_possible_species[genus] += [species]
        # generate a dictionary of possible genera for each species
        if species not in dict_possible_genus:
            dict_possible_genus[species] = [genus]
        else:
            # do not include duplicated geneus
            if genus not in dict_possible_genus[species]:
                dict_possible_genus[species] += [genus]
        
    with open(dataset_dir + 'output/possible_species.pkl', 'wb') as f:
        pickle.dump(dict_possible_species, f) 
    with open(dataset_dir + 'output/possible_genus.pkl', 'wb') as filee:
        pickle.dump(dict_possible_genus, filee) 
output_file.close()
print('Finish printing taxon names.' + '\n')

