# imports
import wget
import zipfile
import os
import pandas as pd
import  wget
import pickle


# globals
dataset_url = 'https://hosted-datasets.gbif.org/datasets/backbone/current/backbone.zip'
dataset_output_dir = '/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/synonym-matching/'
output_dir = '/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/synonym-matching/output/'
zip_file = 'backbone.zip'
txt_dir = '/backbone/Taxon.tsv'
txt = 'Taxon.tsv'
synonym = 'synonym.txt'


# download the dataset from the orgin 
# don't run this function unless the dataset is lost cause it takes a loooong time 
def download_dataset(dataset_url, zip_file,run=False):
    if run == True:
        if os.path.exists(dataset_output_dir + zip_file):
            os.remove(dataset_output_dir + zip_file)
        wget.download(dataset_url, out=dataset_output_dir + zip_file)


# generate necessary tools to help extract synonyms
def preprocess_data():
    # extract files from .zip
    if not os.path.exists(dataset_output_dir + 'backbone/'):
        # shutil.rmtree(output_dir + 'backbone/')
        with zipfile.ZipFile(dataset_output_dir + zip_file,'r') as zip_ref:
            zip_ref.extractall(dataset_output_dir)
    # print("extracting files done")

    # create a dictionary called dic for all of the species in the dataset - dic[taxon ID] = [index, scientific name]
    if not os.path.exists(output_dir + 'dic.pkl'):
        df = pd.read_csv(dataset_output_dir + txt_dir, sep='\t', low_memory=False)
        # print("reading tsv file done")
        dic = {}
        for i in range(len(df)):
            dic[df['taxonID'][i]] = [i, df['scientificName'][i]]
        with open(output_dir + 'dic.pkl', 'wb') as f:
            pickle.dump(dic, f)
    else: 
        with open(output_dir + 'dic.pkl', 'rb') as f:
            dic = pickle.load(f)
    # print("dic pkl done")

    # create a dictionary called syn for all pairs of synonyms from the dataset
    # logic of how to find synonyms in the dataset: 
    # first, find a word in the list and if the acceptedNameUsageID is not null, get the ID
    # second, find the name with that ID and loops over until no accepedNameUsageID is found
    # Note: acceptedName is the most updated species name and others are synonyms (each synonym has at most one accepted name)
    # structure: syn[synonym name] = [acceptedName]
    if not os.path.exists(output_dir + 'syn.pkl'):
        syn = {}
        for i in range(len(df)):
            temp = []
            temp.append(df['scientificName'][i])
            while df['acceptedNameUsageID'][i] in dic:
                package = dic[int(df['acceptedNameUsageID'][i])]
                temp.append(package[1])
                i = package[0]
            if len(temp) > 1:
                syn[temp[0]] = temp[1]
        with open(output_dir + 'syn.pkl', 'wb') as f:
            pickle.dump(syn, f) 
    else:
        with open(output_dir + 'syn.pkl', 'rb') as f:
            syn = pickle.load(f)
    # print("syn pkl done")
    return syn

# helper function - chops off the unneccessary words 
# for example: Ribes bracteosum var. fuscescens Jancz. (this is the word from the dataset)
# We just want Ribes bracteosum var. fuscescens and the last word is the collector name, so we chop off any words that do not start
# with a lower cased alphabet after index 0
def process_word(word):
    sep = word.split()
    if len(sep) > 1:
        chop = sep[1:] # the second word
        end = 0
        while ('a' <= chop[0][0] <= 'z'):
            end += 1
            if len(chop) == 1:
                break
            chop = chop[1:]
        i = 1
        temp = sep[0]
        while (i <= end):
            temp += ' ' + sep[i]
            i += 1
    temp = temp.replace(',','') # in case there's any comma
    return temp

# go through every word in synonym dictionary and chops off the unneccessary words
def syn_pure(syn):
    syn_pure = {}
    count = 0
    for key, val in syn.items():
        listt = []
        flag = 0
        if ' ' in key:
            listt.append(key)
            flag += 1
        if ' ' in val:
            listt.append(val)
            flag += 2
        if len(listt) > 0:
            result = map(process_word, listt)
            result = list(result)
        
        if flag == 0: # one word
            continue 
        elif flag == 1:
            syn_pure[result[0].lower()] = val.lower()
        elif flag == 2:
            syn_pure[key.lower()] = result[0].lower()
        else:
            syn_pure[result[0].lower()] = result[1].lower()
        count += 1
        # print("this is number", count, "key is", key, "val is", val)

    with open(output_dir + 'syn_pure.pkl', 'wb') as f:
        pickle.dump(syn_pure, f)
    # print("syn_pure done")
    # print('yayayya all finally done!!!!')

# main function 
def main():
    download_dataset(dataset_url, zip_file,run=False)
    syn = preprocess_data()
    syn_pure(syn)


if __name__ == "__main__":
    main()