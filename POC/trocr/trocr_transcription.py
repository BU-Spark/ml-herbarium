#Imports and installs
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from craft_text_detector import Craft # Need to edit the export_detected_regions function to prepend 0's
import torch
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets
from tqdm import tqdm
import pandas as pd
import shutil
import pickle
import json

import pycountry
import warnings
import time
import argparse

import trocr
import matching

def main(args):
    
    # Suppressing all the huggingface warnings
    SUPPRESS = True
    if SUPPRESS:
        from transformers.utils import logging
        logging.set_verbosity(40)
    # Turning off this warning, isn't relevant for this application
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    # Turning off VisibleDeprecationWarning
    warnings.filterwarnings("ignore")
    # Location of images
    workdir = args.image_folder # update this to the desired directory on scc
    # Location of the segmentations
    output_dir_craft = os.path.join(workdir,'craft_output')
    #if path deosn't exist, create it
    if not os.path.exists(output_dir_craft):
        os.makedirs(output_dir_craft)
    # Location to save all output files
    save_dir = args.save_path
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))
    # For ground truth labels 
    workdir2 = '/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/drago_testdata/gt_labels' # update this to the desired directory on scc
    # Corpus files
    ALL_SPECIES_FILE = args.species_file
    ALL_GENUS_FILE = args.genus_file
    ALL_TAXON_FILE = args.taxon_file

    print('\n')
    # initialize the CRAFT model
    if torch.cuda.is_available():
        CUDA_CRAFT = True
    else:
        CUDA_CRAFT = False
        print("\033[1mNo GPU detected, results may be slow.\033[0m")
    print('Setting up CRAFT model...')
    craft = Craft(output_dir = output_dir_craft,export_extra = False, text_threshold = .7,link_threshold = .4, crop_type="poly",low_text = .3,cuda = CUDA_CRAFT)

    # CRAFT on images to get bounding boxes
    images = []
    corrupted_images = []
    no_segmentations = []
    boxes = {}
    count= 0
    img_name = []
    box = []
    file_types = (".jpg", ".jpeg",".png")
    for filename in tqdm(sorted(os.listdir(workdir)),desc='CRAFTing images'):
        if filename.endswith(file_types):
            image = os.path.join(workdir,filename)
            try:
                img = Image.open(image) 
                img.verify() # Check that the image is valid
                bounding_areas = craft.detect_text(image)
                if len(bounding_areas['boxes']): #check that a segmentation was found
                    images.append(image)
                    boxes[image] = bounding_areas['boxes']
                    
                else:
                    no_segmentations.append(image)
            except (IOError, SyntaxError) as e:
                corrupted_images.append(image)

    print('Setting up Tr-OCR model...')
    # Deleting empty folders, which occurs if some of the images get no segementation from CRAFT
    output_dir_craft = output_dir_craft
    folders = list(os.walk(output_dir_craft))[1:]
    deleted = []
    for folder in folders:
        if not folder[2]:
            deleted.append(folder)
            os.rmdir(folder[0])
            
    # Setting up the Tr-OCR model and processor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") 
    if torch.cuda.is_available():
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

        # Use all available gpu's
        model= nn.DataParallel(model,list(range(torch.cuda.device_count()))).to(device)
    else:
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)
    # Dataloader for working with gpu's
    trainset = datasets.ImageFolder(output_dir_craft, transform = processor)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False)

    # For matching words to image
    filenames = [s.replace('_crops', '') for s in list(trainset.class_to_idx)]

    # For matching the image name with the label name
    word_log_dic = {k: v for k,v in enumerate(filenames)}
    # For matching the image name with the transriptions
    words_identified = {k: [] for v,k in enumerate(filenames)}

    print('Saving intermediate files to {}'.format(save_dir))
    # Save filenames
    with open(os.path.join(save_dir,'filenames.txt'), 'w') as fp:
        for item in filenames:
            # write each item on a new line
            fp.write("%s\n" % item)
    # Save word_log_dic 
    with open(os.path.join(save_dir,'word_log_dic.json'), 'w') as fp:
        json.dump(word_log_dic, fp)
    # Save words_identified
    with open(os.path.join(save_dir,'words_identified.json'), 'w') as fp:
        json.dump(words_identified, fp)

    #Storing the outputs
    results,confidence,labels = trocr.evaluate_craft_seg(model,processor, words_identified,word_log_dic,testloader,device)
    #Saving all the outputs in dataframe
    df = pd.DataFrame(list(zip(results,confidence,labels)),columns = ['Results','Confidence','Labels'])
    df.to_pickle(os.path.join(save_dir,'full_results.pkl'))

    # First part of final csv with results, confidence level from tr-ocr, and label
    combined_df = trocr.combine_by_label(df)

    # Adding the image path and all bounding boxes 

    df_dictionary = pd.DataFrame(boxes.items(), columns=['Image_Path', 'Bounding_Boxes'])
    combined_df = pd.concat([combined_df, df_dictionary], axis=1, join='inner')

    #Save intermediate file
    combined_df.to_pickle(os.path.join(save_dir,'test.pkl'))

    print('Finding bigrams...')
    # Creating a new column which contains all bigrams from the transcription, with an associated index for each bigram
    bigram_df = combined_df.copy()

    bigram_df['Bigrams'] = bigram_df['Transcription'].str.join(' ').str.split(' ')

    bigram_df['Bigrams'] = bigram_df['Bigrams'].apply(lambda lst: [lst[i:i+2] for i in range(len(lst) - 1)]).apply(lambda sublists: [' '.join(sublist) for sublist in sublists])

    bigram_df['Bigram_idx'] = bigram_df.apply(matching.bigram_indices, axis=1)

    # Associating all biagrams with their respective image
    bigram_idx = []
    for i in range(len(bigram_df)):
        for j in range(len(bigram_df.loc[i, 'Bigrams'])):
            bigram_idx.append((i))
    bigram_idx = pd.Series(bigram_idx)

    # Getting the bigrams as individual strings
    results = pd.Series(bigram_df['Bigrams'].explode().reset_index(drop=True))


    species = pd.Series(list(pd.read_pickle(ALL_SPECIES_FILE)))
    genus = pd.Series(list(pd.read_pickle(ALL_GENUS_FILE)))
    taxon = pd.read_csv(ALL_TAXON_FILE,delimiter = "\t", names=["Taxon"]).squeeze()

    # All countries and subdivisions for matching 
    countries = []
    for country in list(pycountry.countries):
        countries.append(country)


    subdivisions_dict = {}
    subdivisions = []
    for subdivision in pycountry.subdivisions:
        subdivisions.append(subdivision.name)
        subdivisions_dict[subdivision.name] = pycountry.countries.get(alpha_2 = subdivision.country_code).name


    #running the matching against all files
    minimum_similarity = .01 #arbitrary, set here to get every prediction, likely want to set this quite a bit higher
    start = time.time()
    # all_matches = matching.pooled_match(results_series,labels,minimum_similarity =minimum_similarity,Taxon = taxon,Species = species,Genus = genus,Countries = countries,Subdivisions = subdivisions)
    all_matches = matching.pooled_match(results,bigram_idx,minimum_similarity =minimum_similarity,Taxon = taxon,Species = species,Genus = genus)
    end = time.time()
    print('Time to match all strings: ',end-start)
        

    # save all_matches pickle
    with open(os.path.join(save_dir,'all_matches.pkl'), 'wb') as f:
        pickle.dump(all_matches, f)

    # Getting the final dataframe with all output information
    final_df = bigram_df.copy()

    for k,v in all_matches.items():
        final_df = pd.merge(final_df,v[['right_index','Predictions','similarity',k+'_Corpus']],how = 'left',
                        left_on = 'Labels', right_on = 'right_index')
        # Rename the predictions, similarity, and corpus columns
        final_df = final_df.rename(columns = {'Predictions':k+'_Prediction_String','similarity':k+'_Similarity',k+'_Corpus':k+'_Prediction'})
        # Drop the right_index column
        final_df = final_df.drop(columns = ['right_index'])
        # Dealing with the case where there is no match
        final_df[k+'_Index_Location'] = [x[0].index(x[1]) if x[1] in x[0] else 'No Match Found' for x in zip(final_df['Bigrams'], final_df[k+'_Prediction_String'])]

    # Save the final dataframe
    final_df.to_pickle(os.path.join(save_dir,'final_df.pkl'))

    # Save the final dataframe as a csv
    final_df.to_csv(os.path.join(save_dir,'final_df.csv'))
    if args.delete_seg:
        # Delete the segmentation folder
        shutil.rmtree(output_dir_craft)
        print('Segmentation folder deleted')
    
    # Print the location of the final dataframe
    print('Final dataframe saved to: ',os.path.abspath(os.path.join(save_dir,'final_df.csv')))

if __name__ == '__main__':
    # Argument parser which takes in the path to image folder, path to save results, path to species, genus, and taxon files
    # and then passes those arguments to the main function
    # Defaults set to work on scc files


    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/TROCR_Training/goodfiles/', help='path to image folder')
    parser.add_argument('--save_path', type=str, default='/projectnb/sparkgrp/colejh/saved_results2/', help='Path to save results')
    parser.add_argument('--species_file', type=str, default='/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/corpus_taxon/output/possible_species.pkl', help='Corpus file for species')
    parser.add_argument('--genus_file', type=str, default='/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/corpus_taxon/output/possible_genus.pkl', help='Corpus file for genus')
    parser.add_argument('--taxon_file', type=str, default='/usr3/graduate/colejh/corpus_taxon.txt', help='Corpus file for taxon')
    parser.add_argument('--delete_seg','-d', default = 'True', help='Delete segmentation files after running')
    args = parser.parse_args()
    main(args)