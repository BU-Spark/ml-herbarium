# ## Import Modules

import os
import shutil
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn

from craft_text_detector import Craft

from transformers import (TrOCRProcessor, 
                        VisionEncoderDecoderModel)

# add parent directory to path so that we can import our python scripts from all subdirectories
cwd_prefix = "/projectnb/sparkgrp/ml-herbarium-grp/summer2023/kabilanm/ml-herbarium/trocr/evaluation-dataset/handwritten-typed-text-classification/"
import sys
sys.path.append(cwd_prefix)

import detr
from utils.utils import *

import click

@click.command()
@click.option('--input-dir', type=click.Path(exists=True), required=True, help='Input directory containing the images.')
@click.option('--output-dir', type=click.Path(exists=True), required=True, help='Output directory to save the results.')
@click.option('--cache-dir', type=click.Path(exists=True), default="./data/", required=False, help='Cache directory for pretrained models.')
@click.option('--decoder-path', type=click.Path(exists=True), required=True, help='Path to the downloaded decoder model .pth file.')
@click.option('--delete-intermediate', default=False, is_flag=True, help='Flag to delete intermediate files.')

def main(input_dir, output_dir, cache_dir, decoder_path, delete_intermediate):
    # ## Initialize DETR and CRAFT-Related Directories
    detr_inputdir = input_dir

    # Default paths on SCC (you can change these paths however necessary)
    detr_outputdir = os.path.join(output_dir, 'intermediate_files/')
    output_dir_craft = os.path.join(output_dir, 'doc_classification_input/')

    # Create intermediate directories
    os.makedirs(detr_outputdir)
    os.makedirs(output_dir_craft)

    # Create respective directories for "handwritten" and "typed" images
    os.makedirs(os.path.join(output_dir, "typed"))
    os.makedirs(os.path.join(output_dir, "handwritten"))


    # ## DETR Inference
    # Use the DETR model for inference (adopted from Freddie (https://github.com/freddiev4/comp-vision-scripts/blob/main/object-detection/detr.py))
    detr_model = 'spark-ds549/detr-label-detection'
    # The DETR model returns the bounding boxes of the lables indentified from the images
    label_bboxes = detr.run(detr_inputdir, detr_outputdir, detr_model, cache_dir)


    # ## Initialize CRAFT Model and Get Bounding Boxes
    # initialize the CRAFT model
    craft = Craft(output_dir = output_dir_craft, 
                export_extra = False, 
                text_threshold = .7, 
                link_threshold = .4, 
                crop_type="poly", 
                low_text = .3, 
                cuda = True)

    # CRAFT on images to get bounding boxes
    images = []
    corrupted_images = []
    no_segmentations = []
    boxes = {}
    count= 0
    img_name = []
    box = []
    file_types = (".jpg", ".jpeg",".png")
        
    for filename in tqdm(sorted(os.listdir(detr_outputdir))):
        if filename.endswith(file_types):
            image = detr_outputdir+filename
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


    # ## Initialize Device
    # Move the model to the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device


    # ## Initialize Processor and Models
    # Define model and processor
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-stage1', cache_dir=cache_dir)
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-stage1', cache_dir=cache_dir)

    # Freeze TrOCR layers
    for param in model.parameters():
        param.requires_grad = False

    # Define our custom classifier (also decoder)
    classifier = nn.Sequential(
        
        nn.Conv2d(1, 16, kernel_size=1, stride=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(16, 32, kernel_size=1, stride=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 32, kernel_size=1, stride=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(32 * (577 // 8) * (1024 // 8), 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 1)
    )


    # ## Load Pretrained Classifier
    classifier = torch.nn.DataParallel(classifier, [0]) # list(range(torch.cuda.device_count()))
    classifier.load_state_dict(torch.load(decoder_path))

    # Move Models to Device
    model = model.to(device)
    classifier = classifier.to(device)

    score_sum_dict = defaultdict(lambda: [0, 0]) # file_name: (hw_confidence, typed_confidence)
    score_len_dict = defaultdict(lambda: [0, 0]) # file_name: (hw_count, typed_count)


    # ## Process Each Image and Compute Scores
    for dir_ in os.listdir(output_dir_craft):
        for file in os.listdir(os.path.join(output_dir_craft, dir_)):
            
            key = dir_.split("_")[0]
            
            img = Image.open(output_dir_craft+dir_+"/"+file)
            
            pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
            encoder_outputs = model.encoder(pixel_values)
            
            image_representation = encoder_outputs.last_hidden_state

            classifier.eval()
            with torch.no_grad():
                classifier_output = classifier(image_representation.unsqueeze(1))
                
                pred_confidence = torch.sigmoid(classifier_output)
                predicted = torch.round(pred_confidence)
                
                if(predicted == 0):
                    score_sum_dict[key][0] += 1-pred_confidence
                    score_len_dict[key][0] += 1
                if(predicted == 1):
                    score_sum_dict[key][1] += pred_confidence
                    score_len_dict[key][1] += 1

    score_sum_dict = dict(score_sum_dict)
    score_len_dict = dict(score_len_dict)


    score_avg_dict = defaultdict(lambda: [0, 0])


    # ## Final Scoring
    # aggregating and computing final scores
    hw_score, typed_score = 0, 0

    for sum_, len_ in zip(score_sum_dict.items(), score_len_dict.items()):
        if(len_[1][0] == 0):
            hw_score = 0
        elif(len_[1][1] == 0):
            typed_score = 0
        else:
            hw_score = sum_[1][0]/len_[1][0]
            typed_score = sum_[1][1]/len_[1][1]
        score_avg_dict[sum_[0]] = [hw_score, typed_score]

    score_avg_dict = dict(score_avg_dict)
    score_avg_dict


    # ## Classify Files Based on Scores
    # Here, we copy the images to the respective directories based on the average confidence scores computed for each image.
    for file_name, avg_scores in score_avg_dict.items():
        source_file = detr_inputdir+file_name+".jpg"
        
        # Copy the file using shutil.copy2 to the corresponding directory
        # based on the average prediction score
        if(avg_scores[0] >= avg_scores[1]):
            shutil.copy2(source_file, os.path.join(output_dir, "handwritten"))
            
        # add some bias here
        if(avg_scores[0] < avg_scores[1]):
            shutil.copy2(source_file, os.path.join(output_dir, "typed"))

    if delete_intermediate:
        # Delete the intermediate folders
        shutil.rmtree(output_dir_craft)
        shutil.rmtree(detr_outputdir)


if __name__ == '__main__':
    main()
