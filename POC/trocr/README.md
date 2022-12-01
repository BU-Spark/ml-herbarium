# Proof of Concept - Deployment Plan

# File Descriptions
This directory contains the 4 files which demonstrate the training and testing of the models we have been working with. 
## 1. trocr_test.ipynb
This notebook shows the process of setting up craft for segmentation, extracting all bounding boxes around text within the image. These segmented images are then passed through the Tr-OCR model to perform text recognition; we save the transcriptions and model confidence for each of these calculations. Then we perform string matching against a set of files (Taxon, Species, Genus, Country, and Subdivisions of countries) in order to find the best match for each.  All of this information is then accrued in a final output dataframe, which contains the bounding boxes and transciptions for every segmentation, as well as the location of the closest match found between all reference files. We then test the accuracy of the pipeline on roughly 1000 images. 
## 2. trocr_train_wab.ipynb
This file is the main training resource for the Tr-OCR model. We have opted to evaluate all available pre-trained models, fine tuning each on the IAM handwriting dataset. All of the training and validation information is logged using Weights and Biases. 

# Model Deployment Files
Also included are the initial deployment scripts for running the trocr pipeline locally. 
Those include:
## 1. trocr_transcription.py
This is the main function for running the pipeline. It takes in 6 command line argument
* the --image_folder arg specifies the path to the folder containing images to test
* The --save_path arg specifies the path to the location where output files should be stored
* The --species_file,--genus_file, and --taxon_file args specify the location of the species, taxon, and genus corpus files
* An optional -d flag to delete any intermediate directories upon program completion (default is true)

## 2. trocr.py
Contains all the functions which relate to running the trocr portion of the pipeline
## 3. matching.py
Contains all of the functions used to match the results from trocr with the species, genus, and taxon information (countries and subdivisions are working on the SCC, but not locally on my mac, will be fixed)
## 4. predictions.py
Contains the functions for printing out the results for each image when testing with ground truth values. Implemented in the included cleaned_trocr_test.ipynb file. 
## 5. results.py
Contains all the helper functions for visualizing the results from trocr_transcription.py. Is currently only implemented in the included cleaned_trocr_test.ipynb file
## 6. utilities.py
Contains a number of functions which are primarily related to the invluded CVIT_Training.py file.
## 7. CVIT_Training.py
This file can be used to train the trocr model on the CVIT dataset. We did not have time to finish the training (each epoch takes roughly 2 and a half days while training on 4 gpu's), thought this would be a good jumping in point for a new team if this project continues in later Spark courses. 
## 8. cleaned_trocr_test.ipynb
This provides the same functionality as the trocr_transcription.py file, but makes the process more transparent by running in a notebook. 
## 9. requirements.txt
All required python installs for running the pipeline

# Deployment instructions
Create a new directory and cd into it

```
mkdir new_directory
cd new_directory
```
Create a new virtual environment and activate it
```
virtualenv venv
source venv/bin/activate
```

Install all required packages
```
pip install -r requirements.txt
```
Run the pipeline, specifying all required arguments
```
python3 trocr_transcription.py --image_folder=<PATH_TO_IMAGE_FOLDER> --save_path=<PATH_TO_SAVE_OUTPUTS> --species_file=<PATH_TO_SPECIES_CORPUS>
--genus_file=<PATH_TO_GENUS_FILE>
--taxon_file=<PATH_TO_TAXON_FILE>
--delete_seg=<TRUE_OR_FALSE>
```