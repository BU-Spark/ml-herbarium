# Proof of Concept - Deployment Plan

# File Descriptions
This directory contains all of the files associated with the Tr-OCR pipeline. 
## 1. trocr_test.ipynb
This notebook shows the process of setting up craft for segmentation, extracting all bounding boxes around text within the image. These segmented images are then passed through the Tr-OCR model to perform text recognition; we save the transcriptions and model confidence for each of these calculations. Then we perform string matching against a set of files (Taxon, Species, Genus, Country, and Subdivisions of countries) in order to find the best match for each.  All of this information is then accrued in a final output dataframe, which contains the bounding boxes and transciptions for every segmentation, as well as the location of the closest match found between all reference files. We then test the accuracy of the pipeline on roughly 1000 images. 
## 2. trocr_train_wab.ipynb
This file is the main training resource for the Tr-OCR model. We have opted to evaluate all available pre-trained models, fine tuning each on the IAM handwriting dataset. All of the training and validation information is logged using Weights and Biases. 
## 3. CVIT_GPU_TRAINING.ipynb
This notebook is an implementation of the huggingface accelerate library in order to more easily use multiple gpus to train on the CVIT dataset. While the character error rate and loss achieved using this training were extremely low (character error rate drops below .00), it did not translate into an accuracy improvement when running the pipeline. It actually made the results significantly worse, as it does well predicting a single word, but past that, it struggles to properly transcribe anything. 
## 4. CVIT_Training.py
This file can be used to train the trocr model on the CVIT dataset as well. It does not implement the accelerate library to implement the multi-gpu training, but can be used to submit batch jobs for training if necessary. 

## 5. cleaned_trocr_test.ipynb
This provides the same functionality as the trocr_transcription.py file, but makes the process more transparent by running in a notebook, and also includes sections that visualize the final results (images with bounding box overlays).
# Model Deployment Files
Also included are the deployment scripts for running the trocr pipeline locally. 
Those include:
## 1. trocr_transcription.py
This is the main function for running the pipeline. It takes in 6 command line argument
* The --image_folder arg specifies the path to the folder containing images to test
* The --save_path arg specifies the path to the location where output files should be stored (defaults to creating a new save file in your PWD)
* The --species_file,--genus_file, and --taxon_file args specify the location of the species, taxon, and genus corpus files
* An optional -d flag to delete any intermediate directories upon program completion (default is true)

## 2. trocr.py
Contains all the functions which relate to running the trocr portion of the pipeline
## 3. matching.py
Contains all of the functions used to match the results from trocr with the species, genus, and taxon information 
## 4. predictions.py
Contains the functions for printing out the results for each image when testing with ground truth values. Implemented in the included cleaned_trocr_test.ipynb file. 
## 5. results.py
Contains all the helper functions for visualizing the results from trocr_transcription.py. Is currently only implemented in the included cleaned_trocr_test.ipynb file
## 6. utilities.py
Contains a number of functions which are primarily related to the invluded CVIT_Training.py file.

## 7. requirements.txt
All required python installs for running the pipeline

# Deployment instructions
Optional: Create a new directory and cd into it

```
mkdir new_directory
cd new_directory
```
Git clone the ml-herbarium repo
```
git clone https://github.com/BU-Spark/ml-herbarium.git
git checkout feature-final-transformers
cd POC/trocr
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
The species, genus, and taxon files are available in the repository at:

```
/ml-herbarium/corpus/corpus_taxon/corpus_taxon.txt

/ml-herbarium-data/corpus_taxon/output/possible_species.pkl

/ml-herbarium-data/corpus_taxon/output/possible_genus.pkl
```
If on a mac you can right click and hold option to get the absolute path and copy these paths as the arguments.


**Note:** It is HIGHLY recommended to run the pipeline on a GPU. Running on CPU is significanly slower. 