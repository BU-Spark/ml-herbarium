# Proof of Concept - Deployment Plan

# File Descriptions
This directory contains all of the files associated with the Tr-OCR pipeline. 
## 1. cleaned_trocr_test.ipynb
This notebook shows the process of setting up craft for segmentation, extracting all bounding boxes around text within the image. These segmented images are then passed through the Tr-OCR model to perform text recognition; we save the transcriptions and model confidence for each of these calculations. Then we perform string matching against a set of files (Taxon, Species, Genus, Country, and Subdivisions of countries) in order to find the best match for each.  All of this information is then accrued in a final output dataframe, which contains the bounding boxes and transciptions for every segmentation, as well as the location of the closest match found between all reference files. We then test the accuracy of the pipeline on roughly 1000 images. Also includes sections that visualize the final results (images with bounding box overlays).
## 2. trocr_train_wab.ipynb
This file is the main training resource for the Tr-OCR model. We have opted to evaluate all available pre-trained models, fine tuning each on the IAM handwriting dataset. All of the training and validation information is logged using Weights and Biases. 
## 3. CVIT_GPU_TRAINING.ipynb
This notebook is an implementation of the huggingface accelerate library in order to more easily use multiple gpus to train on the CVIT dataset. While the character error rate and loss achieved using this training were extremely low (character error rate drops below .00), it did not translate into an accuracy improvement when running the pipeline. It actually made the results significantly worse, as it does well predicting a single word, but past that, it struggles to properly transcribe anything. 
## 4. CVIT_Training.py
This file can be used to train the trocr model on the CVIT dataset as well. It does not implement the accelerate library to implement the multi-gpu training, but can be used to submit batch jobs for training if necessary. 

# Model Deployment Files
Also included are the deployment scripts for running the trocr pipeline locally. 
Those include:
## 1. trocr_transcription.py
This is the main function for running the pipeline. It takes in 6 command line argument
* The --image_folder arg specifies the absolute path to the folder containing images to test.
* The --save_path arg specifies the absolute path to the location where output files should be stored. Defaults to creating a new folder with the specified name in your PWD if the provided filepath does not exist. 
* The --species_file,--genus_file, and --taxon_file args specify the absolute paths of the species, taxon, and genus corpus files.
* An optional -d flag to delete any intermediate directories upon program completion (default is true).

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
cd ml-herbarium
git checkout dev
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

To run the pipeline, please execute the `cleaned_trocr_test.ipynb` notebook in the current (`trocr`) folder.

**Note:** It is HIGHLY recommended to run the pipeline on a GPU. Running on CPU is significanly slower. 

## Final Dataframe Column Descriptions
### Label
This column describes the position that a given image was processed
### Transcription
This column contains the every transcription that was found in the image. They are ordered based on the relative position of the top left coordinate for each bounding box in an image. 
## Transcription_Confidence
This contains the Tr-OCR model confidences in each transcription. This list of values is ordered based on the `Transcription` column (i.e. you can reference each individual transcription and its confidence using the same index number).
## Image_Path
This is the absolute path of the location for a given image
## Bounding_Boxes
This contains the coordinates of each bouding box found in an image. Exactly like transcription confidence, these lists can be indexed based on positions in the `Transcription` column. 
## Bigrams
This column contains every consecutive (word) bi-gram found by joining the list of strings contained in `Transcription`. 
## Bigram_idx
This column contains the index (relative to `Transcription`) for each word in a bigram. For example, if the `Transcription` column contained the transcriptions 'Herbier Museum' and 'Paris', `Bigrams` would contain 'Herbier Museum' and 'Museum Paris'. `Bigrams_idx` would contain (0,0) and (0,1), as in the first bigram, both words are contained in `Transcription[0]`, while in the second bigram the first word comes from `Transcription[0]`, with the second word coming from `Transcription[1]`.
 
## Taxon_Prediction_String
This column contains the piece of transcribed text that was used to make the final prediction for Taxon. 
## Taxon_Similarity
This column contains the cosine similarity between the string in    `Taxon_Prediction_String` and `Taxon_Prediction`.
## Taxon_Prediction
This column contains the models prediction for the taxon of the input image.
## Taxon_Index_Location
This column contains the index value (in `Bigrams` and `Bigram_idx`) for the string from `Taxon_Prediction_String`. This allows you to pick out the bounding boxe(s) and transcription confidence(s) associtaed with the string used for making your final prediction. 

All columns past this point are formatted exactly the same as the previous 4, but reference a different set of corpus files. For example, if matching against a 'Species' corpus file, the next 4 columns in the csv will be `Species_Prediction_String`, `Species_Similarity`,`Species_Prediction`, and `Species_Index_Location`.
