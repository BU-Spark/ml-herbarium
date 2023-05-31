# Proof of Concept - Deployment Plan

# File Descriptions
This directory contains all of the files associated with the Tr-OCR pipeline. 
## cleaned_trocr_test.ipynb
This notebook shows the process of setting up craft for segmentation, extracting all bounding boxes around text within the image. These segmented images are then passed through the Tr-OCR model to perform text recognition; we save the transcriptions and model confidence for each of these calculations. Then we perform entity recognition and linking against a database of taxons (in order to find the best match for each.  All of this information is then accrued in a final output dataframe, which contains the bounding boxes and transciptions for every segmentation. We then test the accuracy of the pipeline on roughly 250 images.

# Model Deployment Files
## trocr_transcription.py
> **NOTE:** This deployment file has not been updated to reflect the current state of the pipeline. Please consider executing the Jupyter Notebook `cleaned_trocr_test.ipynb`.

This is the main function for running the pipeline. It takes in 6 command line argument

The --image_folder arg specifies the absolute path to the folder containing images to test.
The --save_path arg specifies the absolute path to the location where output files should be stored. Defaults to creating a new folder with the specified name in your PWD if the provided filepath does not exist.
The --species_file,--genus_file, and --taxon_file args specify the absolute paths of the species, taxon, and genus corpus files.
An optional -d flag to delete any intermediate directories upon program completion (default is true).
## trocr.py
Contains all the functions which relate to running the trocr portion of the pipeline
## utilities.py
Contains a number of functions which are primarily related to the invluded CVIT_Training.py file.

## requirements.txt
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
Create a new conda environment and activate it
```
conda create -n my-conda-env
conda activate my-conda-env
```

Install all required packages and Jupter
```
conda install jupyter
pip install -r requirements.txt
```

To start Jupyter Notebooks in the current folder, use the command
```
jupyter notebook
```

To run the pipeline, please execute the `cleaned_trocr_test.ipynb` notebook in the current (`trocr`) folder.

> **NOTE:** It is HIGHLY recommended to run the pipeline on a GPU. Running on CPU is significanly slower. 

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
## Taxon_Output
This contains the taxons predicted by the entity linking step in the TaxoNerd pipeline.
## Confidence_Output
This contains the confidence for taxons predicted by the entity linking step in the TaxoNerd pipeline.

## Evaluation Dataframe Column Descriptions
## Confidence_Threshold
This column contains the confidence threshold use for entity linking in the TaxoNerd pipeline. 
## Taxons_Predicted
This column contains the number of taxons predicted at each `Confidence_Threshold` (from 0 to 0.9).
## Taxons_Accuracy_Predicted
This column contains the number of taxons predicted with a cosine similarity score of `>0.8` at each `Confidence_Threshold`.
