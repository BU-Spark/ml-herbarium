# ML-Herbarium Project [the project is continuing this semester and there will be updates to the file as the semester continues]

This repository contains a software pipeline to process herbarium specimens. The specemin image is processed by `transcribe_labels.py` and the results are stored in the output folder. The result will either be a match to a known taxon in the corpus file or 'NO MATCH'. The current pipeline uses [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/OldVersionDocs.html#tesseract-4) to extract the text from the image and then uses fuzzy matching and structural pattern matching to match the extracted text to the corpus. 

Dependencies can be found in `requirements.txt`

<br />

# Installation Instuctions
## 1. Add dependencies to auto load in your .bashrc file (FOR BU SCC ONLY)
`nano ~/.bashrc`

Add these lines to the file:
```
module load python3/3.10.5
module load leptonica/1.82.0
module load libicu/71.1
module load tesseract/4.1.3
```
`ctrl+x` then return to save and exit

> Note: If you are not using the BU SCC, you will need to install the above dependencies manually.
## 2. Create and activate virtual environment for dependency management
`python3 -m venv .env`
### Activate virtual env
`source .env/bin/activate`
### Install requirements
`pip install -r requirements.txt`

## Note for VS Code
To select the correct Python interpreter, open your command palette (Command or Control+Shift+P), select `Python: Select Interpreter` then choose `Python 3.10.5` in your `.env` path.

# Instructions for Manual Installation of Dependencies
> Note: The following instructions are for manual installation. If you are using the BU SCC, you do not need to follow these instructions. These instructions are for installing ***without*** root access on ***CentOS***. If installing on other distributions, you may want to just install using `apt-get install` or `yum install`.
## 1. Install Leptonica
In your home directory, run:

`git clone https://github.com/DanBloomberg/leptonica.git --depth 1`

`cd leptonica`

`./autogen.sh`

`./configure --prefix=$HOME/.local --disable-shared`

`make`

`make install`

## 2. Install Tesseract
In your home directory, run:

`wget https://github.com/tesseract-ocr/tesseract/archive/4.1.3.tar.gz -O tesseract-4.1.3.tar.gz`

`tar zxvf tesseract-4.1.3.tar.gz`

`export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig`

`./autogen.sh`

`./configure --prefix=$HOME/.local --disable-shared`

`make`

`make install`

`cd ~/.local/share/tessdata`

`wget https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/eng.traineddata`


<br />

# Pipeline Features
## Transcription
The pipeline uses [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/OldVersionDocs.html#tesseract-4) to extract the text from the image and then uses fuzzy matching and structural pattern matching to match the extracted text to the corpus.

To run transcription, open the transcription folder and run `transcribe_labels.py`.
```
Usage: python3 transcribe_labels.py <org_img_dir> [OPTIONAL ARGUMENTS]

OPTIONAL ARGUMENTS:
        -o <output_dir>, --output <output_dir>
        -n <num_threads>, --num-threads <num_threads> (Default: 32)
        -d, --debug
```
> Note: This will take a while to run. Be sure your environment is activated and properly configured.

## Training
Training can be done by following the [Tesseract Training Instructions](https://tesseract-ocr.github.io/tessdoc/tess4/TrainingTesseract-4.00.html) Training can be useful to fine-tune the English Tesseract model to improve its handwriting recognition. A standalone model can also be trained from scratch.

An importatnt note is that all the training data must be on one folder (the ground truth folder). Each training image must be a `.png` or `.tiff` file. Each training image must have a corresponding ground truth file with the same name and the extension `.gt.txt`. More docs and some scripts to generate the ground truth files can be found in the `transcription/tesseract/training` folder.

The traning datasets used so far for training so far are [CVIT](https://cvit.iiit.ac.in/research/projects/cvit-projects/matchdocimgs), [IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database), and [NMIST](http://yann.lecun.com/exdb/mnist/). The IAM dataset seemed to not have enough data to yeild good results for training a Tesseract model. The NMIST dataset only contains single characters, so it also is not ideal. The CVIT dataset is very large, but it is promising to train a good model; however, we did not have enough time to complete the training.

## Scraping
The scraping folder contains the code to scrape the GBIF database to download images for testing and training purposes. The scraping workflow also generates a mock corpus file and ground truth file.
To run transcription, open the dataset folder in the scraping folder and run `datasetscraping.py`.
```
Usage: 'python3 datasetscraping.py <dataset_type> [OPTIONAL ARGS]' where dataset_type is either 'dwca' or 'csv'.

Optional arguments:
        -o, --output_path: Specify the output path for the images. Default is './output/'.
        -p, --percent_to_scrape: Specify the percentage of the dataset to scrape. Default is 0.00015 (~1198 occurrences).
        -u, --dataset_url: Specify the dataset URL to download a new dataset.
        -c, --num_cores: Specify the number of cores to use. Default is 50.
        -k, --keep: Keep current csv dataset, and do not unzip new dataset.
        -h, --help: Print this help message.
```

## Corpus generation
The corpus folder contains the code to generate the corpus file. The corpus file is a `.pkl` file of all possible pairs of genus and species. This file is used in transcription to match the extracted text to the corpus.

## Segmentation (not in use)
The segmentation folder contains the code to segment the labels from the rest of the image. It takes the specemin image and outputs a cropped image containing only the largest label. We chose to not use this step as it is not necessary for the pipeline, and reduces accuracy. It does however reduce the transcription time when segmented images are used instead of full images.

# Accuracy
## Transcription Accuracy
Current metrics for total pipeline accuracy are:
11/30 correct with a 100% accurate match rate. In other words, when the pipeline returns a match, it is almost always correct (otherwise returns no match).

# Original Pipeline
The original pipeline takes a different approach to transcription. It uses MXNet and a custom model to extract the text from the image and then uses fuzzy matching to match the extracted text to the corpus. Overall the new pipeline preforms better than the original pipeline; however, the old pipeline is sometimes better for hand-labeled images. The original pipeline is approximately 10x slower than the new pipeline.

Below are docs to assist with running and training the original pipeline. More are available in the `CRAFT` folder and the `transcription_original` folder.

## Allocate GPU (If using old CRAFT Pipeline)
### check if V100 GPU being used
`lshw -C display`
### syscall for executing jobs on GPU, RUN if lshw -C display does not return V100
`qrsh -l gpus=1 -l gpu_type=V100`
### check that GPU updated
`lshw -C display`

<br/>

## Run the CRAFT pipeline
### Bounding boxes for image masks of text boxes
`cd CRAFT/CRAFT-pytorch-master`
### make script executable
`chmod +x bash_submit.sh`
### execute CRAFT detector on HUH images
`./bash_submit.sh`

<br/>

# Add Users
To add yourself to the repository, open a Pull Request modifying `COLLABORATORS`, entering your GitHub username in a newline.

All Pull Requests must follow the Pull Request Template, with a title formatted like such `[Project Name]: <Descriptive Title>`
