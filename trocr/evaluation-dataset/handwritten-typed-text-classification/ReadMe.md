# Handwritten Vs. Typed Text Classification

> **NOTE:**
> * **More Experiments:**
>   1. You can further experiment with the `trocr-base-handwritten` model instead of the TrOCR large model. 

## Overview
Here, we aim to build a pipeline to classify handwritten text and typed/machine-printed text extracted from images. The ultimate goal of this pipeline is to classify the plant specimen images into typed/handwritten categories to create an evaluation set. The evaluation set will be used to test the main TrOCR pipeline. We utilize various machine learning models and techniques for this purpose. 

The following sections of the ReadMe shed light on the files and folders in this directory and how to run the model scripts. For detailed insights on the models explored and the results of the implementation, refer to the [research.md](https://github.com/BU-Spark/ml-herbarium/blob/research-doc-patch/trocr/evaluation-dataset/handwritten-typed-text-classification/research.md) file.

## Getting Started

### Prerequisites and Installation
The prerequisites are the same as the `TrOCR pipeline`. This pipeline can be used in the same conda environment in the `trocr` folder.


## Folder Structure
```
evaluation-dataset/
└── handwritten-typed-text-classification
    ├── ReadMe.md
    ├── detr.py
    ├── doc_classification.py
    ├── data
    │   ├── CVIT
    │   ├── Doc_Classification
    │   ├── FUNSD
    │   ├── IAM
    │   ├── MSCOCO
    │   ├── SROIE2019
    │   ├── all_preprocessed_data
    │   ├── cookies.txt
    │   ├── models--microsoft--trocr-large-stage1
    │   ├── synthetic-font-images-2.zip
    │   ├── synthetic-font-images.zip
    │   ├── synthetic_font_data
    │   ├── synthetic_font_data_2
    │   └── test_data
    ├── logs
    ├── ml-herbaria-synthetic-text-dataset
    ├── model
    ├── notebooks
    │   ├── Classifier_NN.ipynb
    │   ├── Classifier_Transformer.ipynb
    │   ├── Data_Preparation.ipynb
    │   └── Document_Classification.ipynb
    ├── tensors
    │   ├── train
    │   └── valid
    └── utils
        ├── __init__.py
        └── utils.py
```


### Folders and Files Description

#### `data/`

This folder contains all the data files (raw and preprocessed).
- `CVIT`: Raw image data from CVIT dataset.
- `FUNSD`: Raw image data from FUNSD dataset.
- `IAM`: Raw image data from IAM dataset.
- `MSCOCO`: Raw image data from MSCOCO dataset.
- `SROIE2019`: Raw image data from SROIE2019 dataset.
- `synthetic-font-images*.zip`: ZIPs of the generated synthetic images.
- `synthetic_font_data*`: Folders with synthetic images.
- `test_data`: Handpicked images to evaluate the pipeline.
- `Doc_Classification`: Folder to store inputs and outputs of the document classification (plant specimen images are referred as documents since we classify a whole image based on its text components).

#### `notebooks/`

Contains Jupyter notebooks for data preparation and training.
- `Data_Preparation.ipynb`: Notebook with information on data labeling, export, and availability.
- `Classifier_NN.ipynb`: Experimental notebook with all experiments involving various neural network architecture models.
- `Classifier_Transformer.ipynb`: Notebook containing the best performing pipeline for the classification task. This is to classify each text segment into handwritten/typed class.
- `Document_Classification.ipynb`: Notebook that uses the pipeline to classify the entire specimen images/documents into handwritten/typed class.

#### `logs`

Contains TensorBoard logs generated during model training and evaluation.

#### ml-herbaria-synthetic-text-dataset

This folder contains the scripts to generate synthetic images with different fonts and styles. Check the `ReadMe.md` in the folder for more info.

#### `model`

Contains all model files generated during model training. Most of the model files here are experiments and are not part of the final pipeline.

#### `tensors/`

- `train`: Tensors stored during training.
- `valid`: Tensors stored during validation.

#### `utils/`

Contains utility scripts and files.
- `utils.py`: Contains custom model definitions and custom PyTorch transforms.

#### Scripts

- `detr.py`: Script to use DETR object detection model to detect labels from the specimen images. For the CLI version of this script, check the `trocr/label-extraction/` directory.
- `doc_classification.py`: This CLI script classifies images as either "handwritten" or "typed" based on their content. It utilizes the DETR and CRAFT models for object detection and text segmentation, and then uses a TrOCR-based custom classifier to make the final classification.
> Note: Make sure you've downloaded the necessary pretrained models, to provide in the `--decoder-path` option.

**Usage**

Run the script from the command line using the following format:

```bash
python doc_classification.py --input-dir <INPUT_DIRECTORY> --output-dir <OUTPUT_DIRECTORY> --cache-dir <CACHE_DIRECTORY> --decoder-path <DECODER_MODEL_PATH> [--delete-intermediate]
```

**Command Line Arguments**

- `--input-dir`: The path to the input directory containing the images you want to classify.
  
- `--output-dir`: The path to the output directory where the results will be saved.
  
- `--cache-dir`: The path to the cache directory for pretrained models. (Optional; default is `./data/`)

- `--decoder-path`: The path to the pretrained custom classifier model (in `.pth` format).

- `--delete-intermediate`: A flag that, if present, will delete all intermediate files created during the process. It is recommended to use the flag as it would a fresh run every time the script is executed.

**Example:**

```bash
python doc_classification.py --input-dir ./input_images/ --output-dir ./output_results/ --cache-dir ./cache/ --decoder-path ./my_decoder.pth
```

**Output**

The script will output the classified images in two separate folders under the specified `output-dir`:

- `handwritten/`: Contains images classified as handwritten text.
- `typed/`: Contains images classified as typed text.
