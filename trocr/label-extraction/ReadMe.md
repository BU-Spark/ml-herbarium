# DETR Label Extraction

> **NOTE:**
> * More rigorous tests need to run for the DETR model with the drawback being variety in bounding boxes in the plant sample images (from GBIF).
> * A local copy of the best model (so far) can be found in the `./data/best_model/` directory. This model can be imported using `from_pretrained` method from the `transformers` library.
> * *About variety in bounding boxes:* number of bounding boxes per image, size of bounding boxes etc.. The variety has to be tackled by labeling more data in each case.
>
> * **Recommended next steps (takes considerable time)**
>   1. Identify and group similar images. For example, one group can be images with >=3 labels/<3 labels, another group can be images with relatively small/large bounding boxes (low/high area coverage).
>   2. Create a representative train set and validation set. Have a hold out test set.
>   3. Active learning could be a very helpful tool with labeling the images. I haven't been able to explore much about this, but I will share resources as I come by them.
>   4. Test out data augmentation. The data loader in the `DETR_Finetuned_on_Labeled_Data.ipynb` notebook has some common image augmentation implemented.
>   5. Train the model with different sets of layers unfrozen (parameter.requires_grad) and pick the best one.

## Overview
Here, we aim to use DETR (DEtection TRansformer) to segment labels from our plant sample images, through object detection.

## Getting Started

### Prerequisites and Installation
The prerequisites are the same as the `TrOCR pipeline`. The DETR model can be used in the same conda environment in the `trocr` folder. In addition to the `TrOCR` requirements, `pytorch-lightning` is required.

### Usage
The Jupyter notebooks can be run on SCC as usual. 

The inference script can be run using the following command.
```
python detr_inference.py --image-folder <input-directory> --output-folder <output-directory> --pretrained-model <model-name> --cache-dir <cache-directory>
```

## Folder Structure
```
label-extraction/
├── data/                       
│   ├── detr_train_test_coco/
│   ├── detr_train_test_coco.zip
│   └── models--*/            
├── notebooks/
│   ├── Data_Preparation.ipynb 
│   └── DETR_Finetuned_on_Labeled_Data.ipynb
├── lightning_logs/                                  
│   └── version_*/          
├── ReadMe.md                   
└── detr_inference.py                     
```

### Folders and Files Description:

#### `data/`
This folder contains raw and preprocessed data.
- `detr_train_test_coco/`: COCO format data, not to be modified.
- `detr_train_test_coco.zip`: ZIP file of the above data.
- `models--*/`: Model downloaded from Hugging Face (cached here).

#### `notebooks/`
Contains Jupyter notebooks for data preparation and training.
- `Data_Preparation.ipynb`: Notebook with information on data labeling, export and availability.
- `DETR_Finetuned_on_Labeled_Data.ipynb`: The notebook fine-tunes a DETR (DEtection TRansformer) model for object detection tasks on our custom plant specimen dataset that uses the COCO format for annotations. This code is capable of both training and validating the model, complete with data augmentation, loss calculation, and metric evaluation. Pytorch Lightning is used for model training and evaluation.

#### `lightning_logs/`
Contains logs generated during model training and evaluation.
- `version_*/`: Logs for all experiments. To visualize logs, start a `TensorBoard` instance on SCC OnDemand pointing to this folder.

#### `detr_inference.py`
Inference script to download fine-tuned model from Hugging Face and get predictions using the model.