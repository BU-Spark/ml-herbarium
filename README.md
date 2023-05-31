# ML-Herbarium

This repository contains a software pipeline to process herbarium specimens. There are currently three tasks this project aims to accomplish:
1. Classifying specimens via optical character recognition (OCR) and named entity recognition (NER)
2. [Classifying specimens via image classification](https://github.com/BU-Spark/ml-herbarium-classify.git)
3. [Identifying the phenology of specimens](https://github.com/BU-Spark/ml-herbarium-phenology)

## Getting Started
The `run.ipynb` file contains demonstrates how to run each of the three tasks. Additionally, each of the aformentioned tasks is has supporting files and documentation in its respective folder. The primary task is in the `transcription` folder ([README](./transcription/ReadMe.md)), the secondary in the `vision` folder ([README](./vision/README.md)), and the tertiary in the `phenology` folder ([README](./phenology/README.md)). The 

# Overview
 
The changing climate increases stressors that weaken plant resilience, disrupting forest structure and ecosystem services. Rising temperatures lead to more frequent droughts, wildfires, and invasive pest outbreaks, leading to the loss of plant species. That has numerous detrimental effects, including lowered productivity, the spread of invasive plants, vulnerability to pests, altered ecosystem structure, etc. The project aims to aid climate scientists in capturing patterns in plant life concerning changing climate.

The herbarium specimens are pressed plant samples stored on paper. The specimen labels are handwritten and date back to the early 1900s. The labels contain the curator's name, their institution, the species and genus, and the date the specimen was collected. Since the labels are handwritten, they are not readily accessible from an analytical standpoint. The data, at this time, cannot be analyzed to study the impact of climate on plant life.

The digitized samples are an invaluable source of information for climate change scientists, and are providing key insights into biodiversity change over the last century. Digitized specimens will facilitate easier dissemination of information and allow more people access to data. The project, if successful, would enable users from various domains in environmental science to further studies pertaining to climate change and its effects on flora and even fauna.


## Project Description

The Harvard University Herbaria aims to digitize the valuable handwritten information on herbarium specimens, which contain crucial insights into biodiversity changes in the Anthropocene era. 

The main challenge is to develop a transformer-based optical character recognition (OCR) model using deep learning techniques to accurately **locate and extract the specimen labels on the samples to preserve them digitally**. The secondary task involves **building a plant classifier using taxon labels as ground truth, to supplement the OCR model as a source of a priori knowledge.** The tertiary goal involves **identifying the phenology of the plant specimen under consideration [will be updated after discussing with the client] and possibly predict the biological life cycle stage of the plant**. The successful completion of these objectives will showcase the importance of herbaria in storing and disseminating data for various biological research areas. The ultimate goal is to revive and digitize this valuable information to promote its accessibility to the public and future generations.



## Project Checklist

1. Determine an agreed-upon evaluation set for the OCR model.
2. Incorporate additional information from the plant images to improve the current best-performing TrOCR model. Attempt approaches such as getting the location information from the images.
3. Develop a plant classifier for the secondary objective using taxon labels as the ground truth. Preprocess the images to remove text using masking methods. If the secondary objective is successful, use this information as an additional input into the TrOCR model.
4. Develop a classifier to identify the life cycle stage of a plant by expanding on the previous work of segmenting flowers/fruits in plant images.




## Proposed Solution

1. For the main challenge, a person would look at the plant image with text and try to transcribe the text in the image. They would attempt to get the taxon label and the remaining text in the image. The person would refer to a list of known taxon labels to verify that it is a valid label or match it to the closest label. If the text is hard to recognize, they would look at the surrounding text to try to estimate what the word may be. This is a task that can be completed through AI using state-of-the art models.
2. The secondary task, a person would look at the image of the plant and from prior knowledge of known categories of the plants and how they look, the person would then idenify what the plant is. Utilizing knowledge such as the location of the plant would help improve the assurance of what the plant category should be. This is another task that can be automated through AI methods.
3. The tertiary task would involve a person looking at the image of the plant and then drawing out the locations of the fruits and flowers on the image, they would then count the number of occurrences per plant. The person would also classify what stage the flower or fruit is in based on the phenology of the particular flower or fruit. This is a common AI problem in terms of segmenting where the location of the plants and fruits are on the image, additionally classifying the stage of the life cycle the fruit or flower is in.

## Other Folders
### `./corpus`
The corpus folder contains the code to generate the corpus file. The corpus file is a `.pkl` file of all possible pairs of genus and species. This file is used in transcription to match the extracted text to the corpus.

### `./CRAFT`
The CRAFT folder contains the code to run the CRAFT model. The CRAFT model is used to extract the text from the images and place bounding boxes around the text.

### `./EDA`
The `EDA` folder contains an exploritory data analysis of the dataset. The `EDA.ipynb` file contains the code to generate the EDA. The `EDA_Notebook_Spring_2023.ipynb` file contains the latest output of the EDA.

### `./scraping`
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

### `./trocr`
The trocr folder contains the code to run the TrOCR model. The TrOCR model is used to transcribe the text from the images.

## Resources

### Data Sets

* CNH Portal: https://portal.neherbaria.org/portal/ 
* Pre-1940 plant specimen images in GBIF: https://www.gbif.org/occurrence/gallery?basis_of_record=PRESERVED_SPECIMEN&media_ty[…]axon_key=6&year=1000,1941&advanced=1&occurrence_status=present  
* International Plant Names Index: https://www.gbif.org/dataset/046bbc50-cae2-47ff-aa43-729fbf53f7c5#dataDescription
* Use for synonyms (GBIF is recommended):
GBIF: https://hosted-datasets.gbif.org/datasets/backbone/current/
IPNI:  https://storage.cloud.google.com/ipni-data/
* CVIT: https://cvit.iiit.ac.in/research/projects/cvit-projects/matchdocimgs
* IAM: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database


### References

1. CRAFT (text detection): https://arxiv.org/abs/1904.01941
2. TrOCR: https://arxiv.org/abs/2109.10282
3. "What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis," [10.1109/ICCV.2019.00481](https://doi.org/10.1109/ICCV.2019.00481)
4. Kubeflow: https://www.kubeflow.org/docs/
5. Hugging Face Spaces: https://huggingface.co/docs/hub/spaces
6. GCP Vertex AI: https://cloud.google.com/vertex-ai/docs
7. AWS SageMaker: https://docs.aws.amazon.com/sagemaker/index.html
8. TensorFlow Serving: https://github.com/tensorflow/serving
9. TorchServe: https://github.com/pytorch/serve
