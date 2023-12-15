# HerbariaOCR

> Character recognition from herbaria image samples.

### *Smriti Suresh, Dima Kazlouski, Douglas Moy - 2023-10-06 v1.0.0-dev*

## Overview

The changing climate increases stressors that weaken plant resilience, disrupting forest structure and ecosystem services. Rising temperatures lead to more frequent droughts, wildfires, and invasive pest outbreaks, leading to the loss of plant species. That has numerous detrimental effects, including lowered productivity, the spread of invasive plants, vulnerability to pests, altered ecosystem structure, etc. The project aims to aid climate scientists in capturing patterns in plant life concerning changing climate.
The herbarium specimens are pressed plant samples stored on paper. The specimen labels are handwritten and date back to the early 1900s. The labels contain the curator's name, their institution, the species and genus, and the date the specimen was collected. Since the labels are handwritten, they are not readily accessible from an analytical standpoint. The data, at this time, cannot be analyzed to study the impact of climate on plant life.
The digitized samples are an invaluable source of information for climate change scientists, and are providing key insights into biodiversity change over the last century. Digitized specimens will facilitate easier dissemination of information and allow more people access to data. The project, if successful, would enable users from various domains in environmental science to further studies pertaining to climate change and its effects on flora and even fauna.

## Problem Statement:

Develop a machine learning solution for automating the extraction and digitization of critical information from handwritten and typed text labels on historical herbarium specimens, dating back to the early 1900s. These labels include: location, date collected, collector name etc. The goal is to be able to automatically upload these labels to a database. We will be contributing our work to the existing Github repo.
Include support for recognizing Chinese and Cyrilic characters as well.

## Installation 

> Note : This is a work in progress. We would be adding HerbariaOCR as a PyPI package after formatting the repository a bit more. So stay tuned!

```sh
pip install HerbariaOCR
```

## How To Use

All of our work was done through Jupyter Notebooks on SCC since our data sources were uploaded on the ml-herbarium SCC. Thus, the only thing the user needs to do to run the project is to start a new session on Jupyter-lab, Google Colab, VSCode or wherever you prefer to host your notebooks, and run all the cells in succession. (Please make sure your data paths are correct if you prefer using your own dataset)

Refer to the EDA Fall 2023 notebook for exploring the Herbaria OCR data sources including the Chinese and Cyrllic datasets. We have examples of specimen from each language as well as the data analysis.
The evaluation sets we created to test our model extracted samples from the datasets mentioned in the EDA notebook.

To use the model to extract labels from images through our application, please utilize the AzureVision.ipynb notebook. The notebook has multiple components that are described in depth in the notebook overview section.

On a high level, running the notebook you can access a friendly UI that allows an upload of a herbarium specimen and returns desired information in DARWIN JSON format. You can also run multiple Herbarium images located in a folder through the pipeline to obtain results in a .txt or .pdf format. You can also get results for Cyrillic and Chinese specimens.

Again, please refrence the Notebook Overview section at the head of the AzureVision.ipynb file for more information.

Refer to the LLM Evaluations notebooks for exploring the Evaluation metrics used as well as Accuracy results for English, Cyrillic and Chinese samples respectively.
Note : We have defined "Accuracy" in different ways for the different labels of Taxon, Collector and Geography. It would be more enriching for the user to also pay attention to the comments and description before the code to have a better understanding of the results and how we came up with an accuracy value.

## Deployment

We have made use of two platforms for deployment.
1. The user pipeline can be viewed through our [Gradio app](https://huggingface.co/spaces/smritae01/HerbariaOCR) hosted on HuggingFace which enables one to upload a specific Herbarium specimen and run the Azure-GPT pipeline to extract all the text from the specimen image and display it in the Darwin-Core json format. 
2. (Work in Progress) Our entire project has an nbdev deployment at [HerbariaOCR](https://github.com/BU-Spark/HerbariaOCR) and is currently able to run locally. Just clone the repository, install the dependencies from requirements.txt and run the nbdev_preview command to obtain a link to the website for the project. 

## Pricing

Considering we make use of the Azure AI vision API as well as the Chat GPT4 Turbo API, we do require the users to have API keys for the same purpose and input them in the code wherever specified. The resulting text transcriptions are available for your perusal from the dataset we considered, but pricing definitely needs to be a part of the description of this project.

## Contributing

If you want to contribute to this project, there are primarily two ways:

1. Contribute bug reports and feature requests by submitting [issues](https://github.com/BU-Spark/HerbariaOCR/issues) to the GitHub repo.
2. If you want to create Pull Requests with code changes, read the [contributing guide](https://github.com/BU-Spark/HerbariaOCR/blob/main/CONTRIBUTING.md) on the github repo.

## Resources

### Data Sets

- CNH Portal: https://portal.neherbaria.org/portal/
- Pre-1940 plant specimen images in GBIF: https://www.gbif.org/occurrence/gallery?basis_of_record=PRESERVED_SPECIMEN&media_ty[…]axon_key=6&year=1000,1941&advanced=1&occurrence_status=present  
- International Plant Names Index: https://www.gbif.org/dataset/046bbc50-cae2-47ff-aa43-729fbf53f7c5#dataDescription
- Use for synonyms (GBIF is recommended):
GBIF: https://hosted-datasets.gbif.org/datasets/backbone/current/
IPNI:  https://storage.cloud.google.com/ipni-data/
- Chinese Virtual Herbarium : https://www.cvh.ac.cn/spms/list.php


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
10. [The herbarium of the future](https://www.cell.com/trends/ecology-evolution/fulltext/S0169-5347(22)00295-6)
11. [Harvard University Herbaria](https://huh.harvard.edu/mission)

## Link to Project Description Doc

[Project Description Document](https://docs.google.com/document/d/1dZnUwqAI2QuPxcOWMhyFHBAfiDxz1-M_trMtQh8flsA/edit#heading=h.uoj40lvdvnl3)

