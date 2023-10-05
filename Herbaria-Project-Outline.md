# Herbaria Project Outline

### *Smriti Suresh, Dima Kazlouski, Douglas Moy - 2023-10-06 v1.0.0-dev*

## Overview

The changing climate increases stressors that weaken plant resilience, disrupting forest structure and ecosystem services. Rising temperatures lead to more frequent droughts, wildfires, and invasive pest outbreaks, leading to the loss of plant species. That has numerous detrimental effects, including lowered productivity, the spread of invasive plants, vulnerability to pests, altered ecosystem structure, etc. The project aims to aid climate scientists in capturing patterns in plant life concerning changing climate.
The herbarium specimens are pressed plant samples stored on paper. The specimen labels are handwritten and date back to the early 1900s. The labels contain the curator's name, their institution, the species and genus, and the date the specimen was collected. Since the labels are handwritten, they are not readily accessible from an analytical standpoint. The data, at this time, cannot be analyzed to study the impact of climate on plant life.
The digitized samples are an invaluable source of information for climate change scientists, and are providing key insights into biodiversity change over the last century. Digitized specimens will facilitate easier dissemination of information and allow more people access to data. The project, if successful, would enable users from various domains in environmental science to further studies pertaining to climate change and its effects on flora and even fauna.


### A. Problem Statement:

Develop a machine learning solution for automating the extraction and digitization of critical information from handwritten labels on historical herbarium specimens, dating back to the early 1900s.

### B. Checklist for project completion

1. Increase evaluation dataset to 1000 images
2. Improve transcription accuracy of the taxon label
3. Extract and transcribe the following for minimal metadata capture:
    a. Geography 
    b. Collection code
    c. Barcode
4. For detailed data capture:
    a. Date Collected
    b. Collector Name
    c. Collector Number
    d. Habitat
5. Produce clean code and documentation


### C. Provide a solution in terms of human actions to confirm if the task is within the scope of automation through AI.

1. For the main challenge, a person would look at the plant image with text and try to transcribe the text in the image. They would attempt to get the taxon label and the remaining text in the image. The person would refer to a list of known taxon labels to verify that it is a valid label or match it to the closest label. If the text is hard to recognize, they would look at the surrounding text to try to estimate what the word may be. This is a task that can be completed through AI using state-of-the art models.

2. The secondary task, a person would look at the image of the plant and from prior knowledge of known categories of the plants and how they look, the person would then idenify what the plant is. Utilizing knowledge such as the location of the plant would help improve the assurance of what the plant category should be. This is another task that can be automated through AI methods.

3. The tertiary task would involve a person looking at the image of the plant and then drawing out the locations of the fruits and flowers on the image, they would then count the number of occurrences per plant. The person would also classify what stage the flower or fruit is in based on the phenology of the particular flower or fruit. This is a common AI problem in terms of segmenting where the location of the plants and fruits are on the image, additionally classifying the stage of the life cycle the fruit or flower is in.


### D. Outline a path to operationalization.

The proposed workflow for digitizing herbarium specimen is as follows:

- Imaging: A customized photo station for imaging herbarium sheets and an automated image processing pipeline is used for digitizing the sheets.
- Label transcription: Extracting the label from the specimen sheet, transcribing the label:
-- Minimal metadata capture 
-- Detailed data capture
- Georeferencing: generation of latitude, longitude, and associated uncertainties.

After the above mentioned processing, we aim to provide the consolidated dataset of herbaria specimen with the desired labels of Taxon, Geography, Collection Code, Barcode, Location, Date Collected, Collector Name, Collector Number, Habitat, etc. to the end-users i.e. climate change scientists, to help in providing key insights into biodiversity change in the age of the Anthropocene.

Final Format/Technology used to generate the output files will be updated subsequently after discussion with client. Possible hypothesized outputs include a fully functioning dynamic website with the Herbaria specimen sorted according to various features, for easy accessibility and reach by end-users. 

## Resources

### Data Sets

- CNH Portal: https://portal.neherbaria.org/portal/ 
- Pre-1940 plant specimen images in GBIF: https://www.gbif.org/occurrence/gallery?basis_of_record=PRESERVED_SPECIMEN&media_ty[…]axon_key=6&year=1000,1941&advanced=1&occurrence_status=present  
- International Plant Names Index: https://www.gbif.org/dataset/046bbc50-cae2-47ff-aa43-729fbf53f7c5#dataDescription
- Use for synonyms (GBIF is recommended):
GBIF: https://hosted-datasets.gbif.org/datasets/backbone/current/
IPNI:  https://storage.cloud.google.com/ipni-data/


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


## Weekly Meeting Updates

* Kick-off meeting scheduled on 10/06 @ 4pm

## Link to Meeting Doc

[Project Description Document](https://docs.google.com/document/d/1dZnUwqAI2QuPxcOWMhyFHBAfiDxz1-M_trMtQh8flsA/edit#heading=h.uoj40lvdvnl3)