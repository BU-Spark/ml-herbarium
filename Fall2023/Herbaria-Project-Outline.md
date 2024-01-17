# Herbaria Project Outline

### *Smriti Suresh, Dima Kazlouski, Douglas Moy - 2023-10-06 v1.0.0-dev*

## Overview

The changing climate increases stressors that weaken plant resilience, disrupting forest structure and ecosystem services. Rising temperatures lead to more frequent droughts, wildfires, and invasive pest outbreaks, leading to the loss of plant species. That has numerous detrimental effects, including lowered productivity, the spread of invasive plants, vulnerability to pests, altered ecosystem structure, etc. The project aims to aid climate scientists in capturing patterns in plant life concerning changing climate.
The herbarium specimens are pressed plant samples stored on paper. The specimen labels are handwritten and date back to the early 1900s. The labels contain the curator's name, their institution, the species and genus, and the date the specimen was collected. Since the labels are handwritten, they are not readily accessible from an analytical standpoint. The data, at this time, cannot be analyzed to study the impact of climate on plant life.
The digitized samples are an invaluable source of information for climate change scientists, and are providing key insights into biodiversity change over the last century. Digitized specimens will facilitate easier dissemination of information and allow more people access to data. The project, if successful, would enable users from various domains in environmental science to further studies pertaining to climate change and its effects on flora and even fauna.


### A. Problem Statement:

Develop a machine learning solution for automating the extraction and digitization of critical information from handwritten and typed text labels on historical herbarium specimens, dating back to the early 1900s. These labels include: location, date collected, collector name etc. The goal is to be able to automatically upload these labels to a database. We will be contributing our work to the existing Github repo. 
Include support for recognizing Chinese and Cyrilic characters as well. 

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

1. OCR and Transcription Improvement:

A person would look at the plant image embedded with text, aiming to transcribe this text. Their primary focus is on the taxon label, and the surrounding textual content within the image. In addition, they would be on the lookout for specific items such as geography, collection code, barcode, date collected, collector name, collector number, and habitat.

To ensure accuracy, this person would cross-reference the taxon label with a known list, ensuring it's either a valid label or closely matching one. If they encounter text that's challenging to interpret, they would draw from adjacent text or other contextual cues within the image to infer the likely content. This detailed transcription and validation task aligns well with capabilities of state-of-the-art OCR models, especially with refinements to prioritize the extraction of these newly outlined items.

2. Plant Classification:

Once textual content is identified or masked, the person would then shift their focus to the visual aspects of the plant image. Drawing from their prior knowledge of known plant categories, as well as potential hints from the transcribed information like the taxon label, geography, and collection code, they would determine the plant's identity. Recognizing a plant based on its visual characteristics and aligning it with external clues remains a task achievable with AI, particularly when using advanced computer vision methods and a robust training dataset.

3. Plant Features Identification:

Upon examining the plant image, the person would mark out specific features like fruits and flowers, noting their locations. They would count the number of such features and also categorize them based on their phenological stage, i.e., the stage in their life cycle. This identification and classification step pertains to AI's capabilities in image segmentation and classification, given a well-structured dataset.


### D. Outline a path to operationalization.

The proposed workflow for digitizing herbarium specimen is as follows:

- Imaging: A customized photo station for imaging herbarium sheets and an automated image processing pipeline is used for digitizing the sheets.
- Label transcription: Extracting the label from the specimen sheet, transcribing the label:
-- Minimal metadata capture 
-- Detailed data capture
- Georeferencing: extraction of location label, and associated uncertainties.

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


## Weekly Meeting Updates

### Kick-off meeting Notes 10/06

* Attendance: All team members, Freddie, Tom
#### Notes:
The herbaria is a natural history collection

Behind the few specimens on exhibit are warehouses of preserved specimens

Understanding all the morphological differences between species

Dried and pressed species onto a sheet, 6 million plant specimen, 1.5 million are databased

Multiple digitization efforts going back a decade, but haven’t had the funding to digitize all at once

Pulling out bundles of specimens

* Workflow today: batch imaging, manual web-app, data in form fields to create records, costly and time consuming

#### Action Items

Develop Automatic Tools to extract features from the specimens in order to create db records

*Goals*: Handwriting Recognition and OCR

Marginal improvements to OCR are not necessary

Innovating on Handwriting Recognition

Semantic Identification of the Correct Fields
- Geography
- Taxonomic Name
- Collector Name
- Date Collected

Fully automate the creation of some subset of records
Find opportunities to increase accuracy and performance
Contained/Deployed ideal

### Client meeting Notes 10/16

* Attendees:  Charles, Jonathan, Freddie, Kabilan, Dima, Smriti, Douglas, Thomas

Other labels to extract:
- Taxon (genus/species) – disambiguating against thesaurus
- High level geo – minimal country, preferably province and county
- Date
- Collector  and Collector # (optional)

Charles mentioned the work at UMICH, for which Thomas Gardos posted this link on the Slack channel.

We got this pointer from Jonathan and Charles to related work going on at UMichigan. There's a 10 minute video from BioDigiCon last month here (starts at 44:30). They basically use Google Vision OCR to get a raw dump of all the text from the specimen, then use LLMs to convert it to a JSON schema.
The researcher is Will Weaver, and his project is on GitHub.

Dima showed an initial result below of using ChatGPT-4 + Vision to recognize text from a GBIF image.

Team decided that it is worth exploring these directions (both the UMICH approach as well as the fully converged approach Dima tried below), in addition to improving upon the pipeline built from previous term.

#### Follow up:
Charles, Jonathan – send pointers or query terms to pull representative asian herbaria dataset.

Try “chinese virtual herbarium”, https://www.cvh.ac.cn/ 

Looks like this link is where you can access over 8M images. https://www.cvh.ac.cn/spms/list.php 

Team will clean up the Trello board

## Link to Meeting Doc

[Project Description Document](https://docs.google.com/document/d/1dZnUwqAI2QuPxcOWMhyFHBAfiDxz1-M_trMtQh8flsA/edit#heading=h.uoj40lvdvnl3)