# Spring 2023 Herbaria Project Outline

## Kabilan Mohanraj, Keanu Nichols, Rithik Bhandary, Trived Katragadda, 2023-February-18 v0.0.1-dev_


## Overview

1. Situation and current issues
   The changing climate increases stressors that weaken plant resilience, disrupting forest structure and ecosystem services. Rising temperatures lead to more frequent droughts, wildfires, and invasive pest outbreaks, leading to the loss of plant species. That has numerous detrimental effects, including lowered productivity, the spread of invasive plants, vulnerability to pests, altered ecosystem structure, etc. The project aims to aid climate scientists in capturing patterns in plant life concerning changing climate.
    
    The herbarium specimens are pressed plant samples stored on paper. The specimen labels are handwritten and date back to the early 1900s. The labels contain the curator's name, their institution, the species and genus, and the date the specimen was collected. Since the labels are handwritten, they are not readily accessible from an analytical standpoint. The data, at this time, cannot be analyzed to study the impact of climate on plant life.

2. Key Questions
* [Tertiary task] What phenology information are they looking for? Based on the flower/fruit counting, we have assumed that the change in the number of flowers/fruits has to be studied.
* What is the expected input data volume? And budget? [to decide on the endpoint capabilities]
* Do you have a specific dataset in mind? 
* [Primary task] Do you have a dataset on which our solution is expected to perform well? [Mutually agreed upon evaluation set]


3. Hypothesis

    **Primary task**
    1. In the preprocessing step, CRAFT model can be used to extract the handwritten text from the plant image samples.
    2. State-of-the-art transformer-based OCR model can be employed to produce inferences on the handwritten text extracted in the preceeding step.
    3. Refinement training could be performed using supplemental information from the classification task.

    **Secondary task**
    1. Vision Transformers can be used to classify the plants based on their taxon labels.
    2. These predictions could be provided as a priori knowledge to the OCR model to boost its prediction confidence.

    **Tertiary task**
    1. To identify the phenology of the plant under consideration, the flowers and fruits of the plant in the image have to be segmented. This can be achieved by taking advantage of the Mask-RCNN architechture. Transformer based solution will also be explored.
    2. Once the extraction process is done, the flowers and fruits can be counted and further classified into discrete stages based on the plant phenology.


4. Impact
    The digitized samples are an invaluable source of information for climate change scientists, and are providing key insights into biodiversity change over the last century. Digitized specimens will facilitate easier dissemination of information and allow more people access to data. The project, if successful, would enable users from various domains in environmental science to further studies pertaining to climate change and its effects on flora and even fauna.


### A. Problem Statement: 

The Harvard University Herbaria aims to digitize the valuable handwritten information on herbarium specimens, which contain crucial insights into biodiversity changes in the Anthropocene era. 

The main challenge is to develop a transformer-based optical character recognition (OCR) model using deep learning techniques to accurately **locate and extract the specimen labels on the samples to preserve them digitally**. The secondary task involves **building a plant classifier using taxon labels as ground truth, to supplement the OCR model as a source of a priori knowledge.** The tertiary goal involves **indentifying the phenology of the plant specimen under consideration [will be updated after discussing with the client] and possibly predict the biological life cycle stage of the plant**. The successful completion of these objectives will showcase the importance of herbaria in storing and disseminating data for various biological research areas. The ultimate goal is to revive and digitize this valuable information to promote its accessibility to the public and future generations.



### B. Checklist for project completion

1. Determine an agreed-upon evaluation set for the OCR model.
2. Incorporate additional information from the plant images to improve the current best-performing TrOCR model. Attempt approaches such as getting the location information from the images.
3. Develop a plant classifier for the secondary objective using taxon labels as the ground truth. Preprocess the images to remove text using masking methods. If the secondary objective is successful, use this information as an additional input into the TrOCR model.
4. Develop a classifier to identify the life cycle stage of a plant by expanding on the previous work of segmenting flowers/fruits in plant images.




### C. Provide a solution in terms of human actions to confirm if the task is within the scope of automation through AI. 

1. For the main challenge, a person would look at the plant image with text and try to transcribe the text in the image. They would attempt to get the taxon label and the remaining text in the image. The person would refer to a list of known taxon labels to verify that it is a valid label or match it to the closest label. If the text is hard to recognize, they would look at the surrounding text to try to estimate what the word may be. This is a task that can be completed through AI using state-of-the art models.
2. The secondary task, a person would look at the image of the plant and from prior knowledge of known categories of the plants and how they look, the person would then idenify what the plant is. Utilizing knowledge such as the location of the plant would help improve the assurance of what the plant category should be. This is another task that can be automated through AI methods.
3. The tertiary task would involve a person looking at the image of the plant and then drawing out the locations of the fruits and flowers on the image, they would then count the number of occurrences per plant. The person would also classify what stage the flower or fruit is in based on the phenology of the particular flower or fruit. This is a common AI problem in terms of segmenting where the location of the plants and fruits are on the image, additionally classifying the stage of the life cycle the fruit or flower is in.




### D. Outline a path to operationalization.

* **Pipeline development**
We plan to use the BU Shared Computing Cluster (SCC) for model training during the development phase. For model inference, we plan to use TorchServe or TensorFlow Serving depending on the framework used.
* **Demo hosting**
For hosting the model for demo purposes, we plan to use Hugging Face Spaces.
* **Public user access (depending on the volume of requests and budget)**
To provide public access to the inference endpoint, to monitor the model and invoke retraining, we plan to use Kubeflow deployment on bare-metal servers available on research clouds like Chameleon, NERC and MOC. Cloud services like Vertex AI (GCP) and SageMaker (AWS) are also under consideration.




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


## Weekly Meeting Updates

[Meeting notes Google doc](https://docs.google.com/document/d/1XDWf3pze-2Ry9ydcw5s86mSzK6bKi4NbHPcXhiVO73g/edit?usp=sharing)


