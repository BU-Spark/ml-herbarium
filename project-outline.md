# Fall 2022 ML Herbarium Project Outline 

## _Authors: Zhengqi Dong, Cole Hunter, Zihua Xin, Mark Yang_
** **
## Overview

In concert with the Harvard University Herbaria, we will continue the work of previous teams who created an intelligent word recognition module which can be used to digitize natural history specimens. Herbarium specimens are pressed plant samples stored on paper, and are an invaluable source of information for climate change scientists, providing key insights into biodiversity change in the age of the Anthropocene. 

One of the major issues facing the Harvard Herbaria - as well as others across the country and world - is that most of this data is inaccessible due to a lack of digitized records. Digitized specimens will facilitate easier dissemination of information and allow more people access to this data.

To date, digitization has been a very time consuming process, owing to the fact that these herbariums would need to hire a number of people to manually inspect and log the information on each pressed plant sample. This approach can also become quite costly, as the number of samples stored in these facilities is quite large. 

Each specimen contains a label that includes the name of the curator, their institution, the species and genus, and the date the specimen was collected. A sizable number of these labels are handwritten and date back to the early 1900s. 

This brought about the idea to leverage modern Optical Character Recognition (OCR) technologies in order to massively speed up the process of fully digitizing these records. This undertaking has already had a number of teams contribute, and in its current form, the project consists of a bidirectional LSTM-RNN which is used to transcribe specimen labels. 

Moving forward, the goals of our project team are to:
* Identify potentially novel approaches to improve the accuracy of the model.
* Consolidate the two pipelines for label extraction and text recognition.
* Work to expand the scope of outputs from the model 
    * In its current form the model only returns the species label, which neglects other important distinguishing features which were detailed above
### A. Problem Statement: 

Which machine learning model architecture can most accurately detect and transcribe the text present on digital herbarium plant sample images? 
### B. Checklist for project completion

1. Get existing pipeline up and running on the SCC
    * Before we can hope to upgrade the current implementation, we need to fully understand how it works, so that we can recognize areas for potential improvements
2. Research the broader OCR environment
    * We will conduct research looking at the most popular OCR implementations currently being deployed 
3. Concurrently with 2, we will look into potential segmentation solutions beyond CRAFT
    * Being able to more effectively segment the regions in images where text is present should make the transcription step more efficient and accurate
4. Look into implementing alternative approaches for OCR from step 2, and compare with the current implementation of Tesseract
    * There are a number of OCR techniques which have been proposed and implemented in other applications; comparing their performance on our specific datasets will be a valuable tool in pushing to improve the accuracy of the pipeline
5. Upgrade the existing Tesseract pipeline 
    * Once we have familiarized ourselves with the current codebase, we can begin the process of updating certain portions 
6. Deploy the most performant model
7. Time permitting, if we have deployed a satisfactory text detector, we can look to extract information beyond just species from the images

### C. Provide a solution in terms of human actions to confirm if the task is within the scope of automation through AI. 

In our initial stakeholder meeting, the current process for digitizing these natural history specimens was detailed. Each presseed plant gets fed into a specially designed camera setup, where its picture gets taken, and a worker would then manually record the species’ relevant information, such as plant name, location of collection, taxon, date of collection etc. The process has been streamlined over the years, but the bottleneck remains people manually inspecting the images, and recording the text data. Our approach would seek to replace the human manually gathering the text information from the images with our pipeline. 

### D. Outline a path to operationalization.

At the end of the project, the process detailed above describing the current method for digitizing these records would ideally be modified as follows:
1. A specimen is placed into the special camera rig, and its picture is taken
2. That picture is immediately fed into the pipeline we have been working on
3. The relevant text in the picture will be extracted, transcribed, and stored along with the image
4. As more images are processed, a database can be constructed with identifying information for every digitized plant sample
5. Researchers can then be given access to these databases

The advantage of this setup comes from providing fast, automated transcription and storage of these digital records. 

One of the points that was emphasized in our first stakeholder meeting was that access to these records would be a valuable tool to researchers all around the world, and getting a functional pipeline constructed would be a great first step towards that goal. 

** **
## Resources

### Data Sets
[CNH Portal for access to herbarium records](https://portal.neherbaria.org/portal/) 

[Pre-1940 plant specimen images in GBIF](https://www.gbif.org/occurrence/gallery?basis_of_record=PRESERVED_SPECIMEN&media_ty[…]axon_key=6&year=1000,1941&advanced=1&occurrence_status=present)

[International Plant Names Index](https://www.gbif.org/dataset/046bbc50-cae2-47ff-aa43-729fbf53f7c5#dataDescription)

[GBIF, the Global Biodiversity Information Facility](https://hosted-datasets.gbif.org/datasets/backbone/current/)

[IPNI, the International Plant Names Index](https://storage.cloud.google.com/ipni-data/)

[IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)

[NIST Special Database 19](https://www.nist.gov/srd/nist-special-database-19)

[Handwriting of Names Dataset](https://www.kaggle.com/datasets/landlord/handwriting-recognition)
### References



1. [CRAFT](https://github.com/clovaai/CRAFT-pytorch)
2. [Tesseract](https://github.com/tesseract-ocr/tesseract)
3. [TrOCR](https://arxiv.org/abs/2109.10282)
4. [EAST](https://arxiv.org/abs/1704.03155)
5. [Survey of OCR Techniques](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9837974)
6. [Segmentation Comparisons](https://openaccess.thecvf.com/content_ICCVW_2019/papers/RLQ/Nguyen_State-of-the-Art_in_Action_Unconstrained_Text_Detection_ICCVW_2019_paper.pdf)
7. [PP-OCR](https://arxiv.org/pdf/2009.09941.pdf)

** **
## Weekly Meeting Updates

[Meeting Notes](https://docs.google.com/document/d/1XtBjMV5cdqOrsAPZufLfqE8shQPHs2okjE9wKT970Uo/edit?usp=sharing)


