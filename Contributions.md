**Kabilan Mohanraj:**
1.	Debugged the existing TrOCR pipeline (string similarity matching part). Could not achieve a resolution.
2.	Worked with Rithik to extract an evaluation dataset from GBIF, upon which we carried out analysis to see the spread of the data, so as to establish a mutually agreeable evaluation dataset between the clients and us.
3. Worked to implement the `TextFuseNet` model for text detection but ran into dependency issues (with the environment setup) as the codebase is about 3 years old and the dependencies used are even older. We've tried using the provided Docker file (not an image) to deploy the pipeline, but the dependencies within the Docker image we built are not compatible with the CUDA drivers on the host machine.
4. Worked on the Named Entity Recognition model, as the post-OCR step. Worked to deploy a new pipeline TaxoNerd which is very recent. The model uses BioBERT model trained on taxons. TaxoNerd is the most recent development in our task and has been added to the existing TrOCR pipeline replacing string similarity matching used in Fall 2022 semester.


**Trived Katragadda:**
1. Added to the already present EDA code so that it will be more representative by including more factors thereby presenting a more clear picture of how the dataset is looking and also where we can look for better insights.
2. Deployed the IELT model for the classification purpose in the secondary task which is now the state of the art model for fine grain image classification specifically for the Oxford flower dataset which is a similar one to ours.
3. Trained the model on cub and flower datasets and replicated the results produced in the paper.
4. In the process of adjusting the parental code so that it can accommodate our custom dataset so that we can train on it and get the classification results.


**Rithik Bhandary:**

1.	Debugged the EDA code to handle download failures and multiprocessing issues.
2.	Downloaded 119,000 images for the secondary classification task from GBIF by filtering based on the New England dataset (dataset provided by the client).
3.	Performed the masking of the labels in these images using KerasOCR pipeline, so that there is no bias influencing the classifier.
4.	Created the evaluation dataset of 4K images from GBIF for the primary task.
5.	Prepared the data for the IELT model based on its requirements.


**Keanu Nichols:** 

1. Worked on the instance segmentation task. Whereby I attempted using two different models the EVA model and the Swin Transformer. 
2. The EVA model did not perform well as it had VRAM issues, it required me to convert the plant dataset to a COCO like format. 
3. For the Swin transformer we were able to perform instance segmentation by taking the Cascade Mask R-CNN model they had and then finetuned it on the plant dataset. Also, visualized the output.

**Eamon Niknafs:**

Project manager & team lead, wrote one of the initial implementations of the project, and managed the team's progress thereafter.