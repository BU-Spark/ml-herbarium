# Proof of Concept - Deployment Plan

## File Descriptions
This directory contains the 4 files which demonstrate the training and testing of the models we have been working with. 
1. trocr_test.ipynb
This notebook shows the process of setting up craft for segmentation, extracting all bounding boxes around text within the image. These segmented images are then passed through the Tr-OCR model to perform text recognition; we save the transcriptions and model confidence for each of these calculations. Then we perform string matching against a set of files (Taxon, Species, Genus, Country, and Subdivisions of countries) in order to find the best match for each.  All of this information is then accrued in a final output dataframe, which contains the bounding boxes and transciptions for every segmentation, as well as the location of the closest match found between all reference files. We then test the accuracy of the pipeline on roughly 1000 images. 
2. trocr_train_wab.ipynb
This file is the main training resource for the Tr-OCR model. We have opted to evaluate all available pre-trained models, fine tuning each on the IAM handwriting dataset. All of the training and validation information is logged using Weights and Biases. 
3. pp_ocr_inference.ipynb
This notebook is mainly used for going throught the whole process of running the PP-OCR pipeline, including reading and displaying sample images, using default PP_OCRv3 model to perform detection only, using default PP_OCRv3 model to perform recognition only, using default PP_OCRv3 model to perform both detection and recognition, loading ground truth label and corpuse, and using string grouper to evaluate the performance for all the images in a given directory.
    - How should results be stored? 
      The result for PP_OCR pipeline can be saved in a given directory. it's an optional argument, please refer to the section "Let's use PaddleOCR on both detection and recognization" in this notebook. Note, since there are thousands of image will be processes, so the result won't be saved in batch processing.
    - How about an interface to query the results? Can a result be manually corrected and stored?
      There is a function "def display_OCR_result_with_imgID(imgID)" can be used to extract the image for a given imgID (You can try it in the notebook, it assume you have img_dict store in the local memory). For any case a manual inspection is needed, you can use this function for debugging or analysis purpose!


4. ppocr_test.py
This file shuold be used for batch processing purpose, which allow you to run it in SCC. It contains all the important codes that you need to run the whole OCR pipeline and evaluation. It's less interpretable. If you want to get more understanding about how to use PP_OCR pipeline and see the output that will come out, please refer to [PP_OCR Notebook](./pp_ocr_inference.ipynb) or the [Project Slide](./ML%20Practicum%20Project%20Update.pdf).

5. pp_ocr_deployment.py
Contains code for running pp-ocr algorithm. You can run it by entering the command with a src_path to the image and output_dir where you want to save the result, e.g., $ python deployment.py -s /projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/20220425-160006-matching/912082001.jpg -r /usr4/dl523/dong760/CS549_Herbarium_Project/ml-herbarium/PaddleOCR/output

6. pp_ocr_deployment_notebook.ipynb
A jupyter notebook shows demo how to run pp_ocr_deployment.py, a more interactive version.

7. batch_eval.py
A file that use customized data loader in utils.py to perform evaluation on all images scraped from GBIF website.

8. utils.py
A dependent file that needed for running batch_eval.py
