# Proof of Concept - Deployment Plan

## File Descriptions
This directory contains the 4 files which demonstrate the training and testing of the models we have been working with. 
1. pp_ocr_inference.ipynb

  This notebook is mainly used for going throught the whole process of running the PP-OCR pipeline, including reading and displaying sample images, using default PP_OCRv3 model to perform detection only, using default PP_OCRv3 model to perform recognition only, using default PP_OCRv3 model to perform both detection and recognition, loading ground truth label and corpuse, and using string grouper to evaluate the performance for all the images in a given directory.

   - How should results be stored? 
     The result for PP_OCR pipeline can be saved in a given directory. it's an optional argument, please refer to the section "Let's use PaddleOCR on both detection and recognization" in this notebook. Note, since there are thousands of image will be processes, so the result won't be saved in batch processing.
   - How about an interface to query the results? Can a result be manually corrected and stored?
     There is a function "def display_OCR_result_with_imgID(imgID)" can be used to extract the image for a given imgID (You can try it in the notebook, it assume you have img_dict store in the local memory). For any case a manual inspection is needed, you can use this function for debugging or analysis purpose!

 2. ppocr_test.py (deprecated version)

   This file shuold be used for batch processing purpose, which allow you to run it in SCC. It contains all the important codes that you need to run the whole OCR pipeline and evaluation. It's less interpretable. If you want to get more understanding about how to use PP_OCR pipeline and see the output that will come out, please refer to [PP_OCR Notebook](./pp_ocr_inference.ipynb) or the [Project Slide](./ML%20Practicum%20Project%20Update.pdf).

3. pp_ocr_deployment.py 

   Contains many useful functions for running pp-ocr algorithm, including loading and displaying images, applying pp-ocr algorithm of a given image, checking invalid images with a given directory, and batch evaluation on all images of a given directory.  For example, if you want to use pp-ocr for a given directory, you need to provide the directory that contains this image, and image filename, and a output directory, as an example shown below:

```python
$ python pp_ocr_deployment.py -d /projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/20220425-160006-matching/ -f 912082001.jpg -r /usr4/dl523/dong760/CS549_Herbarium_Project/ml-herbarium/PaddleOCR/output
```

4. pp_ocr_deployment_notebook.ipynb

   A jupyter notebook shows demo how to use pp_ocr_deployment.py to run pp-ocr algorithm on single image or multiple images in a given directory.

5. batch_eval.py

   A file that use customized data loader in utils.py and ocr algorithm from pp_ocr_deployment.py to perform evaluation on all images scraped from GBIF website.

6. utils.py

   A dependent file that needed for running batch_eval.py