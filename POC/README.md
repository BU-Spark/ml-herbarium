# Proof of Concept 

## File Descriptions
This directory contains the 4 files which demonstrate the training and testing of the models we have been working with. 
1. trocr_test.ipynb
This notebook shows the process of setting up craft for segmentation, extracting all bounding boxes around text within the image. These segmented images are then passed through the Tr-OCR model to perform text recognition; we save the transcriptions and model confidence for each of these calculations. Then we perform string matching against a set of files (Taxon, Species, Genus, Country, and Subdivisions of countries) in order to find the best match for each.  All of this information is then accrued in a final output dataframe, which contains the bounding boxes and transciptions for every segmentation, as well as the location of the closest match found between all reference files. We then test the accuracy of the pipeline on roughly 1000 images. 
2. trocr_train_wab.ipynb
This file is the main training resource for the Tr-OCR model. We have opted to evaluate all available pre-trained models, fine tuning each on the IAM handwriting dataset. All of the training and validation information is logged using Weights and Biases. 
3. pp_ocr_inference.ipynb
This notebook is mainly used for going throught the whole process of running the PP-OCR pipeline, including reading and displaying sample images, using default PP_OCRv3 model to perform detection only, using default PP_OCRv3 model to perform recognition only, using default PP_OCRv3 model to perform both detection and recognition, loading ground truth label and corpuse, and using string grouper to evaluate the performance for all the images in a given directory.

4. ppocr_test.py
This file shuold be used for batch processing purpose, which allow you to run it in SCC. It contains all the important codes that you need to run the whole OCR pipeline and evaluation. It's less interpretable. If you want to get more understanding about how to use it and see the output that PP_OCR pipeline will come out, please refer to pp_ocr_inference.ipynb or the [ML Practicum Project Update.pdf]() slide.
