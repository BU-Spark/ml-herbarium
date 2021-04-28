### Specimen transcription

NOTE: 
This script relies on running the specimen images through the CRAFT detector first, found as a subdirectory in this project directory.  
Documentation found here: https://github.com/clovaai/CRAFT-pytorch
This model also requires installing the AWS MXNET library and its dependencies:
Documentation found here: https://github.com/awslabs/handwritten-text-recognition-for-apache-mxnet


Once CRAFT has been run, this script will print out the determined specimen label for each image
python test.py --trained_model=[weightfile] --test_folder=

The script can be run with:
```
python transcribe_labels.py
```
