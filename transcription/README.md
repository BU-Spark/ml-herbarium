### Specimen transcription

NOTE: 
This script relies on running the specimen images through the CRAFT detector first, found as a subdirectory in this project directory.  
Documentation found here: https://github.com/clovaai/CRAFT-pytorch
This model also requires installing the AWS MXNET library and its dependencies:
Documentation found here: https://github.com/awslabs/handwritten-text-recognition-for-apache-mxnet


To run CRAFT, enter the CRAFT directory and run:
```
python test.py --trained_model=craft_mlt_25k.pth --test_folder="../in_data"
```

Once CRAFT has processed the input data, this script can be run with:
```
python transcribe_labels.py
```
