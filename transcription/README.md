### Specimen transcription

NOTE: 
This script relies on running the specimen images through the CRAFT detector first, found as a subdirectory in this project directory.<br />
Documentation found here: https://github.com/clovaai/CRAFT-pytorch

This model also requires installing the AWS MXNET library and its dependencies: <br />
Documentation found here: https://github.com/awslabs/handwritten-text-recognition-for-apache-mxnet


To run CRAFT, change to the CRAFT directory and run:
```
python test.py --trained_model=craft_mlt_25k.pth --test_folder="../../in_data"
```
This will process the images in the global "in_data" folder <br />

Once CRAFT has processed the input data, come back to this directory and run:
```
python transcribe_labels.py
```
This script will take those text boxes outputted by CRAFT along with the original images, to output the labels of the images it was able to transcribe into
this folder as "results.txt". Images one wants to run through the pipeline must be added to the `in_data` folder, along with the corresponding corpus text file. 
