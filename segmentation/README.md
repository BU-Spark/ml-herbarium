### Label and Line Segmentation

NOTE: This script relies on running the specimen images through the CRAFT detector first. 
Documentation can be found here: https://github.com/clovaai/CRAFT-pytorch

Once the text boxes have been obtained from CRAFT (as the .txt file), all that needs to be changed in the script are the directory paths. The script will
import the original images along with the CRAFT text boxes and be able to output the cropped label with a high probability. To do so, it takes the text 
boxes, expands them, combines overlapping boxes, and keeps the largest resulting box as the label; through it can be reprogrammed to keep as many other 
resulting combined boxes to have an even higher probability of retaining the label. The script also contains code that will crop out sub-images of the lines 
of text using CRAFT's text boxes, which can then be used in models such as the AWS MXNET for testing or training. 


To run CRAFT, enter the CRAFT directory and run:
```
python test.py --trained_model=craft_mlt_25k.pth --test_folder="../in_data"
```
This will process the images in the global "in_data" folder 


Then, this script can be run and will create a `labels` folder where it will output the cropped labels.
```
python segmentation.py
```

It will use the text boxes output by CRAFT and segment out the largest blob it finds, which 95% of the time will be the requisite label.
