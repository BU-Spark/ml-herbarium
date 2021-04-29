# Herbarium_Project

For our final deliverable, we have a pipeline to segment the labels from the full specimen images, and another pipeline that will output label transcriptions
for images it was able to correctly identify. Both work on images/data stored in the global "in_data" folder. 

The segmentation is located in the `segmentation` folder, the transcription is located in the `transcription` folder, and the transfer learning is located in the `training_model` folder with further instructions in each
folder respectively. 

Dependencies can be found in requirements.txt

## Instructions for running on the SCC
Run the following commands: (the module load command is tailored for the SCC; skip/modify this command if your system does not have all of the following modules)

```
module load python3/3.8.6 mxnet/1.7.0 opencv/4.5.0 pytorch/1.8.1
git clone https://github.com/mzheng27/Herbarium_Project
cd Herbarium_Project
pip install -r requirements.txt
```
