# Herbarium_Project

For our final deliverable, we have a pipeline to segment the labels from the full specimen images, and another pipeline that will output label transcriptions
for images it was able to correctly identify. Both work on images/data stored in the global "in_data" folder. 

The segmentation is located in the `segmentation` folder, the transcription is located in the `transcription` folder, and the transfer learning is located in the `training_model` folder with further instructions in each
folder respectively. 

Dependencies can be found in requirements.txt

<br />

## Install Requirements
### Install venv if not done yet
`pip install virtualenv `
### Create virtual environment for lower memory overhead
`virtualenv -p python3.8.10 .env`
### Activate virtual env
`source .env/bin/activate`
### Install requirements
`pip install -r requirements.txt`
### If scraping is needed, install more requirements
`pip install -r scraping/requirements.txt`

<br />

## Allocate GPU for Training
### check if V100 GPU being used
`lshw -C display`
### syscall for executing jobs on GPU, RUN if lshw -C display does not return V100
`qrsh -l gpus=1 -l gpu_type=V100`
### check that GPU updated
`lshw -C display`

<br/>

## Run the CRAFT pipeline
### Bounding boxes for image masks of text boxes
`cd CRAFT/CRAFT-pytorch-master`
### make script executable
`chmod +x bash_submit.sh`
### execute CRAFT detector on HUH images
`./bash_submit.sh`

<br/>

# Add Users
To add yourself to the repository, open a Pull Request modifying `COLLABORATORS`, entering your GitHub username in a newline.

All Pull Requests must follow the Pull Request Template, with a title formatted like such `[Project Name]: <Descriptive Title>`
