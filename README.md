# Herbarium_Project

For our final deliverable, we have a pipeline to segment the labels from the full specimen images, and another pipeline that will output label transcriptions
for images it was able to correctly identify. Both work on images/data stored in the global "in_data" folder. 

The segmentation is located in the `segmentation` folder, the transcription is located in the `transcription` folder, and the transfer learning is located in the `training_model` folder with further instructions in each
folder respectively. 

Dependencies can be found in requirements.txt

## Instructions for running on the SCC
Run the following commands: (the module load command is tailored for the SCC; skip/modify this command if don't have / don't need some modules)

```
# check if V100 GPU being used
lshw -C display
# syscall for executing jobs on GPU, RUN if lshw -C display does not return V100
qrsh -l gpus=1 -l gpu_type=V100 
# if first time installing requirements,
rm -rf ~/.local/lib/python3.8
# pip cache not required for 3.8.10
module load python3/3.8.10
# unset this variable to let pip access it
unset PIP_NO_CACHE_DIR
# clear the cache
pip cache purge
# load other necessary modules
module load mxnet/1.7.0
module load pytorch/1.10.2
git clone https://github.com/mzheng27/Herbarium_Project
cd Herbarium_Project
# install venv if not done yet
pip install virtualenv
# create virtual environment for lower memory overhead
python3 -m venv env
# activate virtual env
source env/bin/activate
# necessary installs
pip install -r requirements.txt
# bounding boxes for image masks of text boxes
cd CRAFT/CRAFT-pytorch-master
# make script executable
chmod +x bash_submit.sh
# execute CRAFT detector on HUH images
./bash_submit.sh

```

# Add Users
To add yourself to the repository, open a Pull Request modifying `COLLABORATORS`, entering your GitHub username in a newline.

All Pull Requests must follow the Pull Request Template, with a title formatted like such `[Project Name]: <Descriptive Title>`
