# Herbarium_Project

For our final deliverable, we have a pipeline to segment the labels from the full specimen images, and another pipeline that will output label transcriptions
for images it was able to correctly identify. Both work on images/data stored in the global "in_data" folder. 

The segmentation is located in the `segmentation` folder, the transcription is located in the `transcription` folder, and the transfer learning is located in the `training_model` folder with further instructions in each
folder respectively. 

Dependencies can be found in requirements.txt

<br />

## Install Requirements
### Add Python 3.8.10 to your .bashrc
`nano ~/.bashrc`
Add this to the file: `module load python3/3.8.10`
ctrl+x then return to save and exit
### Create virtual environment for lower memory overhead
`python3 -m venv .env`
### Activate virtual env
`source .env/bin/activate`
### Install requirements
`pip install -r requirements.txt`

## Note for VS Code
To select the correct Python interpreter, open your command palette (Command or Control+Shift+P), select `Python: Select Interpreter` then choose `Python 3.8.10` at path `~/.env/bin/python3.8`.

## Install Leptonica
In your home directory, run:
`git clone https://github.com/DanBloomberg/leptonica.git --depth 1`
`cd leptonica`
`./autogen.sh`
`./configure --prefix=$HOME/.local --disable-shared`
`make`
`make install`

## Install Tesseract
In your home directory, run:
`wget https://github.com/tesseract-ocr/tesseract/archive/4.0.0.tar.gz -O tesseract-4.0.0.tar.gz`
`tar zxvf tesseract-4.0.0.tar.gz`
`export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig`
`./autogen.sh`
`./configure --prefix=$HOME/.local --disable-shared`
`make`
`make install`
`cd ~/.local/share/tessdata`
`wget https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/eng.traineddata`

<br />

## Allocate GPU for Training on SCC
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
