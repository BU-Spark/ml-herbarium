# Model Fine Tuning
To train the model, first clone the tesstrain repo in your home directory:
``` bash
git clone https://github.com/tesseract-ocr/tesstrain.git
```

Be sure you have the dependencies loaded:
``` bash
module load leptonica/1.82.0
module load libicu/71.1
module load tesseract/4.1.3
```
Download the training dataset, `lines.tgz`,  and the groundtruth dataset, `ascii.tgz`, and put it in the `tesseract-training` directory from:
https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database

Generate the training data:
``` bash
cd transcription/handwriting_tesseract_training/
python3 generate_training_data.py
```

Then run the training (for SCC):

``` bash
cd $HOME/tesstrain
make training MODEL_NAME=handwriting-eng START_MODEL=eng DATA_DIR=/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training GROUND_TRUTH_DIR=$HOME/ml-herbarium/transcription/handwriting_tesseract_training/gt PSM=7 TESSDATA=$HOME/ml-herbarium/transcription/handwriting_tesseract_training/tessdata
```

The options for make training are:
```
    MODEL_NAME         Name of the model to be built. Default: foo
    START_MODEL        Name of the model to continue from. Default: ''
    PROTO_MODEL        Name of the proto model. Default: OUTPUT_DIR/MODEL_NAME.traineddata
    WORDLIST_FILE      Optional file for dictionary DAWG. Default: OUTPUT_DIR/MODEL_NAME.wordlist
    NUMBERS_FILE       Optional file for number patterns DAWG. Default: OUTPUT_DIR/MODEL_NAME.numbers
    PUNC_FILE          Optional file for punctuation DAWG. Default: OUTPUT_DIR/MODEL_NAME.punc
    DATA_DIR           Data directory for output files, proto model, start model, etc. Default: data
    OUTPUT_DIR         Output directory for generated files. Default: DATA_DIR/MODEL_NAME
    GROUND_TRUTH_DIR   Ground truth directory. Default: OUTPUT_DIR-ground-truth
    CORES              No of cores to use for compiling leptonica/tesseract. Default: 4
    LEPTONICA_VERSION  Leptonica version. Default: 1.78.0
    TESSERACT_VERSION  Tesseract commit. Default: 4.1.1
    TESSDATA_REPO      Tesseract model repo to use (_fast or _best). Default: _best
    TESSDATA           Path to the .traineddata directory to start finetuning from. Default: ./usr/share/tessdata
    MAX_ITERATIONS     Max iterations. Default: 10000
    EPOCHS             Set max iterations based on the number of lines for training. Default: none
    DEBUG_INTERVAL     Debug Interval. Default:  0
    LEARNING_RATE      Learning rate. Default: 0.0001 with START_MODEL, otherwise 0.002
    NET_SPEC           Network specification. Default: [1,36,0,1 Ct3,3,16 Mp3,3 Lfys48 Lfx96 Lrx96 Lfx256 O1c\#\#\#]
    FINETUNE_TYPE      Finetune Training Type - Impact, Plus, Layer or blank. Default: ''
    LANG_TYPE          Language Type - Indic, RTL or blank. Default: ''
    PSM                Page segmentation mode. Default: 13
    RANDOM_SEED        Random seed for shuffling of the training data. Default: 0
    RATIO_TRAIN        Ratio of train / eval training data. Default: 0.90
    TARGET_ERROR_RATE  Stop training if the character error rate (CER in percent) gets below this value. Default: 0.01
```
