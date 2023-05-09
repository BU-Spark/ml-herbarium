# ML-Herbarium Tertiary Task Swin-Transformer

The original repo can be found [here](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)
This section of the repository deals with the instance segmentation task. We use the repo stored [here](/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/Tertiary_Task/Swin-Transformer-Object-Detection/) on the scc server under sparkgrp

The current model can be found in `./work_dirs/flowers_36_epochs_update/epoch_36.pth` and the visualized outputs are in `outputs/plants_36_epochs_segm`

Dependencies can be found in `requirements.txt`

<br />

## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.

You can also use the script I have written in the event the guide does not work it can be found at `scc_jobs/job_install_packages.sh`.

We then run the file `scc_jobs/job_finetune_2.sh` this will go about downloading the model and finetuning using the data.

The model is [model weight](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_small_patch4_window7.pth) and the config file is [config file](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py)

Because the mmdetection repo has to be cloned I have placed the data at `/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/Tertiary_Task/Swin-Transformer-Object-Detection/data`, when finetuning the model it will need to be place in `/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/Tertiary_Task/Swin-Transformer-Object-Detection/mmdetection/data`.

### Data Visualization

In order to visualize the data I have created a script `scc_jobs/job_flower_test.sh` that will create the visualizations seen in `outputs/plants_36_epochs_segm`




# ML-Herbarium Tertiary Task EVA

This is based on the [repo](https://github.com/baaivision/EVA/tree/master/EVA-01/det) and instructions can be found there in terms of installing the EVA model. One major issue was that when finetuning the model it went from the base 4 GB to 12 GB. When searching online we found that this maybe due to the code from detectron2 using floating-point 32.

I also have a file under `scc_jobs/create_dataset.py` which can help with creating the plants json file to format needed for EVA.
We use the repo stored [here](/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/Tertiary_Task/EVA)

Currently I was not able to debug the model in order to test it as I encountred VRAM issues. I would recommend using the guide [here](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5) in order to format the data into the COCO like style.

Once you have the data in the correct format you can look at changing the configurations in `tools/lazyconfig_train_net.py` or start from scratch and use my code as a guide as to what I have done. I have updated some of the configs that allow you to take the new plant dataset however it may be better to use the original EVA code and paste mine in. As I was also testing with the ballon dataset.


