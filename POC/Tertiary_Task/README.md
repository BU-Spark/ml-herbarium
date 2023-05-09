# ML-Herbarium Tertiary Task

This section of the repository deals with the instance segmentation task. We use the repo stored [here](/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/Tertiary_Task/Swin-Transformer-Object-Detection/scc_jobs)

The current model can be found in `./work_dirs/flowers_36_epochs_update/epoch_36.pth` and the visualized outputs are in `outputs/plants_36_epochs_segm`

Dependencies can be found in `requirements.txt`

<br />

## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.

You can also use the script I have written in the event the guide does not work it can be found at `scc_jobs/job_install_packages.sh`.

We then run the file `scc_jobs/job_finetune_2.sh` this will go about downloading the model and finetuning using the data.

The model is [model weight](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_small_patch4_window7.pth) and the config file is [config file](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py)

This repository can be found under `/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/Tertiary_Task/Swin-Transformer-Object-Detection`. Because the mmdetection repo has to be cloned I have placed the data at `/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/Tertiary_Task/Swin-Transformer-Object-Detection/data`, when finetuning the model it will need to be place in `/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/Tertiary_Task/Swin-Transformer-Object-Detection/mmdetection/data`.

### Data Visualization

In order to visualize the data I have created a script `scc_jobs/job_flower_test.sh` that will create the visualizations seen in `outputs/plants_36_epochs_segm`
