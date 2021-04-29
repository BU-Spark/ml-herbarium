### Training 
This script uses transfer learning on a pretrained CNN-bi-LSTM network (```checkpoint_models/handwriting_line8.params```) with the data provided in the ```training_model/training_data/``` folder. Currently, there are 824 images of species names and ground truth in our transfer learning dataset. 

To run the training script, 
```
cd training_model
python train_model.py
```
The weights of the new models will be save to the checkpoint_models directory. 
