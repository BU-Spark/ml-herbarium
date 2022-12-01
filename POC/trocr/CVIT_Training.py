import os
# Getting all the models off my home directory for space concerns
os.environ['TRANSFORMERS_CACHE'] = '/projectnb/sparkgrp/colejh'

# Imports
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm_notebook as tqdm
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import utilities

save_dir = '/projectnb/sparkgrp/colejh/saved_results/'

#suppressing all the huggingface warnings
SUPPRESS = True
if SUPPRESS:
    from transformers.utils import logging
    logging.set_verbosity(40)

# Ignoring UserWarning and FutureWarning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


directory = '/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/training/CVIT/Images_90K_Normalized/'
count = 0
all_img = []
all_labels = []
no_tag = []
# list_dir = os.listdir(directory)

for (dirpath, dirnames, filenames) in os.walk(directory):
    for file in filenames:
        if file.endswith('.png'):
            image = os.path.join(dirpath, file)
#             display(Image.open(image))
#             if catch_invalid(image):
            try:
                with open(image.split('.')[0]+'.gt.txt') as f:
                    contents = f.readlines()
                all_img.append(image)
                all_labels.append(contents[0])
            except FileNotFoundError as f:
#                             print(f)
                    no_tag.append(image)
                    break # some folders have no ground truth

        count+=1
        if count%10000 == 0:
            print(count)
            

# create dataframe from all_img and all_labels
df = pd.DataFrame({'image':all_img, 'label':all_labels})


# Check for valid images
pool = mp.Pool(mp.cpu_count())

bad = []
for output in tqdm(pool.imap(utilities.catch_invalid2, df['image']), total=len(df['image']),desc = 'Checking for Valid Images'):
    if output == False:
#         bad.append(output[1])
        print(output)

# Delete the bad images from the dataframe
df = df[~df['image'].isin(bad)]

# save dataframe to pickle at specified directory
df.to_pickle(save_dir+'cvit.pkl')

# Read in the dataframe (uncomment if above already run)
# df = pd.read_pickle(save_dir + 'cvit.pkl')

# Show a few random images and labels from CVIT
for i in range(10):
    utilities.show_random_CVIT_image(df)

# Setting up the processor, model and training and validation datasets
model_name = 'microsoft/trocr-base-printed'
processor = TrOCRProcessor.from_pretrained(model_name) 
train_dataset,val_dataset = utilities.create_CVIT(df,processor)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)

model = VisionEncoderDecoderModel.from_pretrained(model_name)


# Checking if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Putting the model on the GPU
# model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
model= nn.DataParallel(model,list(range(torch.cuda.device_count()))).to(device)

# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=4)

# set special tokens used for creating the decoder_input_ids from the labels
model.module.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.module.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.module.config.vocab_size = model.module.config.decoder.vocab_size

# set beam search parameters
model.module.config.eos_token_id = processor.tokenizer.sep_token_id
model.module.config.max_length = 64
model.module.config.early_stopping = True
model.module.config.no_repeat_ngram_size = 3
model.module.config.length_penalty = 2.0
model.module.config.num_beams = 4


# Optimizer and Scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
#     scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, eta_min=0.003)
# Main Training Loop
avg_train_loss = []
avg_val_loss = []
last_val = float('inf')
min_val = float('inf')
patience = 2
EPOCHS = 15
counter = 0
for epoch in range(EPOCHS):  # loop over the dataset multiple times
    # train
    model.train()
    train_loss = 0.0
    running_loss = 0
    for i,batch in enumerate(tqdm(train_dataloader,desc = 'Epoch '+str(epoch+1)),start=0,):
#         batch = {k:v.to(device) for k,v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.sum().backward()
        optimizer.step()
        train_loss += torch.sum(loss).detach().cpu().numpy()
        
        # print statistics and current decayed learning rate
        running_loss += loss.sum().item()
        if i % 10000 == 9999:    # print every 10000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10000))
            running_loss = 0.0
            
    # Save loss locally
    current_loss = train_loss/len(train_dataloader)
    avg_train_loss.append(current_loss)
    print(f"Training loss after epoch {epoch+1}:", current_loss)
#         scheduler.step()

    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            val_loss += torch.sum(loss).detach().cpu().numpy()

        # if validation loss is lower than previous, save model
        current_val_loss = val_loss/len(val_dataloader)
        avg_val_loss.append(current_val_loss)
        print(f"Validation loss after epoch {epoch+1}:", current_val_loss)

        #if validation loss increases for patience epochs, stop training
        if current_val_loss > last_val:
            counter += 1
            last_val = current_val_loss
            if counter == patience:
                print('Validation loss has increased for {} epochs, ending training...'.format(patience))
                avg_val_loss.append(last_val)
                break
        else:
            counter = 0
            if current_val_loss<min_val:
                min_val = current_val_loss
                # saving the best model so far
                torch.save(model.state_dict(), model_directory + model_name.split('/')[1]+'_testing_best_model.pt')
        last_val = current_val_loss

print('Finished Training {}'.format(model_name.split('/')[1]))



directory = '/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/training/CVIT/'
print(utilities.get_size(directory), 'bytes')