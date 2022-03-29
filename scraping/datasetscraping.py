# %% [markdown]
# # Dataset Scraping

# %% [markdown]
# ## Imports

# %%
from dwca.read import DwCAReader
from dwca.darwincore.utils import qualname as qn
import requests
import shutil
import os
import time
import math
import pandas
import mimetypes
import json
import zipfile


# %% [markdown]
# ## Globals

# %%
timestr = time.strftime("%Y%m%d-%H%M%S")

PERCENT_TO_SCRAPE = 0.00002
NUMBER_TO_SKIP = 40000
DATASET_PATH = "/projectnb/sparkgrp/ml-herbarium-data/"
DATASET_ARCHIVE = "data.zip"
DATASET_CSV = "data.csv"
OUTPUT_PATH = "/projectnb/sparkgrp/ml-herbarium-data/scraped-data/" + timestr + "/"
DATASET_URL = (
    "https://api.gbif.org/v1/occurrence/download/request/0195391-210914110416597.zip"
)


# %% [markdown]
# ## Download Dataset
# #### Only run this if the dataset needs to be redownladed

# %%
def download_dataset():
    if os.path.exists(DATASET_PATH+DATASET_ARCHIVE):
        os.remove(DATASET_PATH+DATASET_ARCHIVE)
    ds = requests.get(DATASET_URL, stream=True)
    with open(DATASET_PATH+DATASET_ARCHIVE, "wb") as f:
        shutil.copyfileobj(ds.raw, f)


# %% [markdown]
# ## For DWCA files

# %% [markdown]
# ### Open DWCA File

# %%
def open_dwca():
    dwca = DwCAReader(DATASET_PATH+DATASET_ARCHIVE)
    return dwca


# %% [markdown]
# ### Test DWCA
# Will throw an error if the file is not opened correctly.

# %%
def test_dwca(dwca):
    print(dwca.get_corerow_by_position(0))


# %% [markdown]
# ### Save DWCA Rows to Pandas Dataframe

# %%
def save_dwca_rows_to_pandas(dwca):
    # df = dwca.pd_read('occurrence.txt')
    df = dwca.pd_read("occurrence.txt", low_memory=False)
    return df


# %% [markdown]
# #### Close the archive to free resources
# 

# %%
def close_dwca(dwca):
    dwca.close()

# %% [markdown]
# ## For CSV files

# %% [markdown]
# ### Save CSV Rows to Pandas Dataframe

# %%
def save_csv_rows_to_pandas():
    with zipfile.ZipFile(DATASET_PATH+DATASET_ARCHIVE, "r") as zip_ref:
        zip_ref.extractall(DATASET_PATH+DATASET_CSV)
    df = pandas.read_csv(DATASET_PATH+DATASET_CSV)
    return df


# %% [markdown]
# ## Print Pandas Column Names

# %%
def print_pandas_column_names(df):
  colnames = []
  for col in df.columns:
      colnames.append(col)
  print(colnames)


# %% [markdown]
# ## Get Images

# %% [markdown]
# ### Export GBIF URLs

# %%
def export_gbif_urls(df):
  data = {}

  NUMBER_TO_SKIP = math.floor(df.shape[0] / (df.shape[0] * PERCENT_TO_SCRAPE))
  NUMBER_TO_SCRAPE = math.ceil(df.shape[0] / NUMBER_TO_SKIP)
  print(str(NUMBER_TO_SCRAPE) + " IDs will be scraped.")
  for i in range(1, df.shape[0], NUMBER_TO_SKIP):
      id = df.at[i, 'id']
      data[i] = {'id':str(id)}
  print('Successfully scraped ' + str(len(data)) + ' IDs.')
  return data

# %% [markdown]
# ### Fetch Image URLs and Specimen Data from GBIF API

# %%
def fetch_gbif_images(data):
  print('Data will be fetched for', len(data), 'occurrences.')
  i = 1
  for idx in data:
      print("\rProgress: " + str(i)+'/'+str(len(data)), end="")
      rq = requests.get('https://api.gbif.org/v1/occurrence/' + str(data[idx]['id']))
      data[idx]['img_url']=(json.loads(rq.content)['media'][0]['identifier'])
      data[idx]['img_type']=(json.loads(rq.content)['media'][0]['format'])
      data[idx]['country']=(json.loads(rq.content)['country'])
      data[idx]['genus']=(json.loads(rq.content)['genus'])
      data[idx]['species']=(json.loads(rq.content)['species'])
      i+=1
  print('\nSuccessfully fetched data for', len(data), 'occurrences.')
  return data

# %% [markdown]
# ### Download Images

# %%
def download_images(data):
  i=1
  if not os.path.exists(OUTPUT_PATH):
      os.makedirs(OUTPUT_PATH)
  for idx in data:
      img = requests.get(data[idx]['img_url'], stream=True)
      with open(OUTPUT_PATH+str(idx)+mimetypes.guess_extension(data[idx]['img_type']),'wb') as f:
          shutil.copyfileobj(img.raw, f)
      print("\rProgress: " + str(i)+'/'+str(len(data)), end="")
      i+=1


# %% [markdown]
# ### Export Geograpy Data

# %%
def export_geography_data(data):
  with open(OUTPUT_PATH+'countries.txt', 'w') as f:
      for idx in data:
          f.write(data[idx]['country']+'\n')
  print('\nSuccessfully wrote countries to file.')


# %% [markdown]
# ### Export Taxon Data

# %%
def export_taxon_data(data):
  with open(OUTPUT_PATH+'taxon.txt', 'w') as f:
      for idx in data:
          f.write(data[idx]['genus']+' '+data[idx]['species']+'\n')
  print('\nSuccessfully wrote taxon to file.')

# %% [markdown]
# ## Print Help Message
def print_help_message():
  print("Please specify a dataset to type using: 'python3 datasetscraping.py <dataset_type> [OPTIONAL ARGS]' where dataset_type is either 'dwca' or 'csv'.")
  print("\nOptional arguments:")
  print("\t-o, --output_path: Specify the output path for the images. Default is './output/'.")
  print("\t-p, --percent_to_scrape: Specify the percentage of the dataset to scrape. Default is 0.1.")
  print("\t-u, --dataset_url: Specify the dataset URL to download a new dataset.")
  print("\t-h, --help: Print this help message.")
  sys.exit(1)
# %% [markdown]
# ## Main Function
if __name__ == "__main__":
  import sys
  args = sys.argv[1:]
  if len(args) == 0:
    print_help_message()
  if args[0] == '-h' or args[0] == '--help' or args[0] != 'dwca' and args[0] != 'csv':
    print_help_message()
  if args.index('-o') > 0:
    OUTPUT_PATH = args[args.index('-o')+1]
  if args.index('-p') > 0:
    PERCENT_TO_SCRAPE = float(args[args.index('-p')+1])
  if args.index('-u') > 0:
    DATASET_URL = args[args.index('-u')+1]
  if args[0] == 'dwca':
    print('Scraping dataset from Darwin Core Archive.')
    print('Output path: ' + OUTPUT_PATH)
    print('Opening Darwin Core Archive...')
    dwca = open_dwca(DATASET_PATH+DATASET_ARCHIVE)
    print('Successfully opened Darwin Core Archive.')
    print('Saving rows to Pandas DataFrame...')
    df = save_dwca_rows_to_pandas(dwca)
    print('')
    print('Successfully saved rows to Pandas DataFrame.')
    print('Exporting data to output path...')
    data = export_gbif_urls(df)
    data = fetch_gbif_images(data)
    download_images(data)
    export_geography_data(data)
    export_taxon_data(data)
    print('Successfully exported data to output path. Done!')
  elif args[0] == 'csv':
    print('Scraping dataset from CSV.')
    print('Output path: ' + OUTPUT_PATH)
    print('Opening CSV...')
    df = save_csv_rows_to_pandas(DATASET_PATH+DATASET_ARCHIVE)
    print('Successfully saved CSV rows to Pandas DataFrame.')
    print('Exporting data to output path...')
    data = export_gbif_urls(df)
    data = fetch_gbif_images(data)
    download_images(data)
    export_geography_data(data)
    export_taxon_data(data)
    print('Successfully exported data to output path. Done!')