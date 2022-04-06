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
import json
import zipfile
from tqdm import tqdm
from glob import glob
import multiprocessing as mp


# %% [markdown]
# ## Globals

# %%
timestr = time.strftime("%Y%m%d-%H%M%S")

PERCENT_TO_SCRAPE = 0.00015
NUMBER_TO_SKIP = 40000
DATASET_PATH = "/projectnb/sparkgrp/ml-herbarium-data/"
DATASET_ARCHIVE = "data.zip"
DATASET_CSV = "data.csv"
OUTPUT_PATH = "/projectnb/sparkgrp/ml-herbarium-data/scraped-data/" + timestr + "/"
DATASET_URL = "https://occurrence-download.gbif.org/occurrence/download/request/0196625-210914110416597.zip"
NUM_CORES = min(mp.cpu_count(), 50)
DATA = None


# %% [markdown]
# ## Download Dataset
# #### Only run this if the dataset needs to be redownladed

# %%
def download_dataset():
    if os.path.exists(DATASET_PATH + DATASET_ARCHIVE):
        os.remove(DATASET_PATH + DATASET_ARCHIVE)
    ds = requests.get(DATASET_URL, stream=True)
    total_size_in_bytes = int(ds.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(DATASET_PATH + DATASET_ARCHIVE, "wb") as f:
        for data in ds.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


# %% [markdown]
# ## For DWCA files

# %% [markdown]
# ### Open DWCA File

# %%
def open_dwca():
    dwca = DwCAReader(DATASET_PATH + DATASET_ARCHIVE)
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
    if os.path.exists(DATASET_PATH + "*.csv"):
        os.remove(DATASET_PATH + "*.csv")
    with zipfile.ZipFile(DATASET_PATH + DATASET_ARCHIVE, "r") as zip_ref:
        zip_ref.extractall(DATASET_PATH)
    f = glob(os.path.join(DATASET_PATH, "*-*.csv"))[0]
    os.rename(f, DATASET_PATH + DATASET_CSV)
    print("CSV file extracted")
    df = pandas.read_csv(DATASET_PATH + DATASET_CSV, sep="\t")
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
        if args[0] == "dwca":
            id = df.at[i, "id"]
        elif args[0] == "csv":
            id = df.at[i, "gbifID"]
        data[i] = {"id": str(id)}
    print("Successfully scraped " + str(len(data)) + " IDs.")
    return data


# %%
def scrape_occurrence(key):
    rq = requests.get("https://api.gbif.org/v1/occurrence/" + str(DATA[key]["id"]))
    return_dict = {}
    return_dict[key] = {}
    return_dict[key]["img_url"] = json.loads(rq.content)["media"][0]["identifier"]
    return_dict[key]["img_type"] = json.loads(rq.content)["media"][0]["format"]
    return_dict[key]["country"] = json.loads(rq.content)["country"]
    return_dict[key]["genus"] = json.loads(rq.content)["genus"]
    return_dict[key]["species"] = json.loads(rq.content)["species"]
    return return_dict


# %% [markdown]
# ### Fetch Image URLs and Specimen Data from GBIF API
def fetch_data(data):
    DATA = data
    print("Starting multiprocessing...")
    pool = mp.Pool(NUM_CORES)
    print("Fetching data...")
    for item in tqdm(pool.imap(scrape_occurrence, data), total=len(data)):
        data.update(item)
    pool.close()
    pool.join()

    print("\nSuccessfully fetched data for", len(data), "occurrences.")
    DATA = None
    return data


# %%
def download(key):
    img = requests.get(DATA[key]["img_url"], stream=True)
    with open(
        OUTPUT_PATH + str(key) + "." + data[1]["img_type"].split("/", 1)[1], "wb"
    ) as f:
        shutil.copyfileobj(img.raw, f)


# %% [markdown]
# ### Download Images
def download_images(data):
    DATA = data
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    print("Starting multiprocessing...")
    pool = mp.Pool(NUM_CORES)
    print("Downloading images...")
    for _ in tqdm(pool.imap(download, data), total=len(data)):
        pass
    pool.close()
    DATA = None


# %% [markdown]
# ### Export Geograpy Data

# %%
def export_geography_data(data):
    with open(OUTPUT_PATH + "countries.txt", "w") as f:
        for idx in data:
            f.write(data[idx]["country"] + "\n")
    print("\nSuccessfully wrote countries to file.")


# %% [markdown]
# ### Export Taxon Data

# %%
def export_taxon_data(data):
    with open(OUTPUT_PATH + "taxon.txt", "w") as f:
        for idx in data:
            f.write(data[idx]["genus"] + " " + data[idx]["species"] + "\n")
    print("\nSuccessfully wrote taxon to file.")


# %% [markdown]
# ## Print Help Message
def print_help_message():
    print(
        "Please specify a dataset to type using: 'python3 datasetscraping.py <dataset_type> [OPTIONAL ARGS]' where dataset_type is either 'dwca' or 'csv'."
    )
    print("\nOptional arguments:")
    print(
        "\t-o, --output_path: Specify the output path for the images. Default is './output/'."
    )
    print(
        "\t-p, --percent_to_scrape: Specify the percentage of the dataset to scrape. Default is 0.00015 (~1198 occurrences)."
    )
    print("\t-u, --dataset_url: Specify the dataset URL to download a new dataset.")
    print("\t-c, --num_cores: Specify the number of cores to use. Default is 50.")
    print("\t-h, --help: Print this help message.")
    sys.exit(1)


# %% [markdown]
# ## Main Function
if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if len(args) == 0:
        print_help_message()
    if args[0] == "-h" or args[0] == "--help" or args[0] != "dwca" and args[0] != "csv":
        print_help_message()
    if args.count("-o") > 0:
        OUTPUT_PATH = args[args.index("-o") + 1]
    if args.count("-p") > 0:
        PERCENT_TO_SCRAPE = float(args[args.index("-p") + 1])
    if args.count("-u") > 0:
        DATASET_URL = args[args.index("-u") + 1]
    if args.count("-c") > 0:
        NUM_CORES = int(args[args.index("-c") + 1])
    if args.count("--output_path") > 0:
        OUTPUT_PATH = args[args.index("--output_path") + 1]
    if args.count("--percent_to_scrape") > 0:
        PERCENT_TO_SCRAPE = float(args[args.index("--percent_to_scrape") + 1])
    if args.count("--dataset_url") > 0:
        DATASET_URL = args[args.index("--dataset_url") + 1]
    if args.count("--num_cores") > 0:
        NUM_CORES = int(args[args.index("--num_cores") + 1])
    if args[0] == "dwca":
        print("Scraping dataset from Darwin Core Archive.")
        print("Output path: " + OUTPUT_PATH)
        print("Opening Darwin Core Archive...")
        dwca = open_dwca()
        print("Successfully opened Darwin Core Archive.")
        print("Saving rows to Pandas DataFrame...")
        df = save_dwca_rows_to_pandas(dwca)
        print("")
        print("Successfully saved rows to Pandas DataFrame.")
        print("Exporting data to output path...")
        data = export_gbif_urls(df)
        download_images(data)
        export_geography_data(data)
        export_taxon_data(data)
        print("Successfully exported data to output path. Done!")
    elif args[0] == "csv":
        print("Scraping dataset from CSV.")
        print("Output path: " + OUTPUT_PATH)
        print("Opening CSV...")
        df = save_csv_rows_to_pandas()
        print("Successfully saved CSV rows to Pandas DataFrame.")
        print("Exporting data to output path...")
        data = export_gbif_urls(df)
        download_images(data)
        export_geography_data(data)
        export_taxon_data(data)
        print("Successfully exported data to output path. Done!")
