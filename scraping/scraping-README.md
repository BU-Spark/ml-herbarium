# ML-Herbarium Documentation

# datasetscraping.py

ml-herbarium/scraping/dataset/datasetscraping.py

## download_dataset()

Downloads the dataset from the URL specified in the global variables.
```
Parameters
----------
None

Returns
-------
None

Raises
------
None
```



## open_dwca()
Open a Darwin Core Archive (DwCA) file.

```
Parameters
----------
DATASET_PATH : str
    The path to the DwCA file.
DATASET_ARCHIVE : str
    The name of the DwCA file.

Returns
-------
dwca : DwCAReader
    The DwCAReader object.
```



## test_dwca()
This function tests if the DwCA file was correctly opened.



## save_dwca_rows_to_pandas()
This function takes a Darwin Core Archive (DwCA) object as input and saves the rows of the DwCA to a pandas dataframe.


## close_dwca()

Closes a Darwin Core Archive.
```
Parameters
----------
dwca : DarwinCoreArchive
    The Darwin Core Archive to close.

Returns
-------
None
```



## save_csv_rows_to_pandas()
This function takes a CSV file and saves it to a pandas dataframe.
It also unzips the CSV file if it is not already unzipped.
It returns the pandas dataframe.



## print_pandas_column_names()

Prints the column names of a pandas dataframe.
```
Parameters
----------
df : pandas dataframe
    The dataframe whose column names are to be printed.

Returns
-------
None

Examples
--------
>>> print_pandas_column_names(df)
['col1', 'col2', 'col3']
```



## export_gbif_ids()
This function takes a dataframe as an argument and returns a dictionary of
GBIF IDs. The number of IDs returned is determined by the global variable
PERCENT_TO_SCRAPE. The function will skip a number of rows in the dataframe
equal to the number of rows in the dataframe divided by the number of rows
to be scraped. The number of rows to be scraped is equal to the number of
rows in the dataframe multiplied by the global variable PERCENT_TO_SCRAPE.
The function will then scrape the GBIF ID from the dataframe and add it to
the dictionary.



## scrape_occurrence()

Scrapes the GBIF API for a single occurrence.
```
scrape_occurrence(key, data)

Parameters
----------
key : str
    The key of the occurrence in the data dictionary.
data : dict
    The data dictionary containing the occurrence.

Returns
-------
dict
    A dictionary containing the occurrence's ID, image URL, image type, country, genus, species, and recorder.
```



## fetch_data()

This function takes a dictionary of occurrences and fetches data for each occurrence.
    It uses multiprocessing to speed up the process.



## download()
Downloads the image from the given URL and saves it to the output path.
```
Parameters
----------
key : str
    The key of the image in the data dictionary.
data : dict
    The data dictionary.

Returns
-------
bool
    True if the image was downloaded successfully, False otherwise.
str
    The key of the image in the data dictionary.
```



## download_images()

Downloads images from the given data dictionary.
```
Parameters
----------
data : dict
    A dictionary of image data.

Returns
-------
None

Raises
------
None
```



## export_geography_gt()

This function takes a dictionary of the form:
```
{
    "id": {
        "id": "id",
        "country": "country"
    }
}
```
and writes it to a file.




## export_geography_corpus()
This function takes in a dictionary of data and writes the country names to a file.
The file is named geography_corpus.txt and is located in the OUTPUT_PATH directory.
The function prints a message to the console when it is done.




## export_taxon_gt()
This function takes a dictionary of taxon data and writes it to a file.
The file is named taxon_gt.txt and is located in the OUTPUT_PATH directory.
The file is written in the following format:
`    <taxon_id>: <genus> <species>`
The function prints a success message to the console.




## export_taxon_corpus()
This function takes a dictionary of taxon data and writes it to a file.
The file is named "taxon_corpus.txt" and is written to the OUTPUT_PATH.
The file is written in the following format:
```
genus species
genus species
genus species
...
```



## export_collector_gt()
This function takes in a dictionary of data and writes it to a file.
The file is named "collector_gt.txt" and is located in the OUTPUT_PATH
directory.
```
Parameters:
data (dict): A dictionary of data.

Returns:
None
```



## export_collector_corpus()
This function takes a dictionary of data and writes the collector names to a file.
```
:param data: A dictionary of data.
:return: None
```



## print_help_message()

Prints a help message to the console.
```
Parameters
----------
None

Returns
-------
None

Raises
------
SystemExit
    Exits the program.
```



# download.py

ml-herbarium/scraping/web/download.py

## zipdir()
This function takes in a source directory and a destination directory and zips the source directory into the destination directory.
```
Parameters:
    src (str): The source directory.
    dst (str): The destination directory.
Returns:
    None
```



