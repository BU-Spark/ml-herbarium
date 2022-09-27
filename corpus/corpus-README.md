# ML-Herbarium Documentation

# corpus_geography.py

ml-herbarium/corpus/corpus_geography/corpus_geography.py

## download_dataset()

Downloads a dataset from a given URL and saves it to the output directory.
```
Parameters
----------
dataset_url : str
    The URL of the dataset to download.
zip_file : str
    The name of the zip file to save the dataset to.

Returns
-------
None

Raises
------
None

```



## process_geo()
This function takes in a zip file, a txt file, and a csv file.
It extracts the zip file, converts the txt file to csv format,
and takes the higher geography column from the csv file.
It then returns a list of lists of cities.



## write_geo()

This function writes the countries and states of the cities to a text file.
The text file is named corpus_geography.txt and is located in the output_dir.
```
    The text file is formatted as follows:
    country_name, state_name, city_name
```



