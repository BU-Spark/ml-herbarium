# ML-Herbarium Documentation

# generate_syn.py

ml-herbarium/transcription/synonym/generate_syn.py

## download_dataset()

Downloads the dataset from the given url and saves it to the given zip file.
```
Parameters
----------
dataset_url : str
    The url of the dataset to be downloaded.
zip_file : str
    The name of the zip file to be saved.
run : bool, optional
    If True, the function will run. The default is False.

Returns
-------
None.
```



## preprocess_data()

This function is used to preprocess the data.
It will extract the files from the .zip file, create a dictionary called dic for all of the species in the dataset,
and create a dictionary called syn for all pairs of synonyms from the dataset.
The function will return the syn dictionary.




## process_word()


This function takes a string as input and returns a string as output.
The input string is assumed to be a word or a phrase.
The output string is the same as the input string, except that
if the input string is a phrase, then the output string is the first
word of the phrase.



## syn_pure()


This function takes a dictionary of synonyms and returns a dictionary of synonyms with only one word.
The input dictionary is a dictionary of synonyms, where the key is the word and the value is the synonym.
The output dictionary is a dictionary of synonyms, where the key is the word and the value is the synonym.
The output dictionary only contains synonyms with one word.
The output dictionary is saved as a pickle file.




## main()

This function downloads the dataset from the given url and extracts the zip file.
```
Parameters
----------
dataset_url : str
    The url of the dataset.
zip_file : str
    The name of the zip file.
run : bool
    If True, the function will run.

Returns
-------
None

```



