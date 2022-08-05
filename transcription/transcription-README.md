# ML-Herbarium Documentation

# transcribe_labels.py

ml-herbarium/transcription/transcribe_labels.py

## get_gt()

    Get ground truth from the file.
```
Args:
    fname: file name
    org_img_dir: directory of the original image
Returns:
    ground_truth: a dictionary of ground truth
```



## get_corpus_taxon()

This function takes in the directory of the original images and returns the corpus of taxon names. The corpus is a list of taxon names.
```
The corpus is split into three lists:
    1. corpus_full: a list of full taxon names (e.g. "Acer rubrum")
    2. corpus_genus: a list of genus names (e.g. "Acer")
    3. corpus_species: a list of species names (e.g. "rubrum")
```



## get_corpus()

This function takes in a filename and an original image directory, and returns a corpus of words.
The corpus is a list of all the words in the file, with duplicates removed.
The corpus is also lowercased.
If the words parameter is set to True, the corpus is split on newlines and spaces.
If the words parameter is set to False, the corpus is split on newlines only.




## getangle()
This function takes in an image and returns the angle of skew.
It does this by first blurring the image, then thresholding it.
It then finds the contours of the image, and finds the largest contour.
It then finds the minimum area rectangle of the largest contour, and finds the angle of that rectangle.
It then returns the angle.




## rotateimg()


This function takes in an image and returns the image rotated by the angle of skew.
The angle of skew is calculated by the function getangle.
The image is rotated using the rotation matrix.
The rotation matrix is calculated using the center of the image and the angle of skew.
The image is rotated using the rotation matrix and the image is returned.




## get_img()


This function takes in the path of an image and returns a dictionary with the image name as the key and the image as the value.
If the image is not a valid image, it returns an empty dictionary and the image path.




## get_imgs()

Gets the original images from the URLs in the input dictionary.
```
The input dictionary should be of the form:
    {
        "img_name": "img_url",
        ...
    }
```
The output dictionary will be of the same form, but with the URLs replaced by the images.
The images will be preprocessed by the preprocess_img function.
The function will return a list of the URLs that failed to download.



## has_y_overlap()

Checks if two rectangles overlap in the y-axis.
```
    :param y1: The y-coordinate of the first rectangle.
    :param y2: The y-coordinate of the second rectangle.
    :param h1: The height of the first rectangle.
    :param h2: The height of the second rectangle.
    :return: True if the rectangles overlap, False otherwise.
```



## find_idx_nearby_text()

Finds the index of the text in the same line as the text at the given index.
```
Parameters
----------
ocr_results : dict
    The dictionary of OCR results.
img_name : str
    The name of the image.
result_idx : int
    The index of the text in the OCR results.

Returns
-------
int
    The index of the text in the same line as the text at the given index.
    If there is no text in the same line, returns None.
```



## words_to_lines()

This function takes in the results of the OCR and returns a dictionary of lines.
The keys of the dictionary are the image names.
The values of the dictionary are lists of lists.
Each list in the list represents a line.
Each list contains the indices of the words that are in that line.
The indices are the indices of the words in the OCR results.
```
    The OCR results are a dictionary with the following keys:
        "text" - a list of the words
        "left" - a list of the x coordinates of the words
        "top" - a list of the y coordinates of the words
        "width" - a list of the widths of the words
        "height" - a list of the heights of the words
```
The x_margin is the minimum distance between two words for them to be considered in the same line.



## get_syn_dict()


This function generates a synonym dictionary.
It is a dictionary of dictionaries.
The first level of keys are the genus names.
The second level of keys are the species names.
The values are lists of synonyms.



## import_process_data()

This function imports the data from the given directory.
It returns the images, the geography corpus, the taxon corpus, the taxon ground truth, the geography ground truth, and the taxon corpus.
```
Parameters:
org_img_dir (str): The directory of the original images.
num_threads (int): The number of threads to use for the image import.

Returns:
imgs (list): A list of the images.
geography_corpus_words (list): A list of the geography corpus.
geography_corpus_full (list): A list of the geography corpus.
taxon_gt_txt (list): A list of the taxon ground truth.
geography_gt_txt (list): A list of the geography ground truth.
taxon_corpus_full (list): A list of the taxon corpus.
```



## run_ocr()

Runs OCR on the given image.
```
Parameters
----------
img_name : str
    The name of the image to run OCR on.
imgs : dict
    A dictionary of images.
config : str
    The configuration string to use for OCR.

Returns
-------
dict
    A dictionary of OCR results.
```



## ocr()

Runs OCR on a list of images using Tesseract. Uses Python multiprocessing to map each image to a different core, each running an instance of Tesseract within the `run_ocr()` function.
```

Parameters
----------
imgs : list
    A list of image paths.
num_threads : int
    The number of threads to use.

Returns
-------
ocr_results : dict
    A dictionary of OCR results.
```



## generate_debug_output()


This function takes in the image name, the ocr results, the images, the original image directory, and the output directory.
It then creates a debug image and an original image with the bounding boxes and text on them.
It then saves the images to the output directory.




## ocr_debug()

Generates debug outputs for the OCR results.
```
Parameters
----------
ocr_results : dict
    A dictionary containing the OCR results.
output_dir : str
    The output directory.
imgs : list
    A list of the images.
org_img_dir : str
    The original image directory.

Returns
-------
None
```



## run_match_taxon()

This function takes in the results of the OCR and the possible genus/species dictionaries and returns the best match for the taxon.
```
Parameters
----------
img : tuple
    A tuple containing the image name and the results of the OCR.
corpus_genus : dict
    A dictionary containing the possible genera.
corpus_species : dict
    A dictionary containing the possible species.
output_dir : str
    The directory where the output files will be saved.
debug : bool
    Whether or not to save the debug files.

Returns
-------
dict
    A dictionary containing the image name and the best match for the taxon.
```



## match_genus()

This function takes a tuple of OCR text and confidence, and a dictionary of
genus names and their frequencies. It returns a tuple of the similarity
between the OCR text and the matched genus, the matched genus, the OCR
confidence, and the OCR text.
```
    Parameters
    ----------
    n : tuple
        A tuple of OCR text and confidence.
    corpus_genus : dict
        A dictionary of genus names and their frequencies.

    Returns
    -------
    tuple
        A tuple of the similarity between the OCR text and the matched genus,
        the matched genus, the OCR confidence, and the OCR text.
```



## match_species()

This function takes a tuple of OCR confidence and OCR text, and a dictionary of species names.
It returns a tuple of the OCR confidence, the matched species name, the OCR confidence, and the OCR text.
If the OCR text is a single character, it returns None.
If the OCR text is longer than one character, it uses the fuzzywuzzy library to find the best match.
If the best match is below a threshold of 91, it returns None.




## match_taxon()

    Match words to taxon corpus.
```
Args:
    ocr_results: dict of ocr results
    taxon_corpus_full: dict of taxon corpus
    corpus_genus: key is genus and value is a list of possible species 
    corpus_species: key is species and value is a list of possible genus
    output_dir: output directory
    debug: debug mode
Returns:
    final: dict of matched taxon
```



## determine_match()


This function takes in the ground truth dictionary, the final dictionary, the name of the file, and the output directory.
It then writes the results to a file in the output directory.
It also prints the accuracy, no match, and wrong percentages.
If the ground truth dictionary is None, it just writes the final dictionary to the file.




## main()

This function is the main function of the program. It takes in the directory of the original images,
the directory of the output, and the number of threads to use. It then imports and processes the data,
runs the OCR, and then matches the OCR results to the ground truth.




# test_preprocessing.py

ml-herbarium/transcription/test_preprocessing.py

## getangle()


This function takes in an image and returns the angle of skew.
It does this by first blurring the image, then thresholding it.
It then finds the contours of the image, and finds the largest contour.
It then finds the minimum area rectangle of the largest contour, and finds the angle of that rectangle.
It then returns the angle.




## rotateimg()


This function takes in an image and returns the image rotated by the angle of skew.
The angle of skew is calculated by the function getangle.
The image is rotated using the rotation matrix.
The rotation matrix is calculated using the center of the image and the angle of skew.
The image is rotated using the rotation matrix and the image is returned.




