### Folders and Files Description

#### `src/`
This folder contains all the scripts to generate synthetic images with different text and font styles.

- `staroffice_uno_taxon_geo_dump.py`: This script performs a variety of tasks related to text processing and document manipulation through the LibreOffice UNO API. Initially, the script loads an existing Open Document Text (ODT) file, preparing it for content insertion. It reads two separate corpuses of text, one for taxon names and one for geographic locations, and iterates through them in batches. During this iteration, the script transforms every even-length string to uppercase before inserting it, along with a paragraph break, into the ODT document at the current cursor position. After processing the batches of text up to a defined line limit, the script saves the changes to the ODT file.

- `staroffice_uno_font_manipulation.py`: This script also leverages the LibreOffice UNO API to manipulate text styles in an Open Document Text (ODT) file and then export it as a PDF. First, it connects to a running LibreOffice instance and opens an existing ODT file in a hidden mode, ready for styling. The script defines a set of potential font sizes and styles, then randomly selects and applies these to each paragraph in the document using a cursor that iterates through them. It also applies text decorations like italicization, bolding, or underlining. Finally, the script exports the modified document to a PDF file, saving it at a specified location.

#### `files/`
This folder contains all the ODT and PDF files generated when the scripts are run

#### `images/`
This folder contains all the images generated from the PDF files (1 image per page in the PDF file).

### Setup Instructions

1. Install LibreOffice (https://pypi.org/project/pyoo/ -> check instructions here) for steps 1, 2 and 3.
2. Install PyOO (https://pypi.org/project/pyoo/) and follow the setup instructions from there.

### Usage

**Prerequisites**: Ensure LibreOffice is installed and the UNO API is accessible.

#### `staroffice_uno_taxon_geo_dump.py`

> Place your `corpus_taxon.txt` and `corpus_geography.txt` files in the `corpus/corpus_taxon/` and `corpus/corpus_geography/` directories, respectively. TXT files containing any text can be used to generate synthetic images.
  
1. **Command to Run the Script**: Open the terminal and navigate to the directory where `staroffice_uno_taxon_geo_dump.py` is located. Run the following command:

    ```
    /Applications/LibreOffice.app/Contents/Resources/python staroffice_uno_taxon_geo_dump.py
    ```
    
2. **Output**: Upon successful execution, the script will save the populated ODT file in the specified path and print out status messages such as "Loading document..." and "Printing done... Finishing up".

---

#### `staroffice_uno_font_manipulation.py`

> Make sure you have an existing ODT file that you wish to manipulate.

1. **Command to Run the Script**: Open the terminal and navigate to the directory where `staroffice_uno_font_manipulation.py` is located. Run the following command:

    ```
    /Applications/LibreOffice.app/Contents/Resources/python staroffice_uno_font_manipulation.py
    ```

2. **Output**: Once all the font styles have been applied, the script will save the document as a PDF in the specified directory and print a "Saving PDF... Finishing up" message to indicate the process has completed.


### Deployment Instructions

1. Run LibreOffice server with the command (on MacOS). For Linux, the command can be found on the PyOO page.
```
   /Applications/LibreOffice.app/Contents/MacOS/soffice \
  --accept='socket,host=localhost,port=2002;urp;StarOffice.Service' \
  --headless
```
2. Run the python scripts from the repository (in the `../ml-herbaria-synthetic-text-dataset` directory). First execute `staroffice_uno_taxon_geo_dump.py`. Once you have the ODT file, execute the `staroffice_uno_font_manipulation.py` file. Get the output PDF files.
3. With imagemagick, convert the PDF files to JPEG images. (you can use other applications to convert the PDFs to JPEGs as well)
   (https://imagemagick.org/script/download.php, https://imagemagick.org/archive/python/). Use the following command. Make sure to run the command with the right arguments.
```
convert -density 300 \ 
    -depth 16 \
    -background white \ 
    -alpha remove \
    <path/to/>file.pdf \
    -resize 50% \
    <path/to/>file.jpg
```

4. Each `.jpg` image corresponds to a page in the PDF file.