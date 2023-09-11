### Folders and Files Description

#### `src/`
This folder contains all the scripts to generate synthetic images with different text and font styles.

- `staroffice_uno_taxon_geo_dump.py`: This script performs a variety of tasks related to text processing and document manipulation through the LibreOffice UNO API. Initially, the script loads an existing Open Document Text (ODT) file, preparing it for content insertion. It reads two separate corpuses of text, one for taxon names and one for geographic locations, and iterates through them in batches. During this iteration, the script transforms every even-length string to uppercase before inserting it, along with a paragraph break, into the ODT document at the current cursor position. After processing the batches of text up to a defined line limit, the script saves the changes to the ODT file.

- `staroffice_uno_font_manipulation.py`: This script also leverages the LibreOffice UNO API to manipulate text styles in an Open Document Text (ODT) file and then export it as a PDF. First, it connects to a running LibreOffice instance and opens an existing ODT file in a hidden mode, ready for styling. The script defines a set of potential font sizes and styles, then randomly selects and applies these to each paragraph in the document using a cursor that iterates through them. It also applies text decorations like italicization, bolding, or underlining. Finally, the script exports the modified document to a PDF file, saving it at a specified location.

#### `files/`
This folder contains all the ODT and PDF files generated when the scripts are run

#### `images/`
This folder contains all the images generated from the PDF files (1 image per page in the PDF file).