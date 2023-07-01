import uno
import pathlib
from com.sun.star.beans import PropertyValue
from com.sun.star.text.ControlCharacter import PARAGRAPH_BREAK
import os
from itertools import zip_longest

def create_instance():
    localContext = uno.getComponentContext()
    resolver = localContext.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", localContext )
    ctx = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext" )
    smgr = ctx.ServiceManager
    desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop",ctx)

    return desktop

def make_prop(name, val):
    return PropertyValue(Name=name, Value=val)

def file_url(path):
    path = os.path.abspath(path)
    return pathlib.Path(path).as_uri()

def corpus_generator(filename, batch_size, start_line=0):

    with open(filename, 'r') as file:
        curr_line = 0

        while curr_line < start_line:
            line = file.readline()
            if not line:
                break
            
            curr_line += 1

        while True:
            batch = []
            for _ in range(batch_size):
                line = file.readline()
                if line:
                    batch.append(line.strip())
                else:
                    break
            if not batch:
                break
            yield batch

def dump_text_to_odt_file():
    desktop = create_instance()

    loadArgs = [
        make_prop("UpdateDocMode", 1),
        make_prop("Hidden", True)
    ]

    url = file_url("trocr/evaluation-dataset/handwritten-typed-text-classification/ml-herbaria-synthetic-text-dataset/files/file.odt")

    print(url)

    print("Loading document...")

    doc = desktop.loadComponentFromURL(url, "_blank", 0, loadArgs)

    print("Loaded document...")

    # Create a cursor that spans the entire document
    text = doc.Text
    text.setString("")
    
    # Save the file
    doc.store()

    # print(text.String)
    cursor = text.createTextCursor()

    # text.insertString( cursor, "The first line in the newly created text document.\n", 0 )
    # text.insertString( cursor, "Now we are in the second line\n" , 0 )
    start_line = 2000
    batch_size = 16

    line_limit = 5000 + start_line

    taxon_to_insert = corpus_generator("corpus/corpus_taxon/corpus_taxon.txt", batch_size, start_line)
    location_to_insert = corpus_generator("corpus/corpus_geography/corpus_geography.txt", batch_size, start_line)

    # break_flag = False

    print("Fetching and printing text...")
    for batch in zip_longest(taxon_to_insert, location_to_insert, fillvalue=""):
        if(start_line > line_limit):
            break
        
        for elements in batch:
            if elements is None:
                break
            for string in elements:
                if string == "":
                    continue
                if len(string)%2 == 0:
                    string = string.upper()
                text.insertString(cursor, string+"\n", 0)
                text.insertControlCharacter(cursor.End, PARAGRAPH_BREAK, False)

        start_line += batch_size

    cursor.gotoEndOfParagraph(True)

    # Save the document
    doc.store()

    print("Printing done... Finishing up")

    # Close the document
    doc.dispose()

dump_text_to_odt_file()