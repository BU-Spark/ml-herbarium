import uno
import pathlib
from com.sun.star.beans import PropertyValue
from com.sun.star.awt.FontSlant import (NONE, ITALIC,)
from com.sun.star.awt.FontWeight import (NORMAL, BOLD,)
from com.sun.star.awt.FontUnderline import (NONE, SINGLE,)
import os
import random

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

def get_font_style():
    font_sizes = [20, 22.5, 25, 27.5, 30, 32.5]
    font_styles = ["Courier", "Times New Roman", "Calibri", "Arial", "American Typewriter", "Helvetica", "Futura", "Bodoni 72", "PT Serif", "PT Sans"]

    idx_size = random.randint(0, len(font_sizes)-1)
    idx_font = random.randint(0, len(font_styles)-1)

    return font_sizes[idx_size], font_styles[idx_font]

def apply_font_style(textCursor):
    style = get_font_style()
    # Change the font and styles of all the text the cursor spans
    textCursor.setPropertyValue("CharFontName", style[1])
    textCursor.setPropertyValue("CharHeight", style[0])

    textCursor.setPropertyValue("CharPosture", NONE)
    textCursor.setPropertyValue("CharWeight", NORMAL)
    textCursor.setPropertyValue("CharUnderline", NONE)

    if(random.randint(0, 4096)%3 == 0):
        textCursor.setPropertyValue("CharPosture", ITALIC) # Set italic font style
    if(random.randint(0, 4096)%3 == 1):
        textCursor.setPropertyValue("CharWeight", BOLD)  # Set bold font style
    if(random.randint(0, 4096)%3 == 2):
        textCursor.setPropertyValue("CharUnderline", SINGLE)  # Underline text

def export_pdf():
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

    # Create a cursor that spans one paragraph at a time
    text = doc.Text
    textCursor = text.createTextCursor()

    textCursor.gotoEndOfParagraph(True)
    apply_font_style(textCursor)

    # print(textCursor.String)

    print("Aplying font styles...")
    while(textCursor.gotoNextParagraph(False)):
        textCursor.gotoEndOfParagraph(True)
        apply_font_style(textCursor)

    filterProps = [
        make_prop("IsSkipEmptyPages", False),
    ]
    filterProps = uno.Any("[]com.sun.star.beans.PropertyValue", filterProps)

    saveArgs = [
        make_prop("FilterName", "writer_pdf_Export"),
        make_prop("FilterData", filterProps),
    ]

    pdfName = file_url("trocr/evaluation-dataset/handwritten-typed-text-classification/ml-herbaria-synthetic-text-dataset/files/file.pdf")

    print("Saving PDF... Finishing up", pdfName)

    # Save the document
    doc.storeToURL(pdfName, saveArgs)

    # Close the document
    doc.dispose()

export_pdf()