import uno
import pathlib
from com.sun.star.beans import PropertyValue
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
    font_sizes = [20, 22.5, 25, 27.5, 30]
    font_styles = ["Courier", "Times New Roman", "Calibri", "Arial", "American Typewriter"]

    idx_size = random.randint(0, len(font_sizes)-1)
    idx_font = random.randint(0, len(font_styles)-1)

    return font_sizes[idx_size], font_styles[idx_font]

def apply_font_style(textCursor):
    style = get_font_style()
    # Change the font of all the text the cursor spans
    textCursor.setPropertyValue("CharFontName", style[1])
    textCursor.setPropertyValue("CharHeight", style[0])

def export_pdf():
    desktop = create_instance()

    loadArgs = [
        make_prop("UpdateDocMode", 1),
        make_prop("Hidden", True)
    ]

    url = file_url("files/file.odt")

    print(url)

    print("Loading...")

    doc = desktop.loadComponentFromURL(url, "_blank", 0, loadArgs)

    print("Loaded...")

    # Create a cursor that spans the entire document
    text = doc.Text
    textCursor = text.createTextCursor()

    # tab delimiter
    # different configurations can be lists and we choose randomly from each of them - font size, font style, background color

    textCursor.gotoEndOfParagraph(True)
    apply_font_style(textCursor)

    # print(textCursor.String)

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

    pdfName = file_url("files/file.pdf")

    print("Saving PDF", pdfName)

    doc.storeToURL(pdfName, saveArgs)

    doc.dispose()

export_pdf()