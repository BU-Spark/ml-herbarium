from PIL import Image
import os

images = os.listdir("/usr4/ugrad/en/ml-herbarium/transcription/handwriting_tesseract_training/images")
for image_path in images:
    img = Image.open("/usr4/ugrad/en/ml-herbarium/transcription/handwriting_tesseract_training/images/" + image_path)
    img.save("/usr4/ugrad/en/ml-herbarium/transcription/handwriting_tesseract_training/images/" + image_path.split(".")[0] + ".tiff", 'TIFF')