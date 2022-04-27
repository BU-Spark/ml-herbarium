from PIL import Image
import os

images = os.listdir("images")
for image_path in images:
    img = Image.open("images/" + image_path)
    img.save("images/" + image_path.split(".")[0] + ".tif", 'TIFF')
    os.remove("images/" + image_path)