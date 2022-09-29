import struct
from array import array
import os
# Install this library through the pip install pypng command
import png
import os


trainimg = '/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/MNIST/train-images-idx3-ubyte'
trainlabel ='/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/MNIST/train-labels-idx1-ubyte'
testimg ='/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/MNIST/t10k-images-idx3-ubyte'
testlabel ='/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/MNIST/t10k-labels-idx1-ubyte'
MNIST_dir = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/MNIST"
trainfolder = MNIST_dir + '/train/'
testfolder = MNIST_dir + '/test/'
if not os.path.exists(trainfolder): os.makedirs(trainfolder)
if not os.path.exists(testfolder): os.makedirs(testfolder)

# open (file path, read-write format), used to open a file and return a file object
# rb means to open the file in binary read mode
trimg = open(trainimg,'rb')
teimg = open(testimg,'rb')
trlab = open(trainlabel,'rb')
telab = open(testlabel,'rb')

# The usage of struct is not detailed here
struct.unpack(">IIII", trimg.read(16))
struct.unpack(">IIII", teimg.read(16))
struct.unpack(">II", trlab.read(8))
struct.unpack(">II", telab.read(8))

# The array module is an efficient array storage type implemented in Python
# All array members must be of the same type, which is specified when the array is created
# B means unsigned byte type, b means signed byte type
trimage = array("B", trimg.read())
teimage = array("B", teimg.read())
trlabel = array("b", trlab.read())
telabel = array("b", telab.read())

# The close method is used to close an open file, and the file cannot be read or written after it is closed
trimg.close()
teimg.close()
trlab.close()
telab.close()

# Define 10 subfolders for each of the training set and test set, which are used to store all the numbers from 0 to 9. The folder names are 0-9 respectively
trainfolders = [os.path.join(trainfolder, str(i)) for i in range(10)]
testfolders = [os.path.join(testfolder, str(i)) for i in range(10)]
for dir in trainfolders:
    if not os.path.exists(dir):
        os.makedirs(dir)
for dir in testfolders:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Start saving training image data
for (i, label) in enumerate(trlabel):
    filename = os.path.join(trainfolders[label], str(i) + ".png")
    print("writing "+ filename)
    with open(filename, "wb") as img:
        image = png.Writer(28, 28, greyscale=True)
        data = [trimage[(i*28*28 + j*28): (i*28*28 + (j+1)*28)] for j in range(28)]
        image.write(img, data)

# Start saving test image data
for (i, label) in enumerate(telabel):
    filename = os.path.join(testfolders[label], str(i) + ".png")
    print("writing "+ filename)
    with open(filename, "wb") as img:
        image = png.Writer(28, 28, greyscale=True)
        data = [teimage[(i*28*28 + j*28): (i*28*28 + (j+1)*28)] for j in range(28)]
        image.write(img, data)