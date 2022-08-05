import os

# Data available at: https://cvit.iiit.ac.in/research/projects/cvit-projects/matchdocimgs

# Read the ground truth file
with open("/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/training/CVIT/groundtruth/10K/ALLnames_10K.csv", "r") as n:
    with open("/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/training/CVIT/groundtruth/10K/ALLtext_10K.csv", "r") as t:
        names = n.readlines()
        text = t.readlines()
        for line in names:
            # Save each line to a new file to be read by tesstrain
            # Skip comments
            if line == 'names\n':
                continue
            else:
                dirname = int(line.split("/")[0])
                fname = line.split("/")[1].strip()
                noext = fname.split(".")[0]
                gtpath = "/scratch/ml-herbarium/CVIT_data/10K/" + str(dirname)+"-"+noext+".gt.txt"
                imgpath = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/training/CVIT/Images_90K_Normalized/" + str(dirname)+"/"+noext+".png"
                newpath = "/scratch/ml-herbarium/CVIT_data/10K/" + str(dirname)+"-"+noext+".png"
            with open(gtpath, 'w+') as f:
                f.write(text[dirname].strip())
            os.copyfile(imgpath, newpath)
