import os
import shutil
import tarfile

# NOTE: Training data can be found at: /projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/tesseract-training/training

# Create a directories to store the extracted files
if os.path.exists("tmp"):
        shutil.rmtree("tmp")
os.mkdir("tmp")
os.mkdir("tmp/gt")
if os.path.exists("gt"):
        shutil.rmtree("gt")
os.mkdir("gt")

# Extract gt files
gt = tarfile.open("ascii.tgz")
gt.extractall("tmp/gt")
gt.close()

# Extract line image files
if os.path.exists("lines.tgz"):
    os.mkdir("tmp/lines")
    lines = tarfile.open("lines.tgz")
    lines.extractall("tmp/lines")
    lines.close()
    # Read the ground truth file
    with open("tmp/gt/lines.txt", "r") as f:
        lines = f.readlines()
        # Delete and recreate existing gt directory
        if os.path.exists("gt/lines"):
            shutil.rmtree("gt/lines")
        os.mkdir("gt/lines")
        for line in lines:
            # Save each line to a new file to be read by tesstrain
            # Skip comments
            if line[0] == "#":
                continue
            split = line.split(" ")
            # Skip lines with segmentation errors
            if split[1] == "err":
                continue
            # Output the ground truth file
            with open("gt/lines/" + split[0] + ".gt.txt", "w") as f2:
                # Get the ground truth line
                out = " ".join(split[8:])
                # Replace "|" with " "
                out = out.replace("|", " ")
                # Remove trailing spaces before punctuation
                out = out.replace(" .", ".")
                out = out.replace(" ,", ",")
                out = out.replace(" !", "!")
                out = out.replace(" ?", "?")
                out = out.replace(" :", ":")
                out = out.replace(" ;", ";")
                out = out.replace(" (", "(")
                out = out.replace(" )", ")")
                out = out.replace(" [", "[")
                out = out.replace(" ]", "]")
                out = out.replace(" {", "{")
                out = out.replace(" }", "}")
                out = out.replace(" \"", "\"")
                out = out.replace(" \'", "\'")
                f2.write(out)
            # Copy training image
            outerfolder = split[0].split("-")[0]
            innerfolder = "-".join(split[0].split("-")[:2])
            shutil.move("tmp/lines/" + outerfolder + "/" + innerfolder + "/" + split[0] + ".png", "gt/lines/" + split[0] + ".png")

if os.path.exists("sentences.tgz"):
    os.mkdir("tmp/sentences")
    sentences = tarfile.open("sentences.tgz")
    sentences.extractall("tmp/sentences")
    sentences.close()
    # Read the ground truth file
    with open("tmp/gt/sentences.txt", "r") as f:
        sentences = f.readlines()
        # Delete and recreate existing gt directory
        if os.path.exists("gt/sentences"):
            shutil.rmtree("gt/sentences")
        os.mkdir("gt/sentences")
        for sentence in sentences:
            # Save each line to a new file to be read by tesstrain
            # Skip comments
            if sentence[0] == "#":
                continue
            split = sentence.split(" ")
            # Skip lines with segmentation errors
            if split[2] == "err":
                continue
            # Output the ground truth file
            with open("gt/sentences/" + split[0] + ".gt.txt", "w") as f2:
                # Get the ground truth line
                out = " ".join(split[9:])
                # Replace "|" with " "
                out = out.replace("|", " ")
                # Remove trailing spaces before punctuation
                out = out.replace(" .", ".")
                out = out.replace(" ,", ",")
                out = out.replace(" !", "!")
                out = out.replace(" ?", "?")
                out = out.replace(" :", ":")
                out = out.replace(" ;", ";")
                out = out.replace(" (", "(")
                out = out.replace(" )", ")")
                out = out.replace(" [", "[")
                out = out.replace(" ]", "]")
                out = out.replace(" {", "{")
                out = out.replace(" }", "}")
                out = out.replace(" \"", "\"")
                out = out.replace(" \'", "\'")
                f2.write(out)
            # Copy training image
            outerfolder = split[0].split("-")[0]
            innerfolder = "-".join(split[0].split("-")[:2])
            shutil.move("tmp/sentences/" + outerfolder + "/" + innerfolder + "/" + split[0] + ".png", "gt/sentences/" + split[0] + ".png")

if os.path.exists("words.tgz"):
    os.mkdir("tmp/words")
    words = tarfile.open("words.tgz")
    words.extractall("tmp/words")
    words.close()
    # Read the ground truth file
    with open("tmp/gt/words.txt", "r") as f:
        words = f.readlines()
        # Delete and recreate existing gt directory
        if os.path.exists("gt/words"):
            shutil.rmtree("gt/words")
        os.mkdir("gt/words")
        for word in words:
            # Save each line to a new file to be read by tesstrain
            # Skip comments
            if word[0] == "#":
                continue
            split = word.split(" ")
            # Skip lines with segmentation errors
            if split[1] == "err":
                continue
            # Output the ground truth file
            with open("gt/words/" + split[0] + ".gt.txt", "w") as f2:
                # Get the ground truth line
                out = " ".join(split[8:])
                # Replace "|" with " "
                out = out.replace("|", " ")
                # Remove trailing spaces before punctuation
                out = out.replace(" .", ".")
                out = out.replace(" ,", ",")
                out = out.replace(" !", "!")
                out = out.replace(" ?", "?")
                out = out.replace(" :", ":")
                out = out.replace(" ;", ";")
                out = out.replace(" (", "(")
                out = out.replace(" )", ")")
                out = out.replace(" [", "[")
                out = out.replace(" ]", "]")
                out = out.replace(" {", "{")
                out = out.replace(" }", "}")
                out = out.replace(" \"", "\"")
                out = out.replace(" \'", "\'")
                f2.write(out)
            # Copy training image
            outerfolder = split[0].split("-")[0]
            innerfolder = "-".join(split[0].split("-")[:2])
            shutil.move("tmp/words/" + outerfolder + "/" + innerfolder + "/" + split[0] + ".png", "gt/words/" + split[0] + ".png")

# Remove the temporary directory
shutil.rmtree("tmp")