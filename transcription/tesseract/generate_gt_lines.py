import os
import shutil
import tarfile

# Create a directory to store the extracted files
if os.path.exists("tmp"):
        shutil.rmtree("tmp")
os.mkdir("tmp")
os.mkdir("tmp/lines")
os.mkdir("tmp/gt")

# Extract the files
lines = tarfile.open("lines.tgz")
lines.extractall("tmp/lines")
lines.close()
gt = tarfile.open("ascii.tgz")
gt.extractall("tmp/gt")
gt.close()

# Read the ground truth file
with open("tmp/gt/lines.txt", "r") as f:
    lines = f.readlines()
    # Delete and recreate existing gt directory
    if os.path.exists("gt"):
        shutil.rmtree("gt")
    os.mkdir("gt")
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
        with open("gt/" + split[0] + ".gt.txt", "w") as f2:
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
        shutil.move("tmp/lines/" + outerfolder + "/" + innerfolder + "/" + split[0] + ".png", "gt/" + split[0] + ".png")
# Remove the temporary directory
shutil.rmtree("tmp")