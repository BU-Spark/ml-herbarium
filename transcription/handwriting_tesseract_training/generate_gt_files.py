import os
import shutil

# Read ground_truth.txt
with open("ground_truth.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        # Delete and recreate existing gt directory
        if os.path.exists("gt"):
            shutil.rmtree("gt")
        os.mkdir("gt")
        # Save each line to a new file
        split = line.split(" ")
        with open("gt/" + split[0] + ".gt.txt", "w") as f2:
            f2.write(line[8:])
