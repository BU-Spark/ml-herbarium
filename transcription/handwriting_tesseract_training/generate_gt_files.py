import os
import shutil

# Read ground_truth.txt
with open("ground_truth.txt", "r") as f:
    lines = f.readlines()
    # Delete and recreate existing gt directory
    if os.path.exists("gt"):
        shutil.rmtree("gt")
    os.mkdir("gt")
    for line in lines:
        # Save each line to a new file
        split = line.split(" ")
        with open("gt/" + split[0] + ".gt.txt", "w") as f2:
            out = " ".join(split[8:])
            f2.write(out)
        # Copy training image
        shutil.copy("images/" + split[0] + ".png", "gt/" + split[0] + ".png")
