#read every line in ./GT/handandtyped-ground-truth.txt and write each line with each file named after each file in ./handwriting/
import os

# read every file in ./handwriting/ and create an empty text file named the same as the original file in ./GT/

for file in os.listdir("./handwriting/"):
    if file.endswith(".png"):
        file = file.split(".")[0]
        print(file)
        # then, create a file in ./GT/ with the same name as the original file in ./handwriting/
        # and write the line from ./GT/handandtyped-ground-truth.txt to the new file
        with open("./GT/handandtyped-ground-truth.txt", "r") as f:
            for line in f:
                line = line.split("HB")[1]
                with open("./GT/" + file + ".txt", "w") as g:
                    g.write(line)
    