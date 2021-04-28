import os
import zipfile
import json 
import numpy as np
import urllib.request


def zipdir(src, dst):
    ### Create a Destination Directory 
    os.chdir(dst)
    f1 = open(os.path.join(src,'bruh.json'), 'r')
    data = f1.read()
    data = json.loads(data)
    output = open(os.path.join(src, 'words.txt'), 'w')
    count = 297
    keys = []
    unchaged = " ok 154 408 768 27 51 HB "
    for key in data.keys():
        keys.append(key)
    for i in range(len(data)):
        name = data[keys[i]].get("name")
        name = name.split(" ")
        if len(name) > 2: 
            name = name[:2]
        name = " ".join(name)
        images_urls = data[keys[i]].get("images")
        for url in images_urls: 
            print(count)
            idx1 = str(int(count / 100))
            idx2 = str(int(count % 100))
            if len(idx1) == 1: 
                idx1 = "0" + idx1
            if len(idx2) == 1: 
                idx2 = "0" + idx2 
            if len(idx1) > 2: 
                break
            label = "d01-000u-" + idx1 + "-" + idx2
            line = label + unchaged + name + "\n"
            output.writelines(line)
            urllib.request.urlretrieve(url, os.path.join(dst, label))
            count = count + 1
    #output.close()

path = '~/Desktop/'
dst = '~/Desktop/Images'
zipdir(path, dst)
