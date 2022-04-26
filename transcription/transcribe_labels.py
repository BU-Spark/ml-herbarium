from functools import partial
import os
import re
import shutil
import sys
from PIL import Image
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import multiprocessing as mp
import warnings

from tqdm import tqdm

### --------------------------------- Helper Functions --------------------------------- ###
def get_gt(fname, org_img_dir):
    gt_dir = org_img_dir + fname + "_gt.txt"
    if os.path.exists(gt_dir):
        gt = open(gt_dir).read().split("\n")
        gt = [i.lower() for i in gt if i]
        ground_truth = {s.split(": ")[0]: s.split(": ")[1] for s in gt}
        return ground_truth

    return None

def get_corpus(fname, org_img_dir):
    corpus_dir = org_img_dir + fname + "_corpus.txt"
    corpus = re.split("\n| ", open(corpus_dir).read())
    corpus = [s.lower() for s in corpus]

    corpus_fullname = [(corpus[i]+" "+corpus[i+1]) for i in range(len(corpus)-1) if (i%2==0)]
    corpus_fullname = list(set(corpus_fullname))

    corpus_full = corpus_fullname + list(set(corpus))

    return corpus_full

def get_img(image_path):
    try:
        img = np.array(Image.open(image_path))
    except:
        return {}, image_path
    return {image_path.split("/")[-1]: img}, None

def get_imgs(imgs, num_threads):
    imgs_out = {}
    failures = []

    warnings.filterwarnings("error")
    print("Getting original images...")
    print("Starting multiprocessing...")
    pool = mp.Pool(num_threads)
    for item, error in tqdm(pool.imap(get_img, imgs), total=len(imgs)):
        imgs_out.update(item)
        if error:
            failures.append(error)
    pool.close()
    pool.join()
    for f in failures:
        print("Failed to get image: "+f)
    print("Original images obtained.\n")
    warnings.filterwarnings("default")

    return imgs_out

### --------------------------------- Import data & process --------------------------------- ###
def import_process_data(org_img_dir, num_threads):
    imgs = sorted(os.listdir(org_img_dir))
    imgs = [org_img_dir + img for img in imgs if img[-4:] == ".jpg"]
    imgs = get_imgs(imgs, num_threads)
    taxon_corpus = get_corpus("taxon", org_img_dir)
    geography_corpus = get_corpus("geography", org_img_dir)
    taxon_gt_txt = get_gt("taxon", org_img_dir)
    geography_gt_txt = get_gt("geography", org_img_dir)
    
    return imgs, geography_corpus, taxon_gt_txt, geography_gt_txt, taxon_corpus


### --------------------------------- Optical character recognition --------------------------------- ###
def run_ocr(img_name, imgs):
    results = pytesseract.image_to_data(imgs[img_name], output_type=Output.DICT)
    return {img_name: results}

def ocr(imgs, num_threads):
    ocr_results = {}

    tesspath = os.path.expanduser("~/.local/bin/tesseract")
    pytesseract.pytesseract.tesseract_cmd=tesspath
    print("Running OCR on images using Tesseract "+str(pytesseract.pytesseract.get_tesseract_version())+" ...")
    print("Starting multiprocessing...")
    pool = mp.Pool(num_threads)
    func = partial(run_ocr, imgs=imgs)
    for item in tqdm(pool.imap(func, imgs), total=len(imgs)):
        ocr_results.update(item)
    pool.close()
    pool.join()
    print("OCR complete.\n")

    return ocr_results

### --------------------------------- Match words to corpus --------------------------------- ###
def match_words_to_corpus(ocr_results, name, corpus, org_img_dir, output_dir, debug=False):
    from difflib import get_close_matches
    cnt = 0
    final = {}
    print("Matching words to "+ name +" corpus...")
    if debug:
            if not os.path.exists(output_dir+"/debug/"):
                os.makedirs(output_dir+"/debug/")
            f = open(output_dir+"/debug/"+name+img_name+".txt", "w")
    for img_name,results in tqdm(ocr_results.items(), total=len(ocr_results)):
        if debug:
                for i in range(0, len(result["text"])):
                    x = result["left"][i]
                    y = result["top"][i]
                    
                    w = result["width"][i]
                    h = results["height"][i]
                    text = result["text"][i]
                    conf = int(result["conf"][i])
                    if conf > 70:
                        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                        debug_image = cv2.imread(org_img_dir+img_name).copy()[y:y+h, x:x+w] = (0,0,255)
                        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(debug_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
                        cv2.imwrite(output_dir+"/debug/"+name+img_name+"_"+str(i)+".png", debug_image)
        matched = False
        matches = []
        for result in results:
            for text in result["text"]:
                if debug:
                    f.write("\n\nOCR output:\n")
                    f.write(str(text)+"\n")
                tmp = get_close_matches(text, corpus)
                if debug:
                    f.write("Close matches:\n")
                    f.write(str(tmp)+"\n")
                if len(tmp) != 0:
                    matches.append(tmp)
                    matched = True
                else:
                    split = text.split(" ")
                    for s2 in split:
                        if debug:
                            f.write("Close matches run on "+s2+":\n")
                        tmp = get_close_matches(s2, corpus)
                        if debug:
                            f.write(str(tmp)+"\n")
                        if len(tmp) != 0:
                            matches.append(tmp)
                            matched = True

        has_spaces = [label for strs in matches for label in strs if " " in label]
        if has_spaces:
            final[img_name] = has_spaces[0]
            if debug:
                f.write("\n\n-------------------------\nFinal match (first element of list):\n")
                f.write(str(has_spaces)+"\n")
        else: 
            final[img_name]="NO MATCH"
        if matched: cnt+=1
    print("Done.\n")
    return final
    

### --------------------------------- Determine which are same as ground truth/or just output results --------------------------------- ###
def determine_match(gt, final, fname, output_dir):
    f = open(output_dir+fname+"_results.txt", "w")
    cnt = 0
    wcnt = 0
    ncnt = 0
    if gt != None:
        for key,final_val in final.items():
            if gt[key] == final_val:
                f.write(key+": "+final_val+"\n")
                cnt+=1
            else:
                if final_val=="NO MATCH":
                    f.write(key+": "+final_val+"\n")
                    ncnt+=1
                else:
                    f.write(key+"––WRONG: "+final_val+"––EXPECTED:"+gt[key]+"\n")
                    wcnt+=1

        print(fname+" acc: "+str(cnt)+"/"+str(len(final))+" = "+str((cnt/len(final))*100)+"%")
        print(fname+" no match: "+str(ncnt)+"/"+str(len(final))+" = "+str((ncnt/len(final))*100)+"%")
        print(fname+" wrong: "+str(wcnt)+"/"+str(len(final))+" = "+str((wcnt/len(final))*100)+"%"+"\n")
        f.write("\n"+fname+" acc: "+str(cnt)+"/"+str(len(final))+" = "+str((cnt/len(final))*100)+"%")
        f.write("\n"+fname+" no match: "+str(ncnt)+"/"+str(len(final))+" = "+str((ncnt/len(final))*100)+"%")
        f.write("\n"+fname+" wrong: "+str(wcnt)+"/"+str(len(final))+" = "+str((wcnt/len(final))*100)+"%"+"\n")
        f.close()
    else:
        for key,value in final.items():
            f.write(key+": "+value)
        f.close()

def main():
    warnings.filterwarnings("default")
    org_img_dir = None
    output_dir = None
    num_threads = 50
    debug = False
    args = sys.argv[1:]
    if len(args) == 0:
        print("\nUsage: python3 transcribe_labels.py <org_img_dir> [OPTIONAL ARGUMENTS]")
        print("\nOPTIONAL ARGUMENTS:")
        print("\t-o <output_dir>, --output <output_dir>")
        print("\t-n <num_threads>, --num-threads <num_threads> (Default: 50)")
        print("\t-d, --debug\n")
        sys.exit(1)
    if len(args) > 0:
        org_img_dir = args[0]
        if args.count("-o") > 0:
            output_dir = args[args.index("-o")+1]
        if args.count("--output") > 0:
            output_dir = args[args.index("--output")+1]
        if args.count("-n") > 0:
            num_threads = int(args[args.index("-n")+1])
        if args.count("--num-threads") > 0:
            num_threads = int(args[args.index("--num-threads")+1])
        if args.count("-d") > 0 or args.count("--debug") > 0:
            debug = True
    if org_img_dir[-1] != "/":
        org_img_dir += "/"
    if output_dir == None:
        if "/scraped-data/" in org_img_dir:
            output_dir = org_img_dir.replace('/scraped-data/', '/transcription-results/')
        else:
            output_dir = org_img_dir+"results/"
    imgs, geography_corpus, taxon_gt_txt, geography_gt_txt, taxon_corpus = import_process_data(org_img_dir, num_threads)
    ocr_results = ocr(imgs, num_threads)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    taxon_final = match_words_to_corpus(ocr_results, "taxon", taxon_corpus, org_img_dir, output_dir, debug)
    geography_final = match_words_to_corpus(ocr_results, "geography", geography_corpus, org_img_dir, output_dir, debug)
    determine_match(taxon_gt_txt, taxon_final, "taxon", output_dir)
    determine_match(geography_gt_txt, geography_final, "geography", output_dir)

if __name__ == "__main__":
    main()
