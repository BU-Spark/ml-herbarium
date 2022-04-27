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

def get_corpus(fname, org_img_dir, words = True):
    corpus_dir = org_img_dir + fname + "_corpus.txt"

    if words: 
        corpus = re.split("\n| ", open(corpus_dir).read()) # split on newline or space
    else:
        corpus = re.split("\n", open(corpus_dir).read()) # split on newline
    corpus = [s.lower() for s in corpus]
    corpus = [s for s in corpus if s != ""]
    corpus = list(set(corpus))

    return corpus

def get_img(image_path):
    warnings.filterwarnings("error")
    try:
        img = np.array(Image.open(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,8)
        # img = cv2.bitwise_not(img)
        img = np.array(img)
    except:
        warnings.filterwarnings("default")
        return {}, image_path
    warnings.filterwarnings("default")
    return {image_path.split("/")[-1][:-4]: img}, None

def get_imgs(imgs, num_threads):
    imgs_out = {}
    failures = []

    print("\nGetting original images and preprocessing...")
    print("Starting multiprocessing...")
    pool = mp.Pool(min(num_threads, len(imgs)))
    for item, error in tqdm(pool.imap(get_img, imgs), total=len(imgs)):
        imgs_out.update(item)
        if error:
            failures.append(error)
    pool.close()
    pool.join()
    for f in failures:
        print("Failed to get image: "+f)
    print("Original images obtained and preprocessing complete.\n")

    return imgs_out

def has_y_overlap(y1, y2, h1, h2):
    if y1 < y2 and y2 < y1 + h1:
        return True
    elif y2 < y1 and y1 < y2 + h2:
        return True
    else:
        return False

def find_idx_right_text(ocr_results, img_name, result_idx):
    results = ocr_results[img_name]
    text = results["text"][result_idx]
    x = results["left"][result_idx]
    y = results["top"][result_idx]
    w = results["width"][result_idx]
    h = results["height"][result_idx]
    xmargin = 3*(w/len(text))
    for i in range(len(results["text"])):
        if i != result_idx:
            x2 = results["left"][i]
            y2 = results["top"][i]
            h2 = results["height"][i]
            if has_y_overlap(y, y2, h, h2) and (x+xmargin+w) > x2:
                return i
    return None

def words_to_lines(ocr_results, x_margin):
    lines = {}
    for img_name,results in ocr_results.items():
        lines[img_name] = []
        for i in range(0, len(results["text"])):
            x = results["left"][i]
            y = results["top"][i]
            w = results["width"][i]
            h = results["height"][i]
            for j in range(0, len(results["text"])):
                if i != j:
                    x2 = results["left"][j]
                    y2 = results["top"][j]
                    w2 = results["width"][j]
                    h2 = results["height"][j]
                    if has_y_overlap(y, y2, h, h2) and ((x+x_margin+w) > x2 or (x2+x_margin+w2) > x):
                        lines[img_name].append([i, j])
    return lines # Returns a dictionary of lines, where each line is a list of indices of words in the image from the ocr_results dictionary
    

### --------------------------------- Import data & process --------------------------------- ###
def import_process_data(org_img_dir, num_threads):
    imgs = sorted(os.listdir(org_img_dir))
    imgs = [org_img_dir + img for img in imgs if img[-4:] == ".jpg"]
    imgs = get_imgs(imgs, num_threads)
    taxon_corpus_words = get_corpus("taxon", org_img_dir, words = True)
    taxon_corpus_full = get_corpus("taxon", org_img_dir, words = False)
    geography_corpus_words = get_corpus("geography", org_img_dir, words = True)
    geography_corpus_full = get_corpus("geography", org_img_dir, words = False)
    taxon_gt_txt = get_gt("taxon", org_img_dir)
    geography_gt_txt = get_gt("geography", org_img_dir)
    
    return imgs, geography_corpus_words, geography_corpus_full, taxon_gt_txt, geography_gt_txt, taxon_corpus_words, taxon_corpus_full


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
    pool = mp.Pool(min(num_threads, len(imgs)))
    func = partial(run_ocr, imgs=imgs)
    for item in tqdm(pool.imap(func, imgs), total=len(imgs)):
        ocr_results.update(item)
    pool.close()
    pool.join()
    print("OCR complete.\n")

    return ocr_results

### --------------------------------- OCR debug output --------------------------------- ###
def ocr_debug(ocr_results, output_dir, imgs, org_img_dir):
    if not os.path.exists(output_dir+"/debug/"):
        os.makedirs(output_dir+"/debug/")
    print("Generating debug outputs...")
    for img_name,results in tqdm(ocr_results.items(), total=len(ocr_results)):
        debug_image = imgs[img_name]
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2RGB)
        orig_image = cv2.imread(org_img_dir+img_name+".jpg")
        for i in range(0, len(results["text"])):
            x = results["left"][i]
            y = results["top"][i]
            
            w = results["width"][i]
            h = results["height"][i]
            text = results["text"][i]
            conf = int(results["conf"][i])
            if conf > 30:
                text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(debug_image, text, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
                cv2.putText(debug_image, "Conf: "+str(conf), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
                cv2.putText(orig_image, text, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
                cv2.putText(orig_image, text, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
            elif conf > 0:
                text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 150, 0), 2)
                cv2.putText(debug_image, text, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
                cv2.putText(debug_image, "Conf: "+str(conf), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
                cv2.putText(orig_image, text, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
                cv2.putText(orig_image, "Conf: "+str(conf), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
        cv2.imwrite(output_dir+"/debug/"+img_name+".png", debug_image)
        cv2.imwrite(output_dir+"/debug/"+img_name+"_orig"+".png", orig_image)
    print("Debug outputs generated.\n")


### --------------------------------- Match words to corpus --------------------------------- ###
def match_words_to_corpus(ocr_results, name, corpus_words, corpus_full, output_dir, debug=False):
    from difflib import get_close_matches
    cnt = 0
    final = {}
    print("Matching words to "+ name +" corpus...")
    for img_name,results in tqdm(ocr_results.items(), total=len(ocr_results)):
        if debug:
            f = open(output_dir+"/debug/"+img_name+"_"+name+".txt", "w")
        matched = False
        matches = {}
        for i in range(len(results["text"])):
            text = results["text"][i].lower()
            conf = int(results["conf"][i])
            if conf > 30:    
                if debug:
                    f.write("\n\nOCR output:\n")
                    f.write(str(text)+"\n")
                    f.write("Confidence: "+str(conf)+"\n")
                tmp = get_close_matches(text, corpus_words, n=1, cutoff=0.8)
                if debug:
                    f.write("Close matches:\n")
                    f.write(str(tmp)+"\n")
                if len(tmp) != 0:
                    matches[i]=tmp[0]
                    matched = True
        if debug:
            f.write("\n\nMatched words:\n")
            f.write(str(matches)+"\n")

        matched_pairs = []
        matched_pairs_matched = False
        for m1 in matches.values():
            for m2 in matches.values():
                if m1 != m2:
                    tmp = get_close_matches(m1+" "+m2, corpus_full, n=1, cutoff=0.9)
                    if len(tmp) != 0:
                        matched_pairs.extend(tmp)
                        matched_pairs_matched = True
        if debug:
            f.write("\n\nMatched pairs:\n")
            f.write(str(matched_pairs)+"\n")

        if matched_pairs_matched:
            final[img_name] = matched_pairs[0]
            if debug:
                f.write("\n\n-------------------------\nFinal match (first element of list):\n")
                f.write(str(matched_pairs))
                f.write("\n-------------------------\n")
        elif matched:
            guesses = []
            for i, m in matches.items():
                guess_idx = find_idx_right_text(ocr_results, img_name, i)
                if guess_idx != None:
                    guesses.append("GUESS: "+ m + ocr_results[img_name]["text"][guess_idx])
            if len(guesses) != 0:
                final[img_name] = guesses[0]
                if debug:
                    f.write("\n\n-------------------------\nGuesses:\n")
                    f.write(str(guesses))
                    f.write("\n-------------------------\n")

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
        for img_name,final_val in final.items():
            if gt[img_name] == final_val:
                f.write(img_name+": "+final_val+"\n")
                cnt+=1
            elif "GUESS" in final_val:
                    if gt[img_name] == final_val.split("GUESS: ")[1]:
                        f.write(img_name+"––"+final_val+"\n")
                        cnt+=1
                    else:
                        f.write(img_name+"––"+final_val+"––EXPECTED:"+gt[img_name]+"\n")
                        ncnt+=1
            else:
                if final_val=="NO MATCH":
                    f.write(img_name+": "+final_val+"––EXPECTED:"+gt[img_name]+"\n")
                    ncnt+=1
                else:
                    f.write(img_name+"––WRONG: "+final_val+"––EXPECTED:"+gt[img_name]+"\n")
                    wcnt+=1

        print(fname+" acc: "+str(cnt)+"/"+str(len(final))+" = "+str((cnt/len(final))*100)+"%")
        print(fname+" no match: "+str(ncnt)+"/"+str(len(final))+" = "+str((ncnt/len(final))*100)+"%")
        print(fname+" wrong: "+str(wcnt)+"/"+str(len(final))+" = "+str((wcnt/len(final))*100)+"%"+"\n")
        f.write("\n"+fname+" acc: "+str(cnt)+"/"+str(len(final))+" = "+str((cnt/len(final))*100)+"%")
        f.write("\n"+fname+" no match: "+str(ncnt)+"/"+str(len(final))+" = "+str((ncnt/len(final))*100)+"%")
        f.write("\n"+fname+" wrong: "+str(wcnt)+"/"+str(len(final))+" = "+str((wcnt/len(final))*100)+"%"+"\n")
        f.close()
    else:
        for img_name,value in final.items():
            f.write(img_name+": "+value)
        f.close()

def main():
    org_img_dir = None
    output_dir = None
    num_threads = mp.cpu_count()
    debug = False
    args = sys.argv[1:]
    if len(args) == 0:
        print("\nUsage: python3 transcribe_labels.py <org_img_dir> [OPTIONAL ARGUMENTS]")
        print("\nOPTIONAL ARGUMENTS:")
        print("\t-o <output_dir>, --output <output_dir>")
        print("\t-n <num_threads>, --num-threads <num_threads> (Default: "+str(num_threads)+")")
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
    imgs, geography_corpus_words, geography_corpus_full, taxon_gt_txt, geography_gt_txt, taxon_corpus_words, taxon_corpus_full = import_process_data(org_img_dir, num_threads)
    ocr_results = ocr(imgs, num_threads)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    if debug:
        ocr_debug(ocr_results, output_dir, imgs, org_img_dir)
    taxon_final = match_words_to_corpus(ocr_results, "taxon", taxon_corpus_words, taxon_corpus_full, output_dir, debug)
    geography_final = match_words_to_corpus(ocr_results, "geography", geography_corpus_words, geography_corpus_full, output_dir, debug)
    determine_match(taxon_gt_txt, taxon_final, "taxon", output_dir)
    determine_match(geography_gt_txt, geography_final, "geography", output_dir)

if __name__ == "__main__":
    main()
