import os
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

from string_grouper import match_strings, match_most_similar # You have to install 0.5.0 version, 'pip install string_grouper=='

# Multiprocessing stuff
import multiprocessing as mp
NUM_CORES = min(mp.cpu_count(), 50)

# Import PaddleOCR class from paddleocr
from paddleocr import PaddleOCR, draw_ocr


class My_DataLoader():
    """A general purpose dataloader for loading images in batch
        Examples:
        >>> batch_size= 16
        >>> SRC_DIR = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/>>> drago_testdata/images/test/"
        >>> data_loader = My_DataLoader(SRC_DIR, batch_size)
        >>> batch_dict, label_list = data_loader.get_next_batch()
        >>> print(batch_dict, label_list)
    """
    def __init__(self, SRC_DIR, batch_size=16):
        """
        Args:
            @SRC_DIR: the source directory that contains all image your want to read
            @batch_size: the batch size you want to use for batch processing
        """
        self.SRC_DIR = SRC_DIR
        self.batch_size = batch_size
        self.imgIds_list = self.__get_all_imgesIDs()
        self.batch_dict = {}    # Dict<int: imgId, nparr: img>
        self.loading_idx = 0
        
    def __get_all_imgesIDs(self):
        """Save all image filename into a list
        Args:
        Return:
            @imgIds_list: a list of str, where str is the representation format for image id
        """
        print("Reading all images id into a list...")
        file_list = sorted(os.listdir(self.SRC_DIR))
        imgIds_list = [img[:-4] for img in file_list if ".jpg" == img[-4:]]
        return imgIds_list
        
    def __getitem__(self, imgID, package="pillow"):
        """This function is use when the dataloader is used as a Dictionary, e.g., my_dataloader[imgID]. So, for given imgID. Retrun the image if success; Return False, otherwise.
        Args:
            @imgID: str representation for image id.
            @package: what package to use? e..g, 'pillow', or 'cv2', usuauly people say cv2 is faster, but harder to use, so use 'pillow' in default you not sure
        Return :
            @img: return an image in nparr if in success, or False otherwise, with 3 dimension, (H, W, C), where C is the number of channel, normally 3 for color image
        """
        f_path = os.path.join(self.SRC_DIR, imgID+".jpg")
        try:  # Generally pillow is faster!
            if package=='cv2':
                # With cv2
                img =cv2.imread(os.path.join(self.SRC_DIR, imgID+".jpg"))
                if img is not None:
                    self.batch_dict[imgID] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif package == "pillow":
                # With Image or PIllow package
#                 img = Image.open(f_path)
#                 img.verify() # Check that the image is valid
                with Image.open(f_path) as img:
                    img.load()
                img = np.asarray(img.convert("RGB"))
            else:
                print(f"Error, not implemented! Abort process")
                os.exit(0)
            # Read more here for other methods of loading images, https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
            return img
        except (IOError, SyntaxError) as e:
            print('file ', f_path, ' is not a valid image file')
            print(e)
            return False

    def get_next_batch(self, start_idx=None, package='pillow', remove_invalid_image=False):
        """ Read and load a batch of the image into memory. Multiprocessing is supported
        Args:
            @multi_proc, bool, whether to use multiprocessing or not
            @method, str, what package should use to load the image
        Return:
            @img_dict, dict, a dictionary contains all the image. Mapping imgId to image data.
            @label_list, a list that contains the all imgID as the key index for {@img_dict}
        """
        # Clear all images store in the self.batch_dict -- There is memory limit, so we always have to clear the dictionary
        self.batch_dict.clear()
        invalid_path = []
        if start_idx is None:
            start_idx = self.loading_idx
        if start_idx>=len(self.imgIds_list):
            print(f"Reached end of dataloader, self.loading_idx = {self.loading_idx}")
            return None, None

        ending_dix = start_idx + self.batch_size 
        if ending_dix > len(self.imgIds_list):
            ending_dix = len(self.batch_dict)

        label_list = self.imgIds_list[start_idx:ending_dix]
        for fIdx in label_list:
            self.loading_idx += 1
            img = self.__getitem__(fIdx)
            if img is False:
                invalid_path.append(fIdx)
            else:
                self.batch_dict[fIdx] = img

        print(f"\n{len(self.batch_dict)} of original images obtained.\n")
        print(f"All invalid imageID: {invalid_path}")
        if remove_invalid_image:
            print("Removing all invalid image:")
            for fIdx in invalid_path:
                os.remove(os.path.join(self.SRC_DIR, fIdx+".jpg")) 
        return self.batch_dict, label_list

    def get_gt_dict(self):
        """Load all the ground truth label from given directory define with My_Dataloader.
        Returns:
            @gt_dict: a dictionary contains 
        """
        # Building gt_dict
        gt_dict = {}
        str_lst = ['taxon', 'geography', 'collector']
        # GT_LABEL_DIR = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/drago_testdata/gt_labels"
        GT_LABEL_DIR = self.SRC_DIR
        Taxon_truth = { line.split(":")[0] : line.split(": ")[1].strip() for line in open(os.path.join(GT_LABEL_DIR + '/taxon_gt.txt')) }
        Geography_truth = { line.split(":")[0] : line.split(": ")[1].strip() for line in open(GT_LABEL_DIR + '/geography_gt.txt') }
        Collector_truth = { line.split(":")[0] : line.split(": ")[1].strip() for line in open(GT_LABEL_DIR + '/collector_gt.txt') }
        # comparison_file = {"Taxon":Taxon_truth,"Geography":Geography_truth,"Collector":Collector_truth}
        for imgId in self.imgIds_list:
            if Taxon_truth.get(imgId):
                gt_dict[imgId] = {str_lst[0]: Taxon_truth[imgId], str_lst[1]: Geography_truth[imgId], str_lst[2]: Collector_truth[imgId]}
            else:
                print(f"Warning: Bad image, {imgId} doesn't exist in Ground Truth Label!")
        return gt_dict
    
    def get_all_corpus_set(self):
        """Building corpus set, Use set for storing all taxon, geograph, and collector, with O(1) look up, and remove duplicated item
        Returns:
            @corpus_set: a list of three item, [taxon_corpus_set, geography_corpus_set, collector_corpus_set]
        """
        TAXON_CORPUS_PATH = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/drago_testdata/full_corpus/corpus_taxon.txt"
        GEOGRAPH_CORPUS_PATH = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/drago_testdata/full_corpus/corpus_geography.txt"
        COLLECTOR_CORPUS_PATH = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/drago_testdata/corpus/collector_corpus.txt"
        taxon_corpus_set     = set(pd.read_csv(TAXON_CORPUS_PATH, delimiter = "\t", names=["Taxon"]).squeeze())
        geography_corpus_set = set(pd.read_csv(GEOGRAPH_CORPUS_PATH, delimiter = "\t", names=["Geography"]).squeeze())
        collector_corpus_set = set(pd.read_csv(COLLECTOR_CORPUS_PATH, delimiter = "\t", names=["Collector"]).squeeze())
        all_corpus_set = [taxon_corpus_set, geography_corpus_set, collector_corpus_set]
        return all_corpus_set


# if __name__ == '__main__':