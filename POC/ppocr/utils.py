import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
print(np.__version__)

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
        
    

def check_images(s_dir, ext_list):
    """A function to check all invalid image
    Args:
        @s_dir: the director of image folder
        @ext_list: a list of good image format in str.
    Example:
        >>> source_dir = r'/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/>>> scraped-data/drago_testdata/image/'
        >>> good_exts=['jpg', 'png', 'jpeg'] # list of acceptable extensions
        >>> bad_file_list, bad_ext_list=check_images(source_dir, good_exts)
        >>> if len(bad_file_list) !=0:
        >>>     print('improper image files are listed below')
        >>>     for i in range (len(bad_file_list)):
        >>>         print (bad_file_list[i])
        >>> else:
        >>>     print(' no improper image files were found')
    """
    bad_images=[]
    bad_ext=[]
    s_list= os.listdir(s_dir)
    for f in s_list:               
        f_path=os.path.join (s_dir,f)
        index=f.rfind('.')
        ext=f[index+1:].lower()
        if ext not in ext_list:
            print('file ', f_path, ' has an invalid extension ', ext)
            bad_ext.append(f_path)
        if os.path.isfile(f_path):
            try:
                img=cv2.imread(f_path)
                shape=img.shape
                image_contents = tf.io.read_file(f_path)
                image = tf.image.decode_jpeg(image_contents, channels=3)
            except Exception as e:
                print('file ', f_path, ' is not a valid image file')
                print(e)
                bad_images.append(f_path)
        else:
            print('*** fatal error, you a sub directory ', f, ' in class directory ', f)
#         else:
#             print ('*** WARNING*** you have files in ', s_dir, ' it should only contain sub directories')
    return bad_images, bad_ext

from paddleocr import PaddleOCR, draw_ocr
def display_OCR_result_with_imgID(img_dict, imgID, save_path=None, display_result=False):
    """
    Args:
        @img_dict: Dict<int: imgId, nparr: img>, can be used for extracting image 
        @imgID: str, a str format id that must exist in img_dict, for the image to display
    Example:
        >>> timestr = time.strftime("%Y%m%d%H%M%S_")
        >>> save_path = os.path.join(PROJECT_DIR+"output/")+timestr+".jpg"
        # With batch dict
        >>> display_OCR_result_with_imgID(batch_dict, '1019531437', display_result=True)
        # With Single image
        >>> imgID = str(imgID)
        >>> save_path= '/usr4/dl523/dong760/CS549_Herbarium_Project/ml-herbarium/PaddleOCR/output'
        >>> temp_img_dict = {}
        >>> temp_img_dict[imgID] = data_loader[imgID]
        >>> display_OCR_result_with_imgID(temp_img_dict, imgID, save_path=save_path, display_result=False)
    """
    if imgID not in img_dict.keys():
        print(f"Error, {imgID} doesn't exist in {img_dict.keys()}")
    # Get img as nparr
    img = img_dict[imgID]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Use PaddleOCR to make prediction
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu = False, use_mp=True, show_log=False)
    result = ocr.ocr(img, cls=True)
    result = np.squeeze(np.array(result), axis=0) # Remove redundant dim and transform to nparr, e.g., [1, 19, 4, 2] --> [19, 4, 2]

    # draw result
    from PIL import Image
    print(result)
    print(result.shape)
    # image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(img,   # image(Image|array): RGB image
                       boxes, # boxes(list): boxes with shape(N, 4, 2)
                       txts,  # txts(list): the texts
                       scores, # confidence
                       drop_score=0.1, # drop_score(float): only scores greater than drop_threshold will be visualized
                       font_path='/usr4/dl523/dong760/CS549_Herbarium_Project/ml-herbarium/PaddleOCR/doc/fonts/simfang.ttf') # font_path: the path of font which is used to draw text
    im_show = Image.fromarray(im_show)
    if save_path:
        save_path = os.path.join(save_path, imgID+"_pred_result.jpg")
        im_show.save(save_path)  # Save the result
    if display_result:
        plt.figure("results_img", figsize=(30,30))
        plt.imshow(im_show)
        plt.show()


def display_img_with_imgID(imgID, img_dict):
    """The function will simple used to display an image, for a given imgID where the corresponding image must exist in img_dict, a dictionary.
    Args:
        @img_dict: Dict<int: imgId, nparr: img>, can be used for extracting image 
        @imgID: str, a str format id that must exist in img_dict, for the image to display
    """
    img = img_dict[imgID]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)