import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# get the resulting images and text files
craft_res_dir = "../CRAFT/CRAFT-pytorch-master/result/"
# "/Users/jasonli/Desktop/BU/Junior/Spring2021/CS791/sandbox/test_models/CRAFT-pytorch-master/result"
boxes = []
# imgs = []

for fname in sorted(os.listdir(craft_res_dir)):
	if ".jpg" in fname and "mask" not in fname:
		# imgs.append(cv2.imread(os.path.join(craft_res_dir, fname)))
		tmp_txt = open(os.path.join(craft_res_dir, fname[:len(fname)-3]+"txt"),"r").read().split("\n")[:-1]
		tmp_txt = [line.split(",") for line in tmp_txt]
		tmp_bxs = [[[int(line[i]),int(line[i+1])] for i,val in enumerate(line) if int(i)%2==0] for line in tmp_txt ]
		boxes.append(tmp_bxs)

# get the original images to crop them
# org_img_dir = "/Users/jasonli/Desktop/BU/Junior/Spring2021/CS791/sandbox/test_models/CRAFT-pytorch-master/in_data"
org_img_dir = "../CRAFT/CRAFT-pytorch-master/in_data/"# "/Users/jasonli/Desktop/BU/Junior/Spring2021/CS791/sandbox/herb_dat/imgs"
imgs = []
fnames = []
for fname in sorted(os.listdir(org_img_dir)):
	if ".jpg" in fname:
		imgs.append(cv2.imread(os.path.join(org_img_dir, fname)))
		fnames.append(fname)
n_imgs = len(imgs)
# print(fnames)


### ------------------------------------ HELPER FUNCTIONS ------------------------------------ ###
# determines if two boxes overlap, and if so, combines them
def has_overlap(b1,b2):
	# boxes inputed as [[top left], [top right], [bottom right], [bottom left]]
	tl1, tr1, br1, bl1 = b1
	tl2, tr2, br2, bl2 = b2
	overlap = False

	if tl1[0]>=bl2[0] and tl1[0]<=br2[0] and tl1[1]>=tr2[1] and tl1[1]<=br2[1]: # if top left corner of 1 in 2
		overlap = True
	elif tr1[0]>=bl2[0] and tr1[0]<=br2[0] and tr1[1]>=tr2[1] and tr1[1]<=br2[1]: # if top right corner of 1 in 2
		overlap = True
	elif bl1[0]>=tl2[0] and bl1[0]<=tr2[0] and bl1[1]>=tr2[1] and bl1[1]<=br2[1]: # if bottom left corner of 1 in 2
		overlap = True
	elif br1[0]>=tl2[0] and br1[0]<=tr2[0] and br1[1]>=tr2[1] and br1[1]<=br2[1]: # if bottom right corner of 1 in 2
		overlap = True

	if overlap:
		newTL = [min(tl1[0],tl2[0]),min(tl1[1],tl2[1])]
		newTR = [max(tr1[0],tr2[0]),min(tr1[1],tr2[1])]
		newBL = [min(bl1[0],bl2[0]),max(bl1[1],bl2[1])]
		newBR = [max(br1[0],br2[0]),max(br1[1],br2[1])]
		return [newTL,newTR,newBR,newBL]
 
	return None

# expands boxes according to input margins
def expand_boxes(boxes, diff_axes=False, mx=20, my=40, m=4):
	if not diff_axes:
		mx = m
		my = m
		
	boxes_exp = []
	for box in boxes:
		tl, tr, br, bl = box
		newtl = [tl[0]-mx, tl[1]-my]
		newtr = [tr[0]+mx, tr[1]-my]
		newbr = [br[0]+mx, br[1]+my]
		newbl = [bl[0]-mx, bl[1]+my]
		
		boxes_exp.append([newtl, newtr, newbr, newbl])
		
	return boxes_exp
		
# combines the expanded boxes into a large box (presumably the label)
def combine_boxes(boxes):
	count = len(boxes)
	newboxes = []

	for box in boxes:
		added = False
		
		for i, newb in enumerate(newboxes):
			overlap = has_overlap(box,newb)
			if overlap != None:
				newboxes[i] = overlap
				added = True
				
		if not added:
			newboxes.append(box)

	return newboxes

# sorts the boxes by area smallest to largest
def sort_by_size(boxes):
	areas = []
	for bx in boxes:
		tl,tr,br,bl = bx
		areas.append((br[0]-tl[0])*(br[1]-tl[1]))
		
	return [x for _,x in sorted(zip(areas,boxes))]

# crops the labels from the images
def crop_labels(img, box):
	tl,tr,br,bl = box
	return img[tl[1]:bl[1],tl[0]:tr[0]]

# gets the lines of the image based on text boxes from craft
def get_lines(boxes, vert_m=12):

	newboxes = []
	oldbox = []
	i = 0

	while len(boxes) > 0:
		oldbox = boxes.pop(0)

		tbox2 = boxes.copy()
		for j,b2 in enumerate(boxes): 
			otl, otr, obr, obl = oldbox
			tl, tr, br, bl = b2

			## testing for alignment to connect boxes
			if otr[1]<=tr[1] and obr[1]>=br[1]: # vertically, new box is in range of old box
				pass
			elif otr[1]>=tr[1] and obr[1]<=br[1]: # old box in range of new box
				pass
			elif (abs(otr[1]-tr[1])<=vert_m) and (abs(obr[1]-br[1])<=vert_m): #and ((abs(otr[0]-tl[0])<=adj_m) or (abs(otl[0]-tr[0])<=adj_m)): # within range
				pass
			else:
				continue

			oldbox = [[min(otl[0],tl[0]),min(otl[1],tl[1])],[max(otr[0],tr[0]),min(otr[1],tr[1])],
					  [max(obr[0],br[0]),max(obr[1],br[1])],[min(obl[0],bl[0]),max(obl[1],bl[1])]]

			tbox2.remove(b2)
		boxes = tbox2
		newboxes.append(oldbox)
		
	return newboxes

# crops out images of the lines 
def crop_lines(boxes, imgs):
	line_crops = []
	for i,bxs in enumerate(boxes):
		img_lines = []
		for bx in bxs:
			t1,t2,t3,t4 = bx
			tmp_crop = imgs[i][t1[1]:t4[1],t1[0]:t2[0]]
			if len(tmp_crop) > 0 and len(tmp_crop[0]) > 0:
				img_lines.append(tmp_crop)
		
		line_crops.append(img_lines)
	return line_crops


# segment the labels
boxes_exp = [expand_boxes(bxs, diff_axes=True) for bxs in boxes] # expand boxes
boxes_comb = [combine_boxes(bxs) for bxs in boxes_exp] # combine the expanded boxes
boxes_comb_sorted = [list(reversed(sort_by_size(bxs)))[0] for bxs in boxes_comb] # sort them and take the largest box
labels = [crop_labels(imgs[i], boxes_comb_sorted[i]) for i in range(n_imgs)] # segment label

# segment the lines of text (used to feed into models like mxnet)
lines = [get_lines(bxs) for bxs in boxes]
lines = [expand_boxes(bxs) for bxs in lines]
lines = crop_lines(lines, imgs)

# save cropped labels
save_dir = "../CRAFT/CRAFT-pytorch-master/labels/"
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

for i,label in enumerate(labels):
	cv2.imwrite(os.path.join(save_dir, fnames[i]+"_label.jpg"), label)

# for i,j in enumerate(lines[0]):
# 	cv2.imwrite(os.path.join(save_dir, "test"+str(i)+".jpg"), j)

