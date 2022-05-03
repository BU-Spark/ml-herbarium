import time
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import imutils
from imutils import perspective
from imutils import contours
from tqdm import tqdm

NUM_CORES = min(mp.cpu_count(), 50)

# get the resulting images and text files
craft_res_dir = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/CRAFT-results/20220425-160006/"
# "/Users/jasonli/Desktop/BU/Junior/Spring2021/CS791/sandbox/test_models/CRAFT-pytorch-master/result"
boxes = {}
# imgs = []

def addBox(fname):
	if ".jpg" in fname and "mask" not in fname:
		# imgs.append(cv2.imread(os.path.join(craft_res_dir, fname)))
		tmp_txt = open(os.path.join(craft_res_dir, fname[:len(fname)-3]+"txt"),"r").read().split("\n")[:-1]
		tmp_txt = [line.split(",") for line in tmp_txt]
		tmp_bxs = [[[int(line[i]),int(line[i+1])] for i,val in enumerate(line) if int(i)%2==0] for line in tmp_txt ]
		boxes[fname[4:len(fname)-4]] = tmp_bxs
		return boxes

def fillBoxes():
	print("\nFilling boxes dictionary...")
	print("Starting multiprocessing...")
	list_imgs = sorted(os.listdir(craft_res_dir))
	pool = mp.Pool(NUM_CORES)
	for item in tqdm(pool.imap(addBox, list_imgs), total=len(sorted(os.listdir(craft_res_dir)))):
		if item: boxes.update(item)
	pool.close()
	pool.join()
	print("\nBoxes dictionary filled.\n")

# get the original images to crop them
# org_img_dir = "/Users/jasonli/Desktop/BU/Junior/Spring2021/CS791/sandbox/test_models/CRAFT-pytorch-master/in_data"
org_img_dir = "/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/scraped-data/20220425-160006/"# "/Users/jasonli/Desktop/BU/Junior/Spring2021/CS791/sandbox/herb_dat/imgs"
imgs = {}

def addImg(fIdx):
	imgs[fIdx]=cv2.imread(os.path.join(org_img_dir, fIdx+".jpg"))
	return imgs

def getOrigImgs():
	print("Getting original images...")
	print("Starting multiprocessing...")
	pool = mp.Pool(NUM_CORES)
	for item in tqdm(pool.imap(addImg, boxes), total=len(boxes)):
		imgs.update(item)
	pool.close()
	pool.join()
	print("\nOriginal images obtained.\n")
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
def expand_boxes(bxs, diff_axes=False, mx=20, my=40, m=4):
	if not diff_axes:
		mx = m
		my = m
		
	boxes_exp = []
	for box in bxs:
		tl, tr, br, bl = box
		newtl = [tl[0]-mx, tl[1]-my]
		newtr = [tr[0]+mx, tr[1]-my]
		newbr = [br[0]+mx, br[1]+my]
		newbl = [bl[0]-mx, bl[1]+my]
		
		boxes_exp.append([newtl, newtr, newbr, newbl])
		
	return boxes_exp

def expand_boxes2(img_txt, xmarg, ymarg):
	boxes = []
	for i, box in enumerate(img_txt):
		coord = []
		temppt = []
		xcoords = []
		ycoords = []
		#split the img_txt line into x and y coordinates, separated by commas
		for j, val in enumerate(box):
			val = val[0], val[1]
			if j%2==0:
				temppt = []
				xcoords.append(val[0])
			temppt.append(val)
			if j%2!=0:
				coord.append(temppt)
				ycoords.append(val[1])
		xavg = np.mean(xcoords)
		yavg = np.mean(ycoords)
		coords = []
		for j, val in enumerate(coord):
			coords.append(val[0])
			coords.append(val[1])
		nc = []
		for k, v in enumerate(coords):
			if v[0] >= xavg:
				nc.append(coords[k][0]+xmarg+20)
			elif v[0] < xavg:
				nc.append(coords[k][0]-xmarg-80)
			if v[1] >= yavg:
				nc.append(coords[k][1]+ymarg)
			elif v[1] < yavg:
				nc.append(coords[k][1]-ymarg)
		boxes.append(nc)
	box = [[[0,0],[0,0],[0,0],[0,0]]] * len(boxes)
	#print(box)
	for i, b in enumerate(box):
		#print(b)
		box[i] = [[boxes[i][0],boxes[i][1]],[boxes[i][2],boxes[i][3]],[boxes[i][4],boxes[i][5]],[boxes[i][6],boxes[i][7]]]
		# print(box[i])
		# print(boxes[i])
	
	return box
		
# combines the expanded boxes into a large box (presumably the label)
def combine_boxes(boxes):
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
fillBoxes()
getOrigImgs()
boxes_exp = {key: expand_boxes2(bxs, 40, 40) for key, bxs in boxes.items()} # expand boxes
boxes_comb = {key: combine_boxes(bxs) for key, bxs in boxes_exp.items()} # combine the expanded boxes
boxes_comb_sorted = {key: list(reversed(sort_by_size(bxs)))[0] for key, bxs in boxes_comb.items()} # sort them and take the largest box
labels = {} # segment label

for key, image in imgs.items():
	try:
		labels[key]=crop_labels(image, boxes_comb_sorted[key])
	except:
		labels[key]=None
		print("Error cropping label for image: ", key+".jpg")

# # segment the lines of text (used to feed into models like mxnet)
# lines = [get_lines(bxs) for bxs in boxes]
# lines = [expand_boxes(bxs) for bxs in lines]
# lines = crop_lines(lines, imgs)

# save cropped labels
timestr = time.strftime("%Y%m%d-%H%M%S")

save_dir = '/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/seg-results/'+timestr+'/'
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

for key, label in labels.items():
	try:
		plt.imsave(os.path.join(save_dir, key+"_label.jpg"), label)
		img = cv2.imread(os.path.join(save_dir, key+"_label.jpg"))
		grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		grey = cv2.GaussianBlur(grey, (7,7), 0)
		#find borders for the edges of the image
		boundingbb = cv2.Canny(grey, 50, 100)
		boundingbb = cv2.dilate(boundingbb, None, iterations=1)
		boundingbb = cv2.erode(boundingbb, None, iterations=1)
		# find contours
		contours, hierarchy = cv2.findContours(boundingbb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# find the bounding box of the contours
		countours = countours[0] if imutils.is_cv2() else countours[1]
		for c in countours:
			if cv2.contourArea(c) < 1000:
				continue
			box = cv2.minAreaRect(c)
			box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
			box = np.array(box, dtype="int")
			box = perspective.order_points(box)
			
			orig = img.copy()
			
			cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 5)
			
			for (x,y) in list(box):
				cv2.circle(orig, (int(x), int(y)), 7, (0, 0, 255), -1)
				#cv2.putText(orig, "{}, {}".format(x, y), (int(x) - 20, int(y) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
				cv2.imwrite(os.path.join(save_dir, key+"_label_contour.jpg"), orig)
			
			cv2.waitKey(0)
		# (x, y, w, h) = cv2.boundingRect(countours)
		# # crop the image
		# cropped = img[y:y+h, x:x+w]
		# plt.imsave(os.path.join(save_dir, key+"_label_cropped.jpg"), cropped)
	except Exception as e:
		print(e)
		print("Error saving label for image: ", key+".jpg")


# for i,j in enumerate(lines[0]):
# 	cv2.imwrite(os.path.join(save_dir, "test"+str(i)+".jpg"), j)

