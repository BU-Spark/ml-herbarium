# ML-Herbarium Documentation

# seg_label.py

ml-herbarium/segmentation/seg_label.py

## addBox()

This function takes a filename and returns a dictionary of boxes.
```
Parameters
----------
fname : str
	The name of the file to be processed.

Returns
-------
boxes : dict
	A dictionary of boxes.

Examples
--------
>>> addBox("img_1.jpg")
{'1': [[[0, 0], [0, 0], [0, 0], [0, 0]]]}
```



## fillBoxes()

This function fills the boxes dictionary with the following format:
```
boxes = {
	"img_name": {
		"box_id": {
			"x": x,
			"y": y,
			"w": w,
			"h": h,
			"text": text
		}
	}
}
```



## addImg()

This function adds the image to the dictionary.
```
Parameters
----------
fIdx : str
	The file name of the image.

Returns
-------
imgs : dict
	The dictionary of images.
```



## getOrigImgs()

```
def getOrigImgs():
	print("Getting original images...")
	print("Starting multiprocessing...")
	pool = mp.Pool(NUM_CORES)
	for item in tqdm(pool.imap(addImg, boxes), total=len(boxes)):
		imgs.update(item)
	pool.close()
	pool.join()
	print("\nOriginal images obtained.\n")
```



## has_overlap()
This function takes two boxes as input and returns the box that is the overlap between the two boxes.
The boxes are inputed as [[top left], [top right], [bottom right], [bottom left]]
If there is no overlap, the function returns None.



## expand_boxes()

Expands the boxes in bxs by mx and my.
```
expand_boxes(bxs, diff_axes=False, mx=20, my=40, m=4)

Parameters
----------
bxs : list of lists of lists of ints
	The boxes to expand.
diff_axes : bool
	Whether to expand the boxes by different amounts in the x and y axes.
mx : int
	The amount to expand the boxes by in the x axis.
my : int
	The amount to expand the boxes by in the y axis.
m : int
	The amount to expand the boxes by in both axes if diff_axes is False.

Returns
-------
boxes_exp : list of lists of lists of ints
	The expanded boxes.
```



## combine_boxes()
Combines boxes that overlap.
```
Parameters:
	boxes: A list of boxes.
	
Returns:
	A list of boxes.
```



## sort_by_size()


This function takes a list of boxes, where each box is a list of four points,
and returns the boxes sorted by size. The size of a box is defined as the area of the rectangle formed by the four points. The points are given as tuples of two integers, the x and y coordinates. The boxes are sorted in ascending order of size. The function returns a list of boxes.




## crop_labels()

Crops the labels from the image.
```
Parameters
----------
img : numpy.ndarray
	The image to crop.
box : list
	The box to crop.

Returns
-------
numpy.ndarray
	The cropped image.
```



## get_lines()

This function takes a list of boxes and returns a list of boxes that are connected.
It does this by iterating through the list of boxes and checking if the next box is
within a certain range of the current box. If it is, it is added to the current box.
This is repeated until the list is empty.
```
Parameters:
	boxes (list): A list of boxes.
	vert_m (int): The vertical margin of error.

Returns:
	newboxes (list): A list of boxes that are connected.
```



## crop_lines()
This function takes in a list of lists of tuples, where each tuple is a coordinate of a box.
It also takes in a list of images.
It returns a list of lists of images, where each image is a line of text.




## main()

This function is the main function of the program. It takes in the images and boxes, and segments the labels.
```
Parameters
----------
None

Returns
-------
None
```



