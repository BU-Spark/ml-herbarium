# ML-Herbarium Documentation

# train_model.py

/usr4/ugrad/en/ml-herbarium/training_model/train_model.py

## resize_image()

```
    Helper function to resize an image while keeping the aspect ratio.
    Parameters
    ----------
    image : np.array
        The image to be resized.
    desired_size : (int, int)
        The (height, width) of the resized image
    Returns
    -------
    image : np.array
        The image of size = desired_size
    bounding box : (int, int, int, int)
        (x, y, w, h) in percentages of the resized image of the original
```



## _pre_process_image()

```
    This function takes in an image and returns a processed image.
    The processing includes:
        1. Resizing the image to a fixed size.
        2. Converting the image to grayscale.
        3. Normalizing the image.
        4. Converting the image to a numpy array.
    Args:
        img_in: The image to be processed.
    Returns:
        img_arr: The processed image as a numpy array.
```



## transform()

```
    This function resizes the input image and converts so that it could be fed into the network.
    Furthermore, the label (text) is one-hot encoded.

    Parameters
    ----------
    image : numpy.ndarray
        The image to be transformed.
    label : str
        The label (text) to be transformed.

    Returns
    -------
    image : numpy.ndarray
        The transformed image.
    label_encoded : numpy.ndarray
        The transformed label.
```



## augment_transform()

```
    This function randomly:
        - translates the input image by +-width_range and +-height_range (percentage).
        - scales the image by y_scaling and x_scaling (percentage)
        - shears the image by shearing_factor (radians)

    Args:
        image: A numpy array of shape (height, width, channels).
        label: A numpy array of shape (height, width, channels).

    Returns:
        A tuple of (image, label) where image and label are numpy arrays of shape (height, width, channels).
```



## draw_text_on_image()

```
    Draws text on the bottom of the image.
    
    Parameters
    ----------
    images : numpy.ndarray
        A numpy array of shape (batch_size, height, width, channels)
    text : list
        A list of strings of length batch_size
    
    Returns
    -------
    output_images : numpy.ndarray
        A numpy array of shape (batch_size, height, width*2, channels)
```



## run_epoch()

```
Runs an epoch of training or testing.

Parameters
----------
e : int
    The current epoch number.
network : gluon.Block
    The network to train or test.
dataloader : gluon.data.DataLoader
    The dataloader to use for the epoch.
trainer : gluon.Trainer
    The trainer to use for the epoch.
log_dir : str
    The directory to log to.
print_name : str
    The name to print for the epoch.
update_network : bool
    Whether to update the network.
save_network : bool
    Whether to save the network.

Returns
-------
float
    The epoch loss.
```



## __init__()

```
This function is used to pre-process the image.

Parameters
----------
image_filename : str
    The path of the image file.

Returns
-------
image_arr : numpy.ndarray
    The image array.

Raises
------
ValueError
    If the image is not a valid image.

```



## __getitem__()

```
    This function returns the image and the output for the given index.
    :param idx: The index of the image.
    :return: The image and the output.
```



# evaluate_cer.py

/usr4/ugrad/en/ml-herbarium/training_model/ocr/evaluate_cer.py

## get_arg_max()

```
The greedy algorithm convert the output of the handwriting recognition network
into strings.

Parameters
----------
prob : mxnet.ndarray.ndarray.NDArray
    The output of the handwriting recognition network.

Returns
-------
str
    The string of the handwriting recognition.
```



# word_and_line_segmentation.py

/usr4/ugrad/en/ml-herbarium/training_model/ocr/word_and_line_segmentation.py

