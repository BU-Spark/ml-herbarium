# Progress Update

### Secondary Task - Image Classification
Upon soliciting with the clients, with regards to a diverse genera dataset that would cover most of the essential genera, we were asked to work on the New England dataset. We filtered the GBIF dataset to be consistent with the genera in the New England dataset. We successfully managed to extract 120K images along with its genus annotations as ground truth. The 120K images had text that needed to be masked so that the classifier would be bias free. We implemented the KerasOCR pipeline which detects and creates bounding boxes of the text. We removed the masks for the 120K images and are currently working on developing a classifier for 1053 classes of genera in the New England Dataset.
