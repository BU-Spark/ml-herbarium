# Project Research for ML Herbarium 
*Team Members: Zhengqi Dong, Cole Hunter, Zihua Xin, Mark Yang*

One of the features of this project is that it essentially has two components which create the pipeline needed to produce the desired output. First, the text in images needs to be segmented, which is typically referred to as text detection. Then, the segmented images containing the text need to be evaluated to try and figure out what characters are present in the picture. This step is typically referred to as recognition. 

As discussed below, after looking at the current state of segmentation methods, CRAFT remains the most robust option available, and for this reason we do not expect to modify the previous group’s approaches in using CRAFT for segmentation. However, after researching potential improvements in the recognition phase of the pipeline, we have found a few promising approaches which may be able to improve the project's results, particularly for handwritten text. 

## Open Source Implementations:

[Tr-OCR](https://arxiv.org/abs/2109.10282)

This approach uses a visual transformer as the encoder and a text transformer as the decoder in order to recognize the characters that are in an image. The base model has been trained on hundreds of millions of text images, and in the paper it is recommended to further fine-tune the model on the desired type of text that will be evaluated, either handwritten or printed text. This is particularly interesting for our group, as one of the difficulties that has been noted by previous groups was trying to accurately predict handwritten text. Moving forward, we plan to build out a pipeline that uses CRAFT to get the segmentation boxes for all text in a given herbarium specimen image, then feed those individual text boxes into Tr-OCR for the recognition step. After this has been completed, we can compare the results with those from the current Tesseract implementation to see if it provides better results. Another advantage is that unlike Tesseract, Tr-OCR can be run on a gpu which may provide a significant speed boost in evaluating the expansive collection of images contained in the herbarium. 

[PP-OCR](https://arxiv.org/pdf/2009.09941.pdf)

This is another approach we will be looking at as an alternative to the current tesseract implementation. Like Tr-OCR, this method is able to run on a GPU, which can provide significant speed improvements. One of the reasons we are interested in trying this method is that it seems to do quite well on extracting non-english text from images (particularly chinese). Anecdotal evidence from people comparing its performance with tesseract online seems to suggest that in certain use cases, it might provide a better result. In order to test whether that will be the case for our specific needs, we will be testing its performance on our data moving forward. 

## Research Papers:
[A Comprehensive Study of Optical Character Recognition
](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9837974&tag=1)

A survey paper presents an overview of the OCR technology, its history, major applications and potential future works. This paper highlights the particular challenges for OCR, including difficulty in adapting to different writing styles, different alphabets and fonts of various languages, several languages in the world, quality of scanned images, and quality of paper used for writing texts.

[State-of-the-Art in Action: Unconstrained Text Detection](https://openaccess.thecvf.com/content_ICCVW_2019/papers/RLQ/Nguyen_State-of-the-Art_in_Action_Unconstrained_Text_Detection_ICCVW_2019_paper.pdf) 

This paper presents comparisons between a number of different approaches for segmenting text in images, those being CRAFT, CTPN, EAST, FOTS, PixelLink, and PSENet. It shows that in most cases, CRAFT provides the best segmentation, most likely owing to the fact that it bases its neighborhood grouping on characters in the image rather than on individual pixels. There are also portions of the paper that consider the speed of each implementation, which again show CRAFT as a top performer. The final conclusion is that CRAFT remains the most competent segmentation algorithm of those evaluated, with the researchers mentioning that the post-processing steps taken after initial segmentation are an important part in getting accurate results from any of the models. 

[What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis](https://arxiv.org/pdf/1904.01906.pdf)

This paper attempted to set the standard process by which to compare the effectiveness of OCR models. The authors detail the difficulties of defining certain architectures as state of the art, when the comparisons between models might not be fair to begin with. To do this, the authors tested model implementations against the same test sets. Another important note that is mentioned in this paper is that the diversity of training samples proved to the models during training proved to be a more important factor in improving model accuracy when compared with simply increasing the number of images that were in the training set. This will inform how we move forward in fine-tuning the models we intend to deploy. 

[Text Recognition in the Wild: A Survey](https://arxiv.org/pdf/2005.03492.pdf)

Another survey paper, this provided a comprehensive description of the present difficulties of deploying effective OCR models, while comparing their performance and detailing the authors expectations on what the next steps might be in order to improve current approaches. One recommendation is that it may be feasible to deploy language specific models in order to better capture the text within an image. They also detail the importance of having strong segmentation as the backbone of any OCR application; without it, recognition portions in any pipeline are unlikely to produce the desired results. 
