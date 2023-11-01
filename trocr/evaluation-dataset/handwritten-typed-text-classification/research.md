# [Research] TrOCR Encoder + FFN Decoder

#### Overview
To create a robust classification model for our task, multiple Convolutional Neural Network (CNN) models were explored and assessed. Details of each attempted model, along with their respective implementations, can be accessed in the [Introduction section](https://github.com/BU-Spark/ml-herbarium/blob/dev/trocr/evaluation-dataset/handwritten-typed-text-classification/notebooks/Classifier_NN.ipynb) of the project's Jupyter Notebook.

#### Issues Encountered with CNNs
During experimentation, I identified fundamental limitations with how CNNs process images containing text, affecting our ability to accurately classify text in images into either handwritten or machine-printed categories. 

In specific, it was observed that the text in images, particularly handwritten text, constitutes a minimal portion of the image in terms of pixel count, thereby reducing our Region of Interest (ROI). This small ROI posed challenges in information retention and propagation when image filters were applied, leading to the loss of textual details. To mitigate this, I employed the morphological operation of **erosion** on binarized images to emphasize the text, effectively enlarging the ROI. This process proved useful in counteracting some of the undesirable effects of CNN filters and preserving the integrity of the text in the images.

#### Methodology
Given the encountered limitations with CNNs, I approached the classification task in two primary steps to circumvent the challenges:

1. **Feature Extraction with TrOCR Encoder:**
   Leveraged the encoder part of the TrOCR model to obtain reliable feature representations from the images, focusing on capturing inherent characteristics of text. The encoder from TrOCR was employed due to its capability to retain textual details in its feature representations, which are pivotal for decoding to characters. This stands in contrast to Convolutional Neural Networks (CNNs), which might not preserve such detailed textual information. In essence, the encoder within TrOCR ensures the conservation of textual nuances that are potentially overlooked or lost when using CNNs.

2. **Training a Custom FFN Decoder:**
   Employed a custom Feed-Forward Neural Network (FFN) as the decoder to make predictions based on the feature representations extracted from the encoder. The model was trained specifically to discern the subtle differences in features between the two categories.

This methodology enabled to maintain a high level of accuracy and reliability in our classification task while overcoming the inherent shortcomings identified in CNN models for processing images with text.

#### Readings

The [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) paper inspired me to use this encoder-decoder architecture. In this paper, the authors use multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector. Additionally, BERT-like architectures also act as an inspiration to the encoder-decoder paradigm.

This approach of utilizing an FFN as a decoder, post feature extraction, is important in handling various classification tasks, especially when dealing with specialized forms of data like text in images, because it allows us to define a custom network specific to our task.

#### Results Summary

In our handwritten vs. typed-text classification task, the model performed impressively with an overall accuracy of \(96\%\). The test samples were handpicked to be challenging for the model to classify (since some of these were misclassified by a human).

- *Handwritten Text Class:*
  - *Precision:* \(97.96\%\)
  - *Recall:* \(96.00\%\)
  - *F1-Score:* \(96.97\%\)
  - *Support:* 50 samples

- *Typed Text Class:*
  - *Precision:* \(96.23\%\)
  - *Recall:* \(98.08\%\)
  - *F1-Score:* \(97.14\%\)
  - *Support:* 52 samples

The balanced performance across both classes, as shown in the nearly identical macro average and weighted average metrics, demonstrates the model's robustness in distinguishing between handwritten and typed texts.

