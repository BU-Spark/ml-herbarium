# Herbaria Research Document

### *Smriti Suresh, Dima Kazlouski, Douglas Moy - 2023-10-15 v1.0.0-dev*

## Research Papers 

**1. CRAFT**
* The paper "Character Region Awareness for Text Detection" by Youngmin Baek and his colleagues introduces a novel approach to text detection in images. The key idea is to improve the accuracy of text detection by focusing on character-level regions within the text.
* The method proposed in the paper leverages the insight that text regions are composed of individual characters and their spatial relationships. Instead of treating text as a single entity, it breaks down text into character-level regions and uses a Convolutional Neural Network (CNN) to detect these regions. This character-aware approach helps the model handle irregular or curved text, as it can adapt to the specific shapes of characters and their interactions.
* The paper also introduces a Character Region Proposal Network (CRPN) that generates character region proposals, which are then refined for more accurate text detection. By focusing on characters, the model achieves state-of-the-art results in text detection tasks, particularly for challenging scenarios like curved and multi-oriented text in natural scenes.

**2. TrOCR**
* The paper "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models" by Minghao Li and his collaborators introduces a state-of-the-art approach to Optical Character Recognition (OCR) using Transformer-based deep learning models.
* Traditional OCR systems use Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) to recognize characters in images. In contrast, TrOCR leverages Transformer models, which have been highly successful in natural language processing tasks, and adapts them for OCR.
* The key innovation in TrOCR is the use of pre-trained models. It fine-tunes large-scale pre-trained Transformer models, such as BERT or RoBERTa, on OCR datasets. This transfer learning approach significantly improves the model's performance, making it robust to various fonts, languages, and text layouts.
* TrOCR can handle a wide range of OCR tasks, including printed and handwritten text recognition. It also supports multi-lingual text recognition, making it highly versatile. The paper showcases competitive results, outperforming traditional OCR methods and other deep learning-based OCR systems.

**3. What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis**

* The authors first highlight the inadequacies of existing benchmark datasets for scene text recognition, emphasizing that these datasets do not fully represent the complexity of real-world text scenarios. They introduce a new dataset called STR-SynthText, which contains a diverse range of text variations to better reflect real-world challenges, such as different fonts, styles, and backgrounds.
* The paper also provides a comprehensive analysis of various scene text recognition models, comparing their performances on the new dataset and highlighting the limitations of existing evaluation metrics. The authors emphasize that model evaluation in this domain must consider factors like character-level accuracy and the robustness of the models to different text variations.

**4. Handwriting Transformers**

* The paper presents a transformer-based model named HWT for generating styled handwritten text images, focusing on entangling style and content alongside discerning both global and local writing style patterns.
* The model architecture comprises an encoder-decoder network; the encoder generates a self-attentive style feature sequence, which is utilized by the decoder to produce character-specific style attributes, thus creating the final styled handwritten text image.
* The model excels in imitating a given writer's style for specific query content, showcasing promising results in evaluations, particularly in unseen styles and out-of-vocabulary conditions, making it a significant contribution to styled handwritten text generation

**5. Full Page Handwriting Recognition via Image to Sequence**
* This paper introduces a Neural Network-based Handwritten Text Recognition (HTR) model capable of recognizing entire pages of handwritten or printed text without requiring image segmentation.
* The proposed model architecture addresses the challenge of large-scale handwritten text recognition which is particularly useful for digitizing extensive handwritten labels on herbarium specimens.
* By presenting an approach that can handle full pages of text, this research could provide significant insights for developing robust OCR models to accurately recognize and transcribe handwritten labels on herbarium specimens, aiding in the digital preservation of valuable biodiversity information




## Data Requirements

The data that is fed into the model we develop needs to contain enough handwritten labels for the specimen along with their true labels. It will be important to hand divide the existing datasets to make sure there are enough images with handwritten labels in the training, testing and validation set. The handwritten labels we are looking to extract include: the curator's name, their institution, the species and genus, and the date the specimen was collected.

## Performance Criteria

The model we build is able to, at a reasonable percentage, locate where the handwritten labels are and from that, extract information. Our model should be robust enough to work for labels in varying locations as well as labels with a variety of handwriting styles. It should also be able to filter out the irrelevant text that is not necessary in labeling the specimen.  

## ML approach and reasoning

1. Data Preparation:
* Dataset Creation: Given the nature of the handwritten labels on the herbarium specimens, creating a robust dataset is crucial. Leveraging existing labeled data and augmenting it with additional manually labeled data to cover a wide variety of handwriting styles and specimen label formats is essential.
* Data Augmentation: Augment the dataset by simulating various real-world conditions such as different lighting conditions, orientations, and obstructions. This will ensure the model is robust to different scenarios.
* Data Splitting: Divide the data into training, validation, and test sets to ensure the model can generalize well to unseen data.
2. Model Architecture:
* Adopting Transformer-based OCR: Leverage the TrOCR approach which has shown promise in handling a variety of OCR tasks. The transformer models' ability to handle sequential data makes them apt for this task.
* Character Region Awareness: Utilize insights from the CRAFT paper to focus on character-level regions within the text for better accuracy, especially in recognizing irregular or curved text.
* Multi-Model Architecture: Consider employing a multi-model architecture where the primary OCR model is supplemented with a secondary model for image classification to recognize plant species, which could provide contextual cues for improving OCR accuracy.
3. Training:
* Transfer Learning: Use pre-trained models as a starting point and fine-tune them on the herbarium dataset. This can significantly speed up the training process and potentially lead to better performance.
* Multi-Task Learning: If feasible, design the network to perform both OCR and image classification simultaneously, sharing features between the tasks to improve performance.
4. Evaluation:
* Custom Evaluation Metrics: Develop evaluation metrics that focus on character-level accuracy, as well as word-level accuracy, to thoroughly evaluate the model's performance.
* Benchmarking: Compare the developed model against existing OCR models and methodologies using the same dataset to ensure it meets or exceeds the state-of-the-art performance.
5. Optimization and Testing:
* Hyperparameter Tuning: Conduct extensive hyperparameter tuning to optimize the model's performance.
* obustness Testing: Test the model against a variety of label configurations, handwriting styles, and image quality scenarios to ensure its robustness.
6. Integration and Deployment:
* Pipeline Integration: Integrate the developed OCR model within the existing ML-Herbarium pipeline ensuring seamless interaction between the OCR, image classification, and phenology identification modules.
* Deployment: Utilize platforms like TorchServe or Hugging Face Spaces for deploying the model, ensuring it can be easily accessed and utilized by other researchers and practitioners.
7. Feedback Loop:
* Continuous Improvement: Establish a feedback loop where the model's predictions can be corrected by domain experts, and this corrected data can be used to further train and improve the model over time.
8. Documentation:
* Ensure comprehensive documentation of the model architecture, training processes, and evaluation results to enable reproducibility and further research.
9. Future Exploration:
* Exploration of Advanced Techniques: Explore recent advancements in OCR and NLP like few-shot learning, unsupervised learning, or semi-supervised learning to continuously improve the OCR model's performance.

This structured approach aims to tackle the challenge of handwritten text recognition in a systematic and thorough manner, ensuring the model developed is robust, accurate, and well-integrated within the existing ML-Herbarium project pipeline. By embracing advanced OCR techniques, leveraging transfer learning, and ensuring a strong evaluation framework, this approach seeks to significantly contribute to the digitization and analysis of valuable herbarium specimen data.




## Deliverable Artifacts 
Our model, enough documentation for another team to work with our model, the research we have done to develop this model and what the next steps (if there are any) would be.


### References

1. CRAFT (text detection): https://arxiv.org/abs/1904.01941
2. TrOCR: https://arxiv.org/abs/2109.10282
3. "What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis," [10.1109/ICCV.2019.00481](https://doi.org/10.1109/ICCV.2019.00481)
4. Handwriting Transformers: https://ar5iv.labs.arxiv.org/html/2104.03964
5. Full Page Handwriting Recognition via Image to Sequence: https://arxiv.org/pdf/2103.06450.pdf 
6. Kubeflow: https://www.kubeflow.org/docs/
7. Hugging Face Spaces: https://huggingface.co/docs/hub/spaces
8. TorchServe: https://github.com/pytorch/serve
9. [The herbarium of the future](https://www.cell.com/trends/ecology-evolution/fulltext/S0169-5347(22)00295-6)
10. [Harvard University Herbaria](https://huh.harvard.edu/mission)

