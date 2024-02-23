# AI Face Detection Project Proposal 


## Introduction/Background

- Literature Review:
  - Research suggests some very obvious signs of a deep fake human face including smooth, detail-less skin with small, seemingly random defects and defects on smaller features such as individual hairs, eyelashes, and eyebrows [1]. We will attempt to include these indicators as we train our model.
  - We have also researched the issues caused by deep fake images, including fake news and misinformation [2], but this is covered more in the problems section.
  - To train our model it will also be very important to pre-process the images that we use [3], and the exact processes are covered in our methods section as well.
  - We will also have to use histograms to accurately identify the edges within the images during the pre-processing process [4], which is also covered in more depth in our methods section.
  - Finally, we have researched the ML algorithms that we can use to train our model [5], and they are covered in more detail in our methods section as well.
- Dataset Description: Dataset consisting of images of both real and AI-generated/fake faces, preferably labeled as being “real” or “fake” 
- Dataset Link: https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection 


## Problem Defintion

- Problem: With rapidly evolving AI, detection between real and fake content is an increasing problem. Image generation AIs like DallE, Midjourney, DreamStudio, and SORA can create very realistic human faces. At this pace, they will be able to generate images of people that are indistinguishable from real photographs or portraits, and can even lead to consequences with fake news and the larger spread of misinformation [2].
- Motivation: We want to train an AI that will be able to classify images as real or AI-generated. This can be an effective countermeasure against the problem of fake images since an algorithm can pick up on the smallest of patterns or differences in the images. 

## Methods

#### Data Preprocessing

1. Adding random noise to images - helps prevent our model from overfitting by adding variability to the training data which will improve our model’s accuracy. [3]
2. Grayscale images - makes the images we use less complex to run algorithms on; also results in higher accuracy for image classification when compared to RGB/colored pictures. Our model will also most likely be judging whether a face is real or not based off its facial features, not necessarily color [3]
3. Histogram of oriented gradients - lets us not only detect edges in each image, but also lets us determine the direction of each edge in the image by extracting [4]



#### ML Algorithms

1. Support Vector Machine - supervised algorithm well-suited to handling data with a high dimensionality, also good for complex datasets [5]
2. k-NN - good supervised binary classification algorithm that can group several images based on their similarities [5]
3. k-Means - unsupervised algorithm that is simple, efficient, and easy to understand that can be used for image classification 

## Results and Discussion

- Quantitative Metrics:
  - Accuracy 
  - F1-score 
  - AUC (Area Under the ROC Curve) 
- Project Goals:
  - Accuracy of 70% 
  - F1-score of 0.75
  - AUC of 0.80
- Expected Results:
  - Accuracy of 60%
  - F1-score of 0.60 
  - AUC of 0.70

Due to the sensitive nature of the data, it will be important to minimize False Positives, where our model might classify an AI-generated image as a real one. This should take priority to maintain our model’s trust and the ethical implications of passing a fake image as real. We can explore this in our results using a Confusion Matrix.

## References

1.  A. A. Maksutov, V. O. Morozov, A. A. Lavrenov and A. S. Smirnov,"Methods of Deepfake Detection Based on Machine Learning," in 2020 IEEE Conference of Russian Young Researchers in Electrical and Electronic Engineering (EIConRus), St. Petersburg and Moscow, Russia, 2020, pp. 408-411.
2.  F. Cocch, L. Baraldi, S. Poppi, M. Cornia, L. Baraldi, R. Cucchiara, “Unveiling the Impact of Image Transformations on Deepfake Detection: An Experimental Analysis,” Image Analysis and Processing – ICIAP 2023, 2023.
3.  T. Sree Sharmila, K. Ramar, T. Sree Renga Raja, “Impact of applying pre-processing techniques for improving classification accuracy,” SIViP, vol 8, pp 149–157, 2014. [Online]. Available: Springer Link, https://link.springer.com/. [Accessed Feb. 23, 2023].
4.  A. Sada, Y. Kinoshita, S. Shiota and H. Kiya, "Histogram-Based Image Pre-processing for Machine Learning," in 2018 IEEE 7th Global Conference on Consumer Electronics (GCCE), Nara, Japan, 2018, pp. 272-275.
5.  B. Mahesh, “Machine Learning Algorithms - A Review,” International Journal of Science and Research (IJSR), vol. 9, no. 1, January, 2020. [Online Serial]. Available: https://www.researchgate.net/ profile/Batta-Mahesh/publication/344717762_Machine_Learning_Algorithms_-A_Review/links/5f8b2365299bf1b53e2d243a/Machine-Learning-Algorithms-A-Review.pdf?eid=5082902844932096. [Accessed Feb. 23, 2023].

## Gantt Chart

[Gantt Chart Link](https://gtvault-my.sharepoint.com/:x:/g/personal/jdesai48_gatech_edu/ETR2aOKIeeBDqFoQlrdfBtcBIzWgndGGD5E0Me5BqIWWjA?e=7dxEgs)

## Contribution Table

| Name    | Proposal Contributions                                  |
|---------|----------------------------------------------------------|
| Ethan   | Data preprocessing methods and ML algorithms to be used  |
| Atharva | Problem description and solution motivation              |
| Naman   | Gantt Chart and Video Presentation                      |
| Jai     | Setting up Github repository and Github pages for the project. Project discussion. |
| Nikola  | Literature review and citations, assisting Naman with video presentation |