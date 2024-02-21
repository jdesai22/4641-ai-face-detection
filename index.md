---
layout: default
---

# AI Face Detection Project Proposal 


## Introduction/Background

1. Literature Review
2. Dataset description: dataset consisting of images of both real and AI-generated/fake faces, preferably labeled as being “real” or “fake” 
3. Dataset link: https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection 

## Problem Defintion

* Problem: With rapidly evolving AI, detection between real and fake content is an increasing problem. Image generation AIs like DallE, Midjourney, DreamStudio, and SORA can create very realistic human faces. At this pace, they will be able to generate images of people that are indistinguishable from real photographs or portraits.
* Motivation: We want to train an AI that will be able to classify images as real or AI-generated. This can be an effective countermeasure against the problem of fake images since an algorithm can pick up on the smallest of patterns or differences in the images. 


## Methods

#### Data Preprocessing

1. Adding random noise to images - helps prevent our model from overfitting; also helps improve our model’s accuracy 
2. Grayscale images - makes the images we use less complex to run algorithms on; also results in higher accuracy for image classification when compared to RGB/colored pictures. Our model will also most likely be judging whether a face is real or not based off its facial features, not necessarily color 
3. Histogram of oriented gradients - lets us not only detect edges in each image, but also lets us determine the direction of each edge in the image by extracting 


#### ML Algorithms

1. Support Vector Machine - supervised algorithm well-suited to handling data with a high dimensionality, also good for complex datasets 
2. k-NN - good supervised binary classification algorithm that can group several images based on their similarities 
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

Due to the sensitive nature of the data, it will be important to minimize False Positives, where our model might classify an AI-generated image as a real one. This should take priority to maintain our model’s trust and ethical implications of passing a fake image as real. We will be able to explore this in our results using a Confusion Matrix.

## References

