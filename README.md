# Machine-Learning
This repository houses various resources used in the capstone project for Bangkit Machine Learning. The project aims to develop machine learning models for our application, specifically image classifiers for Gym Equipment.

## Architecture
The model utilizes Convolutional Neural Network (CNN) technology, a robust machine learning approach. This architecture allows the model to efficiently analyze and identify patterns in input data, making it ideal for tasks like image classification. Additionally, we use transfer learning with DenseNet to enhance the model's performance.

![image](https://github.com/user-attachments/assets/5360a61b-eef9-4abb-95ad-7354033f0819)

## Datasets
We are using the following datasets:
* Original Dataset: Gym Equipment Image Dataset from Kaggle, which has been cleaned and expanded using web scraping.
  * Kaggle: https://www.kaggle.com/datasets/rifqilukmansyah381/gym-equipment-image
* Cleaned Dataset: Cleaned Dataset on Google Drive.
  * Google Drive: https://drive.google.com/drive/folders/1VTVt1x4-Oo_gg9wWsNRGpzUzr48ubCut?usp=sharing

## Models
### Model Overview
Utilizes a CNN architecture for accurate image classification:
- Convolutional layer with 3 convulational layers to extract features:
  - First layer: 64 filters, kernel size 3x3, ReLU activation, stride 1.
  - Second layer: 64 filters, similar settings.
  - Third layer: 128 filters, similar settings.
- Adds 3 max pooling layers to downsample feature maps:
- First two use pool size 2x2.
  - Third uses pool size 1x1 (minimal downsampling).
  - Dense layer: Softmax activation for multi-class classification, with x classes.
- Dropout layer with a rate of 0.5 to reduce overfitting.
- Batch Normalization normalizes activations after dropout for stable and faster training.
- Global Average Pooling replaces fully connected layers for spatial dimension reduction.
- Final dense layer with class_count units and softmax activation for classification.
### Data Processing
- 7 gym equipment classes, including bench press, dumbel, 
- Data augmentation using TensorFlow’s ImageDataGenerator for rotation, zoom, flipping, and more.
### Model Training
- Training on augmented datasets for x epochs with a batch size of x.
### Model Evaluation

### Model Saving and Conversion

## Requirements
To run the code, the following libraries are required
- TensorFlow
- Keras
- Matplotlib
- PIL
- os
- google.colab

