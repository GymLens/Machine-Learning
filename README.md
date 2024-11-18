# Machine-Learning
GymLens is a machine learning-based project designed to classify gym equipment from user-uploaded images. The system uses a Convolutional Neural Network (CNN) for image classification and provides comprehensive usage instructions for various types of gym equipment. By leveraging TensorFlow and transfer learning, the project aims to promote fitness awareness and improve accessibility for gym beginners.
## Architecture

## Datasets
Dataset : https://www.kaggle.com/datasets/rifqilukmansyah381/gym-equipment-image

## Models
### Model Overview
Utilizes a CNN architecture for accurate image classification:
- Convolutional layer: x filters, ReLU activation.
- Max pooling layer: Pool size of xxx.
- Dense layer: Softmax activation for multi-class classification, with x classes.
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

