# Machine-Learning
<p align="justify"> This repository houses various resources used in the capstone project for Bangkit Machine Learning. The project aims to develop machine learning models for our application, specifically image classifiers for Gym Equipment. </p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/8bc52373-8bb6-4e50-b760-aefcfbd5e34d" alt="App" style="width:40%; height:auto;">
</p>

## Main Feature: Image Classification using CNN Architecture
<p align="justify"> The model utilizes Convolutional Neural Network (CNN) technology, a robust machine learning approach. This architecture allows the model to efficiently analyze and identify patterns in input data, making it ideal for tasks like image classification. Additionally, we use transfer learning with DenseNet121 to enhance the model's performance. </p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/e1c7aaa3-1324-4587-bc3b-21a60c257c6f" alt="GymLensArch" style="width:60%; height:auto;">
</p>

### Datasets
We are using the following datasets:
* Original Dataset: Gym Equipment Image Dataset from Kaggle, which has been cleaned and expanded using web scraping.
  * Kaggle: https://www.kaggle.com/datasets/rifqilukmansyah381/gym-equipment-image
* Cleaned Dataset: Cleaned Dataset on Google Drive.
  * Google Drive: https://drive.google.com/drive/folders/1VTVt1x4-Oo_gg9wWsNRGpzUzr48ubCut?usp=sharing

### Models
#### Model Overview
Utilizes a CNN architecture for accurate image classification:
- Convolutional layer with 3 convulational layers to extract features.
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

#### Data Processing
<p align="justify"> The dataset used consists of images of gym equipment categorized into 12 classes: bench press, dip bar, dumbbells, elliptical machine, kettlebell, lateral pulldown, leg press machine, pull bar, recumbent bike, stair climber, Swiss ball, and treadmill.
To enhance the dataset's diversity and size, data augmentation is applied using TensorFlow's `ImageDataGenerator`. </p>

Key augmentation techniques include:
- Rescaling: Normalizes pixel values by scaling them to a range of 0 to 1.
- Rotation: Rotates images randomly up to 30 degrees.
- Width & Height Shifting: Translates images up to 20% of their width or height.
- Shearing: Applies shear transformations to images.
- Zooming: Randomly zooms images in or out by up to 20%.
- Horizontal Flipping: Randomly flips images horizontally for more variability. 
These techniques help prevent overfitting and improve model generalization.

#### Model Training
<p align="justify"> The model is trained on augmented datasets for 100 epochs with a batch size of 32. Transfer learning is utilized with the DenseNet121 architecture as the base model, leveraging its pre-trained features to enhance performance on our gym equipment classification task. The augmented dataset, enriched with diverse transformations, helps improve model generalization and robustness. </p>

#### Model Evaluation
<p align="justify"> The trained model is evaluated using the test dataset, providing accuracy and loss scores to measure its performance. Additionally, predictions are generated for the test dataset to showcase the model's classification results. The evaluation includes plotting training and validation loss and accuracy graphs to assess model performance over epochs. A confusion matrix is used to visualize how well the model distinguishes between the 12 gym equipment classes. Detailed predictions, including the predicted class and confidence scores, are logged for further analysis. The results and trained model are saved for reproducibility. </p>

#### Model Saving and Conversion
<p align="justify"> The trained model is stored in HDF5 format (`gym_model.h5`) for later use. To enable compatibility with Android applications, it is converted into TensorFlow Lite (TFLite) format using the TFLite Converter. The resulting TFLite file, `gym_model.tflite`, is optimized for deployment on devices with limited resources. Additionally, a Flask-based API is developed to serve the `gym_model.h5` file. This API is deployed on Google Cloud Run, leveraging Google Cloud Platform (GCP) for scalable and efficient model hosting. </p>

### Requirements
To run the code, the following libraries are required:
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- IPython
- OS

## Side Feature: Chatbot using Vertex AI
### Vertex AI Soft Fine-Tuning (SFT) 🤖  
<p align="justify">The chatbot is built using Vertex AI's Soft Fine-Tuning (SFT) feature, enabling domain-specific adaptation of the Gemini-1.0-pro model to gym-related tasks.</p>  

#### Fine-Tuning Configuration  
- **Base Model:** `gemini-1.0-pro-002`  
- **Adapter-Based Tuning:**  
  - Adapter size: 4  
  - Epochs: 4  
  - Learning rate multiplier: 1.0  
- **Datasets:**  
  - Training: [`data_train.jsonl`](https://drive.google.com/file/d/1zgPJLyIrZw1zHFr8JfcjKuhSdSB09dMV/view?usp=drive_link)
  - Validation: [`data_valid.jsonl`](https://drive.google.com/file/d/13sJ14Z9vd6DaDhfJvdYC1RH3635Yj6Eg/view?usp=sharing) 

#### Features and Benefits  
- Fine-tunes a generative AI model for gym-related tasks.  
- Outputs a tuned model and endpoint, ready for integration.  
- Scalable deployment on Vertex AI with minimal compute overhead.
