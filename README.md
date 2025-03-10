# Tumor Detection and Segmentation using Deep Learning

This repository contains Python code for tumor detection and segmentation using deep learning techniques. The project is structured to handle both classification and segmentation tasks using a custom dataset.

## Project Structure

The project includes the following key components:

1. **Model Initialization**: Scripts to initialize models for training.
   - `model_initialisation.py`: Initialization code for the classification model.
   - `model_initialisation_seg.py`: Initialization code for the segmentation model.

2. **Model Training**: Scripts to train the models.
   - `model_training.py`: Training code for the classification model.
   - `model_training_seg.py`: Training code for the segmentation model.

3. **Dataset Handling**: Custom dataset class and utility functions.
   - `data_lading.py`: Custom dataset class for loading and preprocessing images and masks.

## Dataset

The dataset used in this project is the "Dataset_BUSI_with_GT" from Kaggle and is expected to be structured with images categorized into `benign`, `malignant`, and `normal` folders. Masks for segmentation should follow the naming convention `image_name_mask.png`.

## CNN Architectures

### Classification Model
- **Architecture**: The classification model is a Convolutional Neural Network (CNN) designed to classify breast ultrasound images into three categories: benign, malignant, and normal.
- **Layers**: The model consists of multiple convolutional layers followed by max-pooling layers to extract spatial features. Fully connected layers at the end perform the classification task.
- **Activation Functions**: ReLU activation functions are used in the convolutional layers to introduce non-linearity, and a softmax function is used in the output layer to produce probability distributions over the classes.

### Segmentation Model
- **Architecture**: The segmentation model is based on a U-Net architecture, which is effective for biomedical image segmentation tasks.
- **Layers**: The model includes an encoder path to capture context and a symmetric decoder path that enables precise localization. Skip connections are used to concatenate feature maps from the encoder to the decoder, preserving spatial information.
- **Activation Functions**: Similar to the classification model, ReLU activation functions are used in the convolutional layers. The output layer uses a sigmoid activation function to produce binary segmentation masks.

## Usage

To use the code for training or evaluation, follow these steps:

1. **Setup**: Ensure you have the necessary libraries installed. You can install the required libraries using:
   ```bash
   pip install requirements.txt
