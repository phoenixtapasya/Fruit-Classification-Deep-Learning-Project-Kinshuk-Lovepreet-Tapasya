# Fruit Classification using EfficientNetB0

This project implements a deep learning based **fruit image classification system** using **Transfer Learning with EfficientNetB0** in **TensorFlow / Keras**. The complete implementation is available in:

fruit-classification.ipynb


## Project Overview

The objective of this project is to classify different types of fruits from images using a Convolutional Neural Network (CNN). EfficientNetB0 is used as the backbone model for high accuracy and fast convergence.

The pipeline includes:
- Data loading  
- Image preprocessing and augmentation  
- Transfer learning  
- Fine-tuning  
- Model evaluation  


## Features

- Transfer learning with EfficientNetB0  
- Automatic data loading using `image_dataset_from_directory`  
- Image augmentation and normalization  
- Two-stage training:
  - Feature extraction (base model frozen)
  - Fine tuning (partial unfreezing)
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  
- High accuracy with low overfitting  


## Requirements

- Python 3.8+  
- TensorFlow 2.x  
- NumPy  
- Matplotlib (optional)  
- Jupyter Notebook / Kaggle  

Install dependencies:
pip install tensorflow numpy matplotlib

## How to Run

Open **fruit-classification.ipynb** in Kaggle or Jupyter Notebook  
Update the dataset path 
Run all cells sequentially  
The model will train in two stages and automatically save checkpoints  



