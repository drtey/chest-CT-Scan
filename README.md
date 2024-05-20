# Chest CT Scan Image Classification

## Project Objectives

The main objective of this project is to develop a deep learning model to classify chest CT scan images to detect the presence of lung diseases. This project has several sub-objectives:

1. **Data Preprocessing**: Implement preprocessing techniques to prepare the CT images for model training.
2. **Model Construction**: Use a convolutional neural network (CNN) architecture, specifically ResNet-50, to train the classification model.
3. **Optimization and Evaluation**: Optimize the model using advanced techniques and evaluate its performance on a test dataset.
4. **Implementation and Usage**: Develop an inference pipeline to use the model for classifying new CT images.

## Technologies Used

This project uses several technologies and Python libraries for data processing, model training, and result visualization. Some of the main technologies used are:

- **Programming Language**: Python
- **Environments and Tools**: Jupyter Notebook
- **Deep Learning Libraries**:
  - PyTorch
  - Torchvision
- **Image Processing Libraries**:
  - OpenCV
  - Pillow
- **Data Manipulation and Analysis Libraries**:
  - NumPy
  - Pandas
- **Visualization Libraries**:
  - Matplotlib
  - Seaborn

## Dataset

The dataset used in this project is the [Chest CT Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) available on Kaggle. This dataset contains chest CT scan images categorized into different classes based on the detected lung condition.

### Dataset Description

- **Number of Images**: 1000+ CT images
- **Categories**: Various categories representing different lung conditions
- **Format**: Images are in JPEG/PNG format
- **Resolution**: Variable

## Techniques Used

### Data Preprocessing

Before training the model, the images undergo a preprocessing pipeline that includes:

- **Resizing**: Adjusting images to a uniform size suitable for model input.
- **Normalization**: Scaling pixel values to improve training efficiency.
- **Data Augmentation**: Applying random transformations such as rotations, translations, and brightness changes to increase the variety of the training dataset and improve model generalization.

### Model Construction

The main model used in this project is an optimized ResNet-50. Key steps include:

1. **Loading Pretrained Model**: Using a ResNet-50 model pretrained on ImageNet as a starting point.
2. **Modifying the Output Layer**: Adjusting the output layer to match the number of classes in our dataset.
3. **Defining Loss Function and Optimizer**: Using cross-entropy as the loss function and Adam as the optimizer.

### Optimization and Evaluation

To optimize and evaluate the model, the following techniques were used:

- **Training and Validation**: Splitting the dataset into training and validation sets to monitor model performance during training.
- **Evaluation Metrics**: Using accuracy, recall, F1-score, and confusion matrix to assess model performance.
- **Regularization Techniques**: Implementing dropout and early stopping to prevent overfitting.

### Implementation and Usage

The developed inference pipeline allows loading new CT images and classifying them using the trained model. The pipeline includes:

- **Model Loading**: Functions to load the trained model from storage.
- **Preprocessing New Images**: Applying the same preprocessing techniques to new images.
- **Prediction**: Generating and visualizing model predictions.

## Installation

To install all the necessary dependencies, use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt