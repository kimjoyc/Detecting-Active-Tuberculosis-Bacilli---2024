"""
Data Preprocessing for Microscopy Images

This script loads microscopy images and their corresponding labels, preprocesses 
the images (resizing and normalization), and applies data augmentation to the training set. 
It also prepares the data for training and testing a deep learning model.

Dependencies:
- OpenCV for image processing
- NumPy and Pandas for data manipulation
- TensorFlow/Keras for data augmentation and training
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the CSV file containing file paths and labels
csv_file = '../datasets/tb-wellgen-smear/v1/tb-labels.csv'
labels_df = pd.read_csv(csv_file)

# Limit the dataset to the first 100 entries for quick testing (can be removed for full dataset)
labels_df = labels_df[:100]

# Perform train-test split
train_df, test_df = train_test_split(labels_df, test_size=0.25, random_state=42)

# Define image dimensions for resizing
# Adjust as needed for your model's input size
IMG_HEIGHT = 32
IMG_WIDTH = 32

def preprocess_image(file_path):
    """
    Preprocess a single image.

    Args:
        file_path (str): Path to the image file.

    Returns:
        np.ndarray: Resized and normalized image as a NumPy array.
    """
    # Load the image
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Image not found at path: {file_path}")

    # Resize the image to the specified dimensions
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))

    # Normalize pixel values to the range [0, 1]
    img = img.astype(np.float32) / 255.0
    return img

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rotation_range=20,       # Random rotation up to 20 degrees
    width_shift_range=0.2,   # Random horizontal shift up to 20% of image width
    height_shift_range=0.2,  # Random vertical shift up to 20% of image height
    shear_range=0.2,         # Shear transformation
    zoom_range=0.2,          # Random zoom within 20%
    horizontal_flip=True,    # Random horizontal flipping
    vertical_flip=True,      # Random vertical flipping
    fill_mode='nearest'      # Fill mode for pixels outside boundaries
)

# Preprocess training images
train_images = np.array([preprocess_image(file_path) for file_path in train_df['file_path']])
train_labels = train_df['tb_positive'].values

# Initialize the data generator for training
train_generator = train_datagen.flow(
    train_images,
    train_labels,
    batch_size=32,
    shuffle=True
)

# Preprocess testing images (no augmentation)
test_images = np.array([preprocess_image(file_path) for file_path in test_df['file_path']])
test_labels = test_df['tb_positive'].values

# Calculate class weights to handle class imbalance (optional)
num_classes = 2
class_weights = dict(
    zip(range(num_classes), (train_df['tb_positive'].value_counts() / len(train_df)).values)
)

# Summary output for verification
print(f"Training data: {len(train_images)} samples")
print(f"Testing data: {len(test_images)} samples")
print(f"Class weights: {class_weights}")
