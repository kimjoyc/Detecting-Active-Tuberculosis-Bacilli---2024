# Data Preprocessing Script: `processing.py`

This script provides a data preprocessing pipeline for microscopy images. It handles image loading, resizing, normalization, data augmentation, and train-test splitting, preparing the data for deep learning model training.

---

## Features

- **Data Loading**:
  - Reads image file paths and labels from a CSV file.
  - Supports initial testing with a subset of the data (e.g., first 100 entries).

- **Preprocessing**:
  - Resizes images to a uniform size (default: `32x32` pixels).
  - Normalizes pixel values to a range of `[0, 1]`.

- **Data Augmentation**:
  - Random transformations such as:
    - Rotation (up to 20°)
    - Horizontal and vertical shifts
    - Shearing
    - Zooming
    - Flipping (horizontal and vertical)

- **Train-Test Split**:
  - Splits data into training and testing sets (default: 75% training, 25% testing).

- **Class Weight Calculation**:
  - Optionally calculates class weights for handling imbalanced datasets.

---

## Dependencies

Make sure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `opencv-python`
- `scikit-learn`
- `tensorflow`

You can install them with:

```bash
pip install numpy pandas opencv-python scikit-learn tensorflow
