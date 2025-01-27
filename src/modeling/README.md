# U-Net Model for Binary Classification of TB Bacilli

This repository contains a Python implementation of a **U-Net model architecture** tailored for binary classification tasks, specifically for detecting the presence or absence of TB bacilli in images. The model is implemented using TensorFlow/Keras and includes the ability to handle imbalanced datasets.

## Model Overview

The U-Net model has two main components:
- **Contracting Path (Encoder):** Captures features at multiple levels of abstraction by progressively down-sampling input images.
- **Expanding Path (Decoder):** Uses up-sampling and skip connections to provide precise localization while combining high-level and low-level features.

## Features

- **Binary classification:** Predicts the presence or absence of TB bacilli.
- **Dropout layers:** Reduce overfitting, ensuring robust performance.
- **Skip connections:** Enable better feature retention for improved accuracy.
- **Fully connected classification head:** Converts the extracted spatial features into binary classification probabilities.
- **Training flexibility:** Allows integration with custom datasets, data augmentation, and class balancing.

---

## Architecture Details

### 1. Contracting Path (Encoder)
- Consists of convolutional layers (3x3 kernels) followed by **ReLU activation** and **2x2 max-pooling**.
- Doubles the number of feature channels at each down-sampling step to capture complex patterns.

### 2. Bottleneck
- Located at the deepest part of the U-Net.
- Two convolutional layers with the highest feature dimensions.
- Includes dropout layers for regularization.

### 3. Expanding Path (Decoder)
- Each up-sampling step includes:
  - **Transposed convolution (2x2 kernel):** Restores spatial resolution.
  - **Convolutional layers (3x3 kernel):** Further refine features.
  - **Skip connections:** Concatenate features from the encoder for better localization.

### 4. Classification Head
- The bottleneck features are flattened and passed through fully connected dense layers.
- Output: A single neuron with a **sigmoid activation function** to classify images as 0 or 1.

---

## Training and Evaluation

### Training Configuration
- **Loss function:** Binary cross-entropy
- **Optimizer:** Adam
- **Metrics:** Accuracy
- **Data Augmentation:** (Optional) Use techniques like flipping, rotation, or zooming to increase training data diversity.

### Example Training Workflow
```python
# Model compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(
    train_generator,  # Replace with actual training data generator
    steps_per_epoch=len(train_images) // batch_size,  # Replace with train dataset size
    epochs=10,
    validation_data=(test_images, test_labels),  # Replace with validation dataset
    class_weight=class_weights  # Optional: Address class imbalance
)
