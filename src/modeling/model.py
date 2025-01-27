"""
This script defines a U-Net model architecture for binary classification tasks, particularly for the presence or absence of TB bacilli.

Model Architecture:
- The U-Net architecture consists of two main components: a contracting path (encoder) and an expanding path (decoder).
- Skip connections are used to concatenate feature maps from the encoder to the corresponding layers in the decoder, enabling the model to leverage both high-level and low-level features for better localization and segmentation accuracy.

Detailed Architecture:
1. **Contracting Path (Encoder):**
   - Composed of repeated applications of two 3x3 convolutional layers followed by a ReLU activation and a 2x2 max pooling operation.
   - At each down-sampling step, the number of feature channels is doubled, capturing increasingly complex features.

2. **Bottleneck:**
   - Located at the bottom of the U-Net, it consists of two convolutional layers with the highest number of feature channels.
   - A dropout layer is included to prevent overfitting and ensure robust feature extraction.

3. **Expanding Path (Decoder):**
   - Each up-sampling step consists of a 2x2 up-convolution (transposed convolution) followed by two 3x3 convolutional layers and ReLU activations.
   - Feature maps from the contracting path are concatenated with the up-sampled feature maps using skip connections, which provide spatial context for precise localization.

4. **Classification Head:**
   - Instead of the traditional segmentation output, the flattened bottleneck features are passed through fully connected layers.
   - These dense layers enable binary classification by summarizing the extracted spatial and feature information.

Features:
- Uses a binary cross-entropy loss function and sigmoid activation in the output layer.
- Incorporates a pre-trained backbone (optional) for feature extraction.
- Handles class imbalance using techniques like class weighting or oversampling during training.

Training and Evaluation:
- Includes an example of training with binary labels using a dataset split into training and validation sets.
- Monitors loss and validation accuracy during training.
- Allows hyperparameter tuning (e.g., learning rate, batch size, epochs).
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, concatenate, Flatten, Dense
)

def unet_model(input_shape=(128, 128, 3)):
    """
    Defines the U-Net model architecture for binary classification.

    Parameters:
        input_shape (tuple): Shape of the input images (height, width, channels).

    Returns:
        model (tf.keras.Model): Compiled U-Net model.
    """
    inputs = Input(input_shape)

    # Contracting path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottom of U-Net
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Flatten and fully connected layers for classification
    flatten = Flatten(input_shape=drop5.shape[1:])(drop5)
    dense1 = Dense(512, activation='relu')(flatten)
    dense2 = Dense(256, activation='relu')(dense1)
    outputs = Dense(1, activation='sigmoid')(dense2)

    # Model definition
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Initialize the model
model = unet_model()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Training configuration example (replace placeholders with actual data generators/datasets)
batch_size = 32
steps_per_epoch = len(train_images) // batch_size  # Assuming train_images is defined

history = model.fit(
    train_generator,  # Replace with actual training data generator
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=(test_images, test_labels),  # Replace with actual validation data
    class_weight=class_weights  # Replace with calculated class weights
)
