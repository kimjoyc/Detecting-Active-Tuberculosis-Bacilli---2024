"""
Evaluate a trained deep learning model on a holdout dataset for detecting TB bacilli.
The script calculates metrics such as ROC-AUC and PR-AUC, visualizes model predictions,
and prepares the holdout dataset predictions for further analysis or submission.

Fine-tuning suggestions are included to improve the model's performance based on results.
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

def preprocess_image(file_path):
    """
    Preprocesses an image for model inference.
    
    Args:
        file_path (str): Path to the image file.
    
    Returns:
        np.ndarray: Preprocessed image ready for model input.
    """
    # Example preprocessing (to be replaced with the actual preprocessing logic)
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    img = load_img(file_path, target_size=(224, 224))  # Example size
    img = img_to_array(img) / 255.0  # Normalize pixel values
    return img

# Step 1: Evaluate the model on the test dataset
print("Evaluating the model on the test dataset...")
predictions = model.predict(test_images)

# Step 2: Calculate ROC curve
fpr, tpr, _ = roc_curve(test_labels, predictions)

# Step 3: Calculate ROC-AUC score
roc_auc = roc_auc_score(test_labels, predictions)

# Step 4: Plot ROC curve
plt.plot(fpr, tpr, marker='.', label=f"ROC-AUC: {roc_auc:.4f}")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
print(f"ROC-AUC: {roc_auc:.4f}")

# Step 5: Calculate Precision-Recall curve
precision, recall, _ = precision_recall_curve(test_labels, predictions)

# Step 6: Calculate PR-AUC score
pr_auc = auc(recall, precision)

# Step 7: Plot Precision-Recall curve
plt.plot(recall, precision, marker='.', label=f"PR-AUC: {pr_auc:.4f}")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
print(f"PR-AUC: {pr_auc:.4f}")

# Step 8: Load and preprocess holdout dataset
print("Processing the holdout dataset...")
csv_file = '/home/ngsci/datasets/tb-wellgen-smear/supplementary/contest/tb-holdout-manifest.csv'
holdout_df = pd.read_csv(csv_file)
holdout_df = holdout_df[:100]  # Limit to 100 samples for demonstration

# Preprocess holdout images without augmentation
holdout_images = np.array([preprocess_image(file_path) for file_path in holdout_df['file_path']])

# Step 9: Make predictions on the holdout dataset
predictions = model.predict(holdout_images)
holdout_df['probability'] = predictions  # Add prediction probabilities to the dataframe

# Step 10: Drop the file_path column for a cleaner output
holdout_df.drop(columns=['file_path'], inplace=True)

# Step 11: Save modified holdout DataFrame to a CSV file
modified_holdout_csv_path = "modified_holdout.csv"
holdout_df.to_csv(modified_holdout_csv_path, index=False)
print(f"Modified holdout predictions saved to {modified_holdout_csv_path}")

# Load the saved CSV for verification
mod_df = pd.read_csv("modified_holdout.csv")
print(mod_df.head())  # Display the first few rows of the holdout predictions

"""
Suggestions for fine-tuning:
1. Use transfer learning by leveraging pre-trained models on similar image datasets (e.g., medical imaging).
2. Experiment with hyperparameter optimization (e.g., learning rate, batch size, number of layers).
3. Add data augmentation techniques to increase the diversity of the training dataset.
4. Use advanced loss functions to handle imbalanced datasets (e.g., Focal Loss).
5. Investigate alternative architectures that may better suit the dataset.
"""
