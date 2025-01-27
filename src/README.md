# Detecting Active Tuberculosis Bacilli - 2024

This repository contains all necessary code, data processing scripts, and evaluation methodologies for detecting active tuberculosis bacilli using machine learning approaches.

---

## Repository Structure

### `/src/`
This is the main source code directory that includes the submodules for data processing, model training, and evaluation.

#### Subdirectories:
- `dataprocessing/`: Scripts and utilities for preprocessing and managing the datasets.
- `modeling/`: Machine learning and deep learning models for tuberculosis detection.
- `evaluation/`: Tools and scripts for evaluating the performance of trained models.

---

### File Descriptions

#### `dataprocessing/`
- Contains scripts for:
  - Data cleaning
  - Feature extraction
  - Preparing datasets for training and evaluation
- **README.md**: Describes preprocessing steps, datasets used, and any transformation logic applied.

#### `modeling/`
- Houses machine learning and deep learning models.
- Includes:
  - Model architecture files
  - Training scripts
  - Hyperparameter configurations
- **README.md**: Explains each model, the training process, and how to load pre-trained models if available.

#### `evaluation/`
- Tools for:
  - Computing performance metrics (e.g., accuracy, F1-score)
  - Visualizing results
  - Comparison between models
- **README.md**: Details evaluation criteria, metrics used, and how to interpret evaluation results.

---

## Getting Started

### Prerequisites
- Python 3.x
- Required libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `tensorflow` or `pytorch`
  - `matplotlib`

### Installation
Clone the repository:
```bash
git clone https://github.com/your-username/Detecting-Active-Tuberculosis-Bacilli---2024.git
