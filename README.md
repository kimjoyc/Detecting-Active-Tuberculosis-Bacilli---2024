# Detecting-Active-Tuberculosis-Bacilli---2024
[View Paper Inspiration PDF](Hackathon_FCN.pdf)

![Python](https://img.shields.io/badge/Python-100%25-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## Overview
This repository contains the code and models for the **Detecting Active Tuberculosis Bacilli - 2024** contest hosted by Nightingale Open Science and Wellgen Medical. The goal of the competition is to develop an algorithm capable of detecting the presence of TB bacilli in microscopy images of sputum samples. The dataset contains thousands of labeled images of TB smear microscopy, with the task being to predict whether a given image contains TB bacilli.

## Goal
The challenge is focused on developing machine learning models to predict the presence of **Mycobacterium tuberculosis** in patient samples based on microscopy images. The images are labeled with either a **positive** or **negative** presence of TB bacilli, with the dataset being imbalanced.

### Evaluation Metric
Submissions are evaluated based on **Precision-Recall Area Under Curve (PR-AUC)**, which is used to measure the model's ability to correctly identify positive and negative images, especially given the imbalanced dataset (5.3% of the training set contains TB-positive images).

## Dataset
The dataset consists of microscopy images from sputum samples, collected across Asia, specifically Taiwan, China, India, and Japan. These images were taken using a microscopic scanner and are labeled by medical technicians, with a small subset containing TB bacilli. 

### Dataset Details:
- **Training dataset**: 75,087 images.
  - **Positive labels (TB-Positive)**: 5.3% of images
  - **Negative labels (TB-Negative)**: 94.7% of images
- **Holdout dataset**: A random 25% subset, used for final validation.

The dataset is not publicly available due to patient confidentiality.

## Installation

### Prerequisites
- Python 3.8 or higher
- Recommended: Anaconda or Virtualenv for managing dependencies.

### Clone the repository:
```bash
git clone https://github.com/kimjoyc/Detecting-Active-Tuberculosis-Bacilli---2024.git
cd Detecting-Active-Tuberculosis-Bacilli---2024
