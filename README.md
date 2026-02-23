# A Unified Framework for Multi-Class Web and Network Attack Detection Using Classical and Deep Learning

This project presents a unified and reproducible framework for multi-class intrusion detection in web and network environments. It integrates classical machine learning algorithms with deep learning architectures under a consistent preprocessing and leakage-aware evaluation pipeline.

The framework supports automated dataset construction, fair model comparison, and export of prediction outputs compatible with Splunk for security monitoring and analysis.

# Overview

Modern web and network infrastructures are exposed to increasingly sophisticated attacks such as DDoS, SQL Injection, and Remote Code Execution. Traditional rule-based systems often fail to detect complex or evolving threats.

This framework evaluates:

Classical Machine Learning Models

Random Forest

Logistic Regression

Linear Support Vector Classifier (Linear SVC)

SGD Classifier

Deep Learning Models

Long Short-Term Memory (LSTM)

One-Dimensional Convolutional Neural Network (1D-CNN)

All models are trained and evaluated under consistent preprocessing and controlled data-splitting conditions to ensure reproducibility and fair comparison.

# Key Features

Automatic dataset building from user-provided CSV files

Unified preprocessing pipeline

Group-based train/test split (TCP Stream) to prevent flow-level leakage

Explicit stratified validation for deep learning models

Classical vs Deep Learning model comparison

Weighted evaluation metrics for multi-class datasets

Splunk-compatible prediction exports

Lightweight repository (no dataset stored)

# Project Structure

project_root/
│
├── data/
│   ├── raw/                # User-provided dataset CSV files
│   └── processed/          # Auto-generated merged dataset
│
├── results/                # Trained models and exported outputs
│
└── scripts/
    ├── config.py
    ├── build_dataset.py
    ├── train_model.py
    └── score_all_models_for_splunk.py
    
# Providing Your Dataset

Place your dataset CSV file(s) inside:

data/raw/

The system will automatically:

Detect all .csv files in data/raw/

Use a label column if it exists

If no label column exists, infer the label from the filename

Merge all files into:

data/processed/merged_attacks.csv
## Important Notes

For correct multi-class detection:

Either provide separate CSV files per attack type

Or include a label column in your dataset

No dataset is stored in this repository.

# Installation
## 1. Create Virtual Environment
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
## 2. Install Dependencies
pip install -r requirements.txt
# Training Models
python scripts/train_model.py

This step will:

Automatically build the merged dataset (if missing)

Train 4 classical ML models

Train LSTM and 1D-CNN models

Save trained models into results/

# Generate Splunk-Compatible Outputs
python scripts/score_all_models_for_splunk.py

This generates:

results/splunk_model_predictions.csv

results/splunk_model_summary.csv

These files can be directly ingested into Splunk dashboards for monitoring and performance analysis.

# Evaluation Strategy
## Group-Based Train/Test Split

The dataset is split using GroupShuffleSplit based on TCP Stream identifiers.
This prevents flow-level data leakage by ensuring that packets from the same TCP stream do not appear in both training and test sets.

This design leads to more realistic generalization estimates for intrusion detection tasks.

Deep Learning Validation Protocol

Deep learning models (LSTM and 1D-CNN) use an explicit stratified validation set derived from the training data.

Keras validation_split is intentionally not used to avoid unintended class imbalance or ordering effects.

## Evaluation Metrics

All models are evaluated on the held-out test set using:

Accuracy

Weighted Precision

Weighted Recall

Weighted F1-score

Weighted metrics are used to properly account for class imbalance in multi-class intrusion detection scenarios.

# Design Principles

Reproducibility

Modularity

Lightweight repository design

Deployment-aware evaluation

Leakage-aware experimental setup

Research-oriented experimentation

The integration with Splunk bridges the gap between offline experimentation and operational security monitoring environments.

# Future Work

Real-time traffic streaming

Hybrid ensemble models

Online / streaming inference

Advanced anomaly detection

SOC alert integration

## Authors

Saeid Khalilian, Bita Sabet, Talaya Farasat, Joachim Posegga

University of Passau
