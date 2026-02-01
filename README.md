# Web Attack Detection with Deep Learning and Machine Learning (Splunk Integration)

This project implements multiple **machine learning and deep learning models** for detecting **web attacks (e.g., SQL Injection)** and integrates the results with **Splunk** for security monitoring and analysis.

The system is designed to support SOC-style workflows by providing:
- Trained ML/DL models
- Model evaluation and comparison
- Prediction outputs compatible with Splunk dashboards

---

## 📌 Features

- Detection of web attacks (SQL Injection datasets)
- Classical ML models:
  - Logistic Regression
  - Random Forest
  - Linear SVC
  - SGD Classifier
- Deep Learning models:
  - CNN
  - LSTM
- Model comparison and scoring
- Splunk-ready prediction & summary CSV outputs
- Modular and extensible Python codebase

---
Data Preparation

Merge and clean multiple SQL Injection datasets

Feature extraction and preprocessing

Model Training

Train multiple ML and DL models

Save trained models locally (ignored in Git)

Evaluation

Generate confusion matrices

Compare performance across models

Splunk Integration

Export predictions as CSV

Generate summary files for dashboards

▶️ How to Run
1️⃣ Create Virtual Environment
python -m venv venv310
source venv310/bin/activate   # Linux / macOS
venv310\Scripts\activate      # Windows
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Train Models
python scripts/train_model.py
4️⃣ Score All Models for Splunk
python scripts/score_all_models_for_splunk.py
📊 Outputs
The following files are generated (and ignored by Git):

Trained ML/DL models (.joblib, .keras)

Confusion matrix images

Splunk-compatible CSV files:

splunk_model_predictions.csv

splunk_model_summary.csv

🔐 Notes
Trained models and generated results are not committed to Git.

Only source code and configuration files are version-controlled.

The project is designed for academic research and SOC experimentation.

🚀 Future Improvements
Online / streaming prediction for Splunk

Advanced anomaly detection

Reduction of false positives

Real-time SOC alerting integration

👤 Author
Bita Sabet
Saeid Khalilian


