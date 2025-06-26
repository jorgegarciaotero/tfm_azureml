# TFM - Azure Machine Learning Project

This repository contains the complete Azure Machine Learning workflows developed for my **Master's Thesis (TFM)**, including model training, evaluation, serialization, and deployment using the Azure ML ecosystem.

## 📌 Overview

The project explores different machine learning techniques (LSTM, XGBoost, SVM, Naive Bayes) to address a time-series prediction problem. It includes:

- Model experimentation and training notebooks
- Serialized models and scalers
- Pipelines for batch inference
- Azure ML-compatible configurations

---

## 📁 Repository Structure
/tfm_azureml
├── improved_lstm_savedmodel/   # Exported SavedModel format for final LSTM
├── pkls/                       # Serialized models and scalers (e.g., .pkl, .keras)
│   ├── features_improved_lstm.pkl
│   ├── scaler_improved_lstm.pkl
│   └── improved_lstm.keras
├── trainings/                  # Jupyter notebooks for model experimentation
│   ├── data_prep_svm_nbayes.ipynb
│   ├── experimentacion_lstm_con_fuga_datos.ipynb
│   ├── experimentacion_lstm_corregido.ipynb
│   └── experimentacion_xgboost.ipynb
├── batch_consumer.ipynb        # Notebook version of the batch consumer
├── batch_consumer.py           # Script for batch inference
├── batch_consumer_pipeline.py  # Batch inference integrated as Azure ML pipeline
├── model_training.ipynb        # Notebook for LSTM model training
├── submit_pipeline.py          # Submit job/pipeline to Azure ML
├── out.csv                     # Sample prediction output
└── README.md                   # Project documentation

---

## 🚀 Main Features

- ✅ Multiple model experiments (LSTM, XGBoost, SVM, Naive Bayes)
- ✅ Final model saved in both `.keras` and `SavedModel` format
- ✅ Preprocessing with custom scalers
- ✅ Python scripts for automation and batch prediction
- ✅ Compatible with Azure ML pipelines

---

## 🧪 Training Notebooks (in `/trainings`)

| Notebook                                 | Description |
|------------------------------------------|-------------|
| `data_prep_svm_nbayes.ipynb`             | Preprocessing for SVM & Naive Bayes |
| `experimentacion_lstm_con_fuga_datos.ipynb` | LSTM with data leakage scenario |
| `experimentacion_lstm_corregido.ipynb`   | Cleaned-up LSTM training |
| `experimentacion_xgboost.ipynb`          | XGBoost training and evaluation |

---

## 📦 Models and Artifacts (in `/pkls`)

- `improved_lstm.keras` - Final trained LSTM model
- `features_improved_lstm.pkl` - Feature pipeline
- `scaler_improved_lstm.pkl` - Scaler used in training

---

## ⚙️ How to Run

### Prerequisites

- Python 3.8+
- `azureml-core`, `pandas`, `tensorflow`, `scikit-learn`

```bash

## Example: Run batch inference
python batch_consumer.py 2025-05-15

## Example: Submit Azure ML pipeline
python submit_pipeline.py 2025-05-15





