#  Credit Card Fraud Detection – End-to-End Machine Learning Project

This project implements a full end-to-end fraud detection pipeline using the **Credit Card Fraud Dataset** (Kaggle).  
It follows a real-world ML workflow including data preparation, model training, hyperparameter tuning, deployment, and monitoring.

------------------------------------------------

##  Project Overview

Financial fraud detection is a real-world challenge due to:

- Extremely imbalanced data  
- Evolving fraud patterns  
- Need for high recall with low false alarms  

This project trains **three different models**:

### ✔ 1. Logistic Regression (Baseline Supervised Model)  
### ✔ 2. LightGBM (Best Performing Supervised Model)  
### ✔ 3. Isolation Forest (Unsupervised Anomaly Detector)

A **Streamlit web application** is built to interactively test transactions using any of the 3 trained models.

----------------------------------------------------

##  Project Structure

├── notebook.ipynb # Main notebook with full ML pipeline

├── app.py # Streamlit web app

├── model_lr.pkl # Saved Logistic Regression model

├── model_lgb.pkl # Saved LightGBM model

├── model_isoforest.pkl # Saved Isolation Forest model

├── scaler.pkl # Scaler for input preprocessing

├── README.md # Project documentation

-------------------------------------------------------------

##  Machine Learning Pipeline

### **1. Data Preparation**
- Scaling (`StandardScaler`)
- Train/test split
- SMOTE applied *inside* cross-validation (no data leakage)

### **2. Feature Engineering**  

- Isolation Forest uses normalized features

### **3. Models Trained**

| Model               | Type         | Purpose                       |
|---------------------|--------------|-------------------------------|
| Logistic Regression | Supervised   | Baseline benchmark            |
| LightGBM            | Supervised   | Best performing model         |
| Isolation Forest    | Unsupervised | Detect unknown fraud patterns |

--------------------------------------------------------------------

##  Evaluation Metrics

Because the dataset is highly imbalanced, we evaluate using:

- Precision  
- Recall  
- F1-score  
- AUC-ROC  
- Average Precision (AP)  
- Confusion Matrix  
- PR Curve Threshold Optimization  

LightGBM achieved the best results.

------------------------------------------------------------------------------

##  Model Deployment (Streamlit App)

A Streamlit app allows users to test transactions and choose any of the 3 models:

- Logistic Regression  
- LightGBM  
- Isolation Forest  

### Run locally

-----------------------------------------------------------------------------

### Features:
- Dynamic input sliders for PCA components  
- Fraud probability output  
- Warning message for predicted fraud  
- Isolation Forest shows anomaly score  

------------------------------------------------------------------------------

##  Monitoring & Maintenance

A monitoring section is included to simulate:

- Data drift detection (KS test)
- Prediction drift monitoring
- Retraining policies
- Threshold adjustment rules

----------------------------------------------------------------------------

##  Results Summary

| Model               | AUC      | Average Precision | Notes                      |
|---------------------|----------|-------------------|----------------------------|
| Logistic Regression |  Good    |  Stable           | Great baseline             |
| LightGBM            |  Best    |  Best             | Recommended                |
| Isolation Forest    |  Low     |  Low              | Use as anomaly signal      |



-------------------------------------------------------------------------------

##  Author
- **MALLOUK Nadia && MIKANE Fatima-Ezzhrae**  
- Data engineering Students, INPT

-------------------------------------------------------------------------------

##  Dataset
Credit Card Fraud dataset (Kaggle)
https://www.kaggle.com/mlg-ulb/creditcardfraud/version/3

------------------------------------------------------------------------------
