# Customer Churn Prediction System

An end-to-end machine learning project that predicts customer churn using customer demographics, account information, and service usage data. This project demonstrates a complete ML workflow from data preprocessing to model deployment via an application program interface (API).

---

## Overview

Customer churn prediction is a critical business problem where companies aim to preemptively identify customers likely to leave their service.

This project consists of a machine learning pipeline that:

- Processes and cleans raw customer data
- Handles both numerical and categorical features
- Trains and compares multiple classification models
- Evaluates performance using appropriate metrics
- Saves the best-performing model
- Serves predictions through a FastAPI-based API

---

## Key Features

- End-to-end ML pipeline using **scikit-learn**
- Feature preprocessing with **ColumnTransformer**
- Model comparison:
  - Logistic Regression
  - Random Forest
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
- Model persistence using **joblib**
- REST API using **FastAPI**
- Modular and production-style project structure

---

## Project Structure

```text
customer-churn-ml/
│
├── data/
├── models/
├── notebooks/
├── src/
├── app/
├── tests/
├── requirements.txt
├── README.md
└── run.py
```

---

## Objective

Given customer data such as tenure, contract type, and service usage, predict whether the customer is likely to churn.

---

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib
- FastAPI
- joblib
- pytest

---

## Models Used

This project compares:

- **Logistic Regression** (baseline model)
- **Random Forest Classifier** (nonlinear model)

The best model is selected based on F1 score.

---

## Evaluation Metrics

To properly evaluate churn prediction, multiple metrics are used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/customer-churn-ml.git
cd customer-churn-ml
```

---

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Train the model

```bash
python run.py
```

---

### 5. Run the API

```bash
uvicorn app.main:app --reload
```

---

## API Usage

### Endpoint

```
POST /predict
```

### Example Request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 1050.2
}
```

---

## Future Directions

- Hyperparameter tuning (GridSearchCV)
- Add gradient boosting models (XGBoost / LightGBM)
- Deploy API to cloud (Render / AWS)
- Add Streamlit frontend
- Dockerize the application
- Add CI/CD pipeline

---
