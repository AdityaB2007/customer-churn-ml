from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn_model.joblib"
)

def load_model():
    return joblib.load(MODEL_PATH)

def predict_churn(customer_data, model=None):
    if model is None:
        model = load_model()

    customer_df = pd.DataFrame(
        [customer_data]
    )

    prediction = model.predict(
        customer_df
    )[0]

    probability = model.predict_proba(
        customer_df
    )[0, 1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }

if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0
    }

    result = predict_churn(
        sample_customer
    )

    print(result)