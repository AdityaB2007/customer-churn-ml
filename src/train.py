import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_and_clean_data
from src.preprocessing import build_preprocessor
from src.evaluate import evaluate_model

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

def prepare_data(df):
    X = df.drop(
        columns=["Churn", "customerID"]
    )

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

def build_models():
    logistic_model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000
                )
            )
        ]
    )

    random_forest_model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42
                )
            )
        ]
    )

    return {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model
    }

def train_models():
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_and_clean_data()

    X_train, X_test, y_train, y_test = prepare_data(df)

    models = build_models()

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")

        model.fit(
            X_train,
            y_train
        )

        results[name] = evaluate_model(
            model,
            X_test,
            y_test
        )

    results_df = pd.DataFrame(results).T

    print("\nModel Results:")
    print(results_df)

    joblib.dump(
        models["Logistic Regression"],
        MODEL_DIR / "logistic_regression.joblib"
    )

    joblib.dump(
        models["Random Forest"],
        MODEL_DIR / "random_forest.joblib"
    )

    best_model_name = results_df["F1 Score"].idxmax()

    best_model = models[best_model_name]

    joblib.dump(
        best_model,
        MODEL_DIR / "churn_model.joblib"
    )

    print(
        f"\nBest model based on F1 Score: "
        f"{best_model_name}"
    )

    print(
        f"Saved best model to: "
        f"{MODEL_DIR / 'churn_model.joblib'}"
    )

    return models, results_df

if __name__ == "__main__":
    train_models()