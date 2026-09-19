import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay
)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    results = {
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "Precision": precision_score(
            y_test,
            predictions
        ),
        "Recall": recall_score(
            y_test,
            predictions
        ),
        "F1 Score": f1_score(
            y_test,
            predictions
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities
        )
    }

    return results

def print_classification_report(model, X_test, y_test):
    print(
        classification_report(
            y_test,
            model.predict(X_test)
        )
    )

def plot_confusion_matrix(model, X_test, y_test, title):
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.title(title)
    plt.show()

def plot_roc_curves(
    logistic_model,
    random_forest_model,
    X_test,
    y_test
):
    logistic_probabilities = (
        logistic_model.predict_proba(X_test)[:, 1]
    )

    random_forest_probabilities = (
        random_forest_model.predict_proba(X_test)[:, 1]
    )

    logistic_fpr, logistic_tpr, _ = roc_curve(
        y_test,
        logistic_probabilities
    )

    random_forest_fpr, random_forest_tpr, _ = roc_curve(
        y_test,
        random_forest_probabilities
    )

    logistic_auc = roc_auc_score(
        y_test,
        logistic_probabilities
    )

    random_forest_auc = roc_auc_score(
        y_test,
        random_forest_probabilities
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        logistic_fpr,
        logistic_tpr,
        label=f"Logistic Regression (AUC = {logistic_auc:.2f})"
    )

    plt.plot(
        random_forest_fpr,
        random_forest_tpr,
        label=f"Random Forest (AUC = {random_forest_auc:.2f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.show()