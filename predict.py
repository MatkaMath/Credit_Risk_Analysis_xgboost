import os
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, X_test_pca, y_test):
    # Ensure the 'models' folder exists
    if not os.path.exists("models/xgboost_model.pkl"):
        raise FileNotFoundError("Trained model not found! Train the model first.")

    # Load trained model and test data
    with open("models/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)

    X_test_pca = np.load("models/X_test_pca.npy")
    y_test = np.load("models/y_test.npy")
    
    # Make predictions
    y_pred = model.predict(X_test_pca)
    y_proba = model.predict_proba(X_test_pca)[:, 1]

    # Compute evaluation metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)

    # Print results
    print("ROC-AUC Score:", roc_auc)
    print(report)

    # Ensure the results folder exists
    os.makedirs("results", exist_ok=True)

    # Save classification report as CSV
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv("results/classification_report.csv", index=True)

    # Save classification report as TXT
    with open("results/classification_report.txt", "w") as f:
        f.write(f"ROC-AUC Score: {roc_auc:.4f}\n\n")
        f.write(report)
