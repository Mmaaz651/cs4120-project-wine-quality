import mlflow
import mlflow.sklearn
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error
)
from sklearn.preprocessing import StandardScaler
from src.data import load_and_split_data
from src.evaluate import plot_confusion_matrix, plot_residuals
from src.utils import summarize_metrics

def run_baselines():
    os.makedirs("notebooks", exist_ok=True)
    mlflow.set_experiment("WineQuality_Baselines_Improved")

    # ---------------- Load and Prepare Data ---------------- #
    (X_train, X_val, X_test,
     y_class_train, y_class_val, y_class_test,
     y_reg_train, y_reg_val, y_reg_test) = load_and_split_data()

    # ---- Simple feature engineering (course-level) ---- #
    for df in [X_train, X_val, X_test]:
        if "alcohol" in df.columns and "density" in df.columns:
            df["alcohol_density_ratio"] = df["alcohol"] / df["density"]

    # ---- Standardization (for linear models) ---- #
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # -------- Classification Baselines -------- #
    class_results = {}
    classifiers = {
        # L2 regularization (default) for better generalization
        "LogisticRegression": LogisticRegression(
            penalty="l2", C=1.0, max_iter=1000, random_state=42
        ),
        # Moderate tree depth and split threshold for stable learning
        "DecisionTreeClassifier": DecisionTreeClassifier(
            max_depth=6, min_samples_split=4, random_state=42
        ),
    }

    for name, model in classifiers.items():
        with mlflow.start_run(run_name=name):
            # Logistic Regression on scaled data; tree on raw data
            if "Logistic" in name:
                model.fit(X_train_scaled, y_class_train)
                y_pred_test = model.predict(X_test_scaled)
                y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_class_train)
                y_pred_test = model.predict(X_test)
                y_prob_test = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_class_test, y_pred_test)
            f1 = f1_score(y_class_test, y_pred_test)
            roc = roc_auc_score(y_class_test, y_prob_test)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("roc_auc", roc)
            mlflow.sklearn.log_model(model, name)
            plot_confusion_matrix(y_class_test, y_pred_test, f"notebooks/{name}_confusion.png")

            class_results[name] = {"Accuracy": acc, "F1": f1, "ROC_AUC": roc}

    summarize_metrics("Classification", class_results)

    # -------- Regression Baselines -------- #
    reg_results = {}
    regressors = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(
            max_depth=6, min_samples_split=4, random_state=42
        ),
    }

    for name, model in regressors.items():
        with mlflow.start_run(run_name=name):
            if "Linear" in name:
                model.fit(X_train_scaled, y_reg_train)
                y_pred_test = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_reg_train)
                y_pred_test = model.predict(X_test)

            mae = mean_absolute_error(y_reg_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_test))

            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.sklearn.log_model(model, name)
            plot_residuals(y_reg_test, y_pred_test, f"notebooks/{name}_residuals.png")

            reg_results[name] = {"MAE": mae, "RMSE": rmse}

    summarize_metrics("Regression", reg_results)


if __name__ == "__main__":
    run_baselines()
