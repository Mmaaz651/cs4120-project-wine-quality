import mlflow
import mlflow.sklearn
import numpy as np
import os # To save plots
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error
)
from src.data import load_and_split_data
from src.features import scale_features
from src.evaluate import plot_confusion_matrix, plot_residuals
from src.utils import summarize_metrics

def run_baselines():
    os.makedirs("notebooks", exist_ok=True)  # or "plots", depending on repo
    mlflow.set_experiment("WineQuality_Baselines")

    (X_train, X_val, X_test,
     y_class_train, y_class_val, y_class_test,
     y_reg_train, y_reg_val, y_reg_test) = load_and_split_data()

    # Feature scaling
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)

    # -------- Classification Baselines -------- #
    class_results = {}
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42, max_depth=5)
    }

    for name, model in classifiers.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_class_train)
            y_pred_test = model.predict(X_test)

            acc = accuracy_score(y_class_test, y_pred_test)
            f1 = f1_score(y_class_test, y_pred_test)
            roc = roc_auc_score(y_class_test, model.predict_proba(X_test)[:, 1])

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("roc_auc", roc)
            mlflow.sklearn.log_model(model, name)
            plot_confusion_matrix(y_class_test, y_pred_test, f"{name}_confusion.png")

            class_results[name] = {"Accuracy": acc, "F1": f1, "ROC_AUC": roc}

    summarize_metrics("Classification", class_results)

    # -------- Regression Baselines -------- #
    reg_results = {}
    regressors = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42, max_depth=5)
    }

    for name, model in regressors.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_reg_train)
            y_pred_test = model.predict(X_test)

            mae = mean_absolute_error(y_reg_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_test))

            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.sklearn.log_model(model, name)
            plot_residuals(y_reg_test, y_pred_test, f"{name}_residuals.png")

            reg_results[name] = {"MAE": mae, "RMSE": rmse}

    summarize_metrics("Regression", reg_results)

if __name__ == "__main__":
    run_baselines()
