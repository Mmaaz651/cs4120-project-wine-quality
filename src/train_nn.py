"""
CS-4120 Final Project – Wine Quality (White Wine)
Neural Network Training & Evaluation (Keras + MLflow)
All plots are saved in notebooks/.
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, mean_squared_error
)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.tree import DecisionTreeClassifier

import os
os.chdir("..")
os.makedirs("notebooks", exist_ok=True)


# ------------------------------
# 1. Load and preprocess data
# ------------------------------
df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        sep=";"
    )
df["high_quality"] = (df["quality"] >= 7).astype(int)

X = df.drop(columns=["quality", "high_quality"])
y_class = df["high_quality"]
y_reg = df["quality"]

X_train, X_temp, y_train_c, y_temp_c, y_train_r, y_temp_r = train_test_split(
    X, y_class, y_reg, test_size=0.3, random_state=42
)
X_val, X_test, y_val_c, y_test_c, y_val_r, y_test_r = train_test_split(
    X_temp, y_temp_c, y_temp_r, test_size=0.5, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ------------------------------
# 2. Define NN models
# ------------------------------
def build_classification_model(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_regression_model(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


# ------------------------------
# 3. Train models + plot results
# ------------------------------
mlflow.set_experiment("WineQuality_Final_Keras")

# --- Classification ---
with mlflow.start_run(run_name="Classification_NN"):
    clf = build_classification_model(X_train.shape[1])
    early_stop = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
    hist_c = clf.fit(
        X_train, y_train_c,
        validation_data=(X_val, y_val_c),
        epochs=50, batch_size=32, callbacks=[early_stop], verbose=1
    )

    # Predictions
    y_pred_prob = clf.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test_c, y_pred)
    f1 = f1_score(y_test_c, y_pred)
    roc = roc_auc_score(y_test_c, y_pred_prob)

    mlflow.log_metrics({"Accuracy": acc, "F1": f1, "ROC_AUC": roc})
    mlflow.keras.log_model(clf, "Classification_NN")

    # --- Plot 1: Learning curve ---
    plt.plot(hist_c.history["loss"], label="Train Loss")
    plt.plot(hist_c.history["val_loss"], label="Val Loss")
    plt.title("Learning Curve – Classification NN")
    plt.xlabel("Epochs")
    plt.ylabel("Binary Crossentropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("notebooks/learning_curve_class.png", bbox_inches="tight")
    plt.close()

    # --- Plot 3: Confusion matrix ---
    cm = confusion_matrix(y_test_c, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix – Classification NN")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("notebooks/confusion_matrix_nn.png", bbox_inches="tight")
    plt.close()

# --- Regression ---
with mlflow.start_run(run_name="Regression_NN"):
    reg = build_regression_model(X_train.shape[1])
    early_stop = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
    hist_r = reg.fit(
        X_train, y_train_r,
        validation_data=(X_val, y_val_r),
        epochs=50, batch_size=32, callbacks=[early_stop], verbose=1
    )

    preds = reg.predict(X_test).ravel()
    mae = mean_absolute_error(y_test_r, preds)
    rmse = np.sqrt(mean_squared_error(y_test_r, preds))

    mlflow.log_metrics({"MAE": mae, "RMSE": rmse})
    mlflow.keras.log_model(reg, "Regression_NN")

    # --- Plot 2: Learning curve ---
    plt.plot(hist_r.history["loss"], label="Train Loss")
    plt.plot(hist_r.history["val_loss"], label="Val Loss")
    plt.title("Learning Curve – Regression NN")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("notebooks/learning_curve_reg.png", bbox_inches="tight")
    plt.close()

    # --- Plot 4: Residuals vs Predicted ---
    plt.scatter(preds, preds - y_test_r, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Predicted – Regression NN")
    plt.xlabel("Predicted Quality")
    plt.ylabel("Residuals (Pred - True)")
    plt.tight_layout()
    plt.savefig("notebooks/residuals_nn.png", bbox_inches="tight")
    plt.close()

# ------------------------------
# 4. Plot 5 – Feature Importance
# ------------------------------
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X, y_class)
feat_imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(6, 5))
feat_imp.plot(kind="barh")
plt.title("Feature Importance – Decision Tree (Reference)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("notebooks/feature_importance.png", bbox_inches="tight")
plt.close()

# ------------------------------
# 5. Final metrics summary
# ------------------------------
print(f"\n Classification NN → Accuracy={acc:.3f}, F1={f1:.3f}, ROC-AUC={roc:.3f}")
print(f" Regression NN → MAE={mae:.3f}, RMSE={rmse:.3f}")
print("All plots saved in notebooks/. MLflow run logged successfully.")
