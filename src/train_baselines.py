import mlflow
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from src.data import X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val
from src.features import scale_features

X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_val)

# Classification: Logistic Regression
mlflow.set_experiment("wine_classical_baselines")
with mlflow.start_run():
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_class_train)
    preds = clf.predict(X_val_scaled)
    acc = accuracy_score(y_class_val, preds)
    f1 = f1_score(y_class_val, preds)
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)
