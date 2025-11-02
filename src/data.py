import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(seed=42):
    """
    Loading white wine dataset, adding binary classification column,
    and splitting into train/validation/test sets.
    """
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        sep=";"
    )

    # Adding binary target for classification. 1 if >=7, 0 otherwise
    df["high_quality"] = (df["quality"] >= 7).astype(int)

    X = df.drop(columns=["quality", "high_quality"])
    y_class = df["high_quality"]
    y_reg = df["quality"]

    # Splitting into train/val/test (70/15/15) for classification and regression separately

    # Step 1: Splitting original data into 70% training and 30% temporary
    # 'stratify=y_class' ensures class balance (same proportion of high/low quality wines)
    X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
        X, y_class, y_reg, test_size=0.3, random_state=seed, stratify=y_class
    )

    # Step 2: Splitting the 30% temporary set equally into 15% validation and 15% test
    # We again stratify to preserve class balance in both validation and test sets
    X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
        X_temp, y_class_temp, y_reg_temp, test_size=0.5, random_state=seed, stratify=y_class_temp
    )

    return (X_train, X_val, X_test,
            y_class_train, y_class_val, y_class_test,
            y_reg_train, y_reg_val, y_reg_test)
