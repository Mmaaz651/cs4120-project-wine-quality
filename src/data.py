import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('../data/winequality-white.csv', sep=';')

# Create classification label
df['high_quality'] = (df['quality'] >= 7).astype(int)

# Split features and targets
X = df.drop(['quality', 'high_quality'], axis=1)
y_class = df['high_quality']
y_reg = df['quality']

# Split train/val/test
X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
    X, y_class, y_reg, test_size=0.3, random_state=42)
X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
    X_temp, y_class_temp, y_reg_temp, test_size=0.5, random_state=42)
