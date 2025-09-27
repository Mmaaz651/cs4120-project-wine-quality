from sklearn.metrics import confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.hist(residuals, bins=30)
    plt.show()
