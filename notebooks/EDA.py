import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load Data ---
try:
    # Load directly from UCI repository
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        sep=";"
    )
except:
    # Fallback to local path if offline
    df = pd.read_csv("../data/winequality-white.csv", sep=";")

# Add binary target for classification
df["high_quality"] = (df["quality"] >= 7).astype(int)

# --- Plot 1: Target Distribution ---
plt.figure(figsize=(6, 4))
sns.countplot(x="high_quality", data=df, palette="Set2")
plt.title("Target Distribution: High vs Low Quality")
plt.xlabel("High Quality (1 = ≥ 7)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("target_distribution.png")  # Saved in notebooks/
plt.show()

# --- Plot 2: Correlation Heatmap ---
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")  # Saved in notebooks/
plt.show()
