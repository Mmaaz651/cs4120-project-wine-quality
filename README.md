# CS-4120 Machine Learning Project — Wine Quality (White Wine)

### Team: Wine Quality — G1
**Members:**  
- Muhammad Maaz  
- Mubashir Ahmed Chaudhry  

### Dataset
- White Wine Quality Dataset (UCI Repository)  
- 4,898 samples, 11 numeric features + 1 quality label  

### Tasks
- **Classification:** Predict high (≥7) vs low (<7) quality.  
- **Regression:** Predict numerical quality score.
---

## Project Structure

```
project/
├── data/                    # Contains data README (no CSV committed)
│   └── README.md
├── notebooks/               # Exploratory Data Analysis
│   └── EDA.ipynb
├── src/                     # Core source code
│   ├── data.py              # Loads and splits dataset (train/val/test)
│   ├── features.py          # Scales numerical features
│   ├── evaluate.py          # Generates confusion matrix & residual plots
│   ├── utils.py             # Summarizes metrics in tables
│   └── train_baselines.py  # Runs baseline models and logs results
├── models/                  # (Optional) model files (if saved manually)
├── mlruns/                  # MLflow experiment tracking (auto-created)
├── README.md                # Project documentation (this file)
└── requirements.txt         # Python dependencies
```

---

## 📊 Dataset

The **White Wine Quality** dataset is automatically loaded from the UCI Machine Learning Repository.

```python
pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
    sep=";"
)
```

### Alternate Method (Offline Use)

If you prefer to work offline or the UCI website is unavailable:

1. Download the dataset manually from:  
   https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv

2. Place it inside the `data/` folder

3. Replace the above line in `src/data.py` with:

```python
pd.read_csv("data/winequality-white.csv", sep=";")
```
---

## ⚙Setup Instructions

### Step 1: Create a virtual environment

It's recommended to isolate dependencies.

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### Step 2: Install dependencies

Install everything listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 3: Verify installation

Run Python and ensure you can import all required libraries:

```python
import pandas, numpy, sklearn, mlflow
```

---

## How to Run the Project

### Run Baseline Models

To train models and log metrics:

```bash
python -m src.train_baselines
```

This will:

- Train Logistic Regression and Decision Tree (for classification)
- Train Linear Regression and Decision Tree Regressor (for regression)
- Generate:
  - Confusion matrices (`*_confusion.png`)
  - Residuals vs Predicted plots (`*_residuals.png`)
  - Metric tables printed in the console
- Log all experiments using MLflow under `mlruns/`

### View MLflow Dashboard

Launch the experiment tracking UI:

```bash
mlflow ui
```

Then open your browser and go to:

👉 http://127.0.0.1:5000

Here you can visually compare all model runs, metrics, and artifacts.

---

## 🧠 Outputs Summary

| Output Type | Description |
|-------------|-------------|
| EDA Notebook | Target distribution & correlation heatmap |
| Confusion Matrix | From best classification baseline |
| Residuals Plot | From best regression baseline |
| Table 1 | Classification metrics for all baselines |
| Table 2 | Regression metrics for all baselines |
| MLflow Logs | Automatically created in `mlruns/` |
