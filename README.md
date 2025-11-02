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
│   ├── train_baselines.py  # Runs baseline models and logs results
│   └── train_nn.py
├── models/                  # (Optional) model files (if saved manually)
├── mlruns/                  # MLflow experiment tracking (auto-created)
├── README.md                # Project documentation (this file)
├── requirements.txt         # Python dependencies
└── .gitignore
```

---

## Dataset

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

## Setup Instructions

### Step 1 – Create a Virtual Environment

It's recommended to isolate dependencies.

```
python -m venv venv
```

Activate the environment:

* **Mac/Linux**

```
source venv/bin/activate
```

* **Windows**

```
venv\Scripts\activate
```

### Step 2 – Install Dependencies

Install all required libraries:

```
pip install -r requirements.txt
```

### Step 3 – Verify Installation

Check that core packages import correctly:

```
python
```

then inside the shell:

```
import pandas, numpy, sklearn, mlflow
```

If no errors appear, type `exit()`.

---

## How to Run the Project

### Option 1 – Run via Terminal

Execute the baseline training script:

```
python -m src.train_baselines
```

This will:

* Train Logistic Regression and Decision Tree (classification)
* Train Linear Regression and Decision Tree Regressor (regression)
* Generate:
  * Confusion matrices (`*_confusion.png`)
  * Residuals vs Predicted plots (`*_residuals.png`)
* Print metric tables in the console
* Log all experiments in the `mlruns/` folder using MLflow

### Option 2 – Run Directly in PyCharm

If you are using PyCharm:

1. Open the project folder in PyCharm.
2. Go to File → Settings → Project → Python Interpreter and select your `venv` environment.
3. In the Project pane, open the file `src/train_baselines.py`.
4. Right-click anywhere inside the file and choose "Run 'train_baselines'" (green run button ▶️ in the top-right corner).
5. PyCharm will execute the script and display logs and metric tables in the run console.

This method automatically uses your environment and project paths — no command line needed.

### View MLflow Dashboard

Launch the experiment tracking UI:

```bash
mlflow ui
```

Then open your browser and go to:

 http://127.0.0.1:5000

Here you can visually compare all model runs, metrics, and artifacts.

---

## Outputs Summary

| Output Type | Description |
|-------------|-------------|
| EDA Notebook | Target distribution & correlation heatmap |
| Confusion Matrix | From best classification baseline |
| Residuals Plot | From best regression baseline |
| Table 1 | Classification metrics for all baselines |
| Table 2 | Regression metrics for all baselines |
| MLflow Logs | Automatically created in `mlruns/` |
