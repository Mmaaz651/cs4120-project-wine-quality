# Wine Quality Dataset

This project uses the **White Wine Quality** dataset from the UCI Machine Learning Repository.

Dataset source:  
https://archive.ics.uci.edu/ml/datasets/wine+quality  

The dataset is **loaded directly from the UCI website** in the code using:

```python
pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=";")
```

### Alternate Method (Offline Use)

If you prefer to work offline or the UCI website is temporarily unavailable:

1. **Download the dataset manually** from the UCI page:  
   [https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv)

2. **Place the file** inside the `data/` folder of this project.

3. **Replace the loading line** above in the code with:

   ```python
   pd.read_csv("data/winequality-white.csv", sep=";")
