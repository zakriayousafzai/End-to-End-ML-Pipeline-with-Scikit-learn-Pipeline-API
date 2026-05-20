# Developer Guide: End-to-End ML Pipeline with Scikit-learn

This guide is written for developers who know Python and want to understand this entire project deeply, including syntax and design choices.

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Key Concepts Explained](#key-concepts-explained)
3. [Project Structure](#project-structure)
4. [How the Data Flows](#how-the-data-flows)
5. [File-by-File Breakdown](#file-by-file-breakdown)
6. [Understanding the Libraries](#understanding-the-libraries)
7. [Training Process Explained](#training-process-explained)
8. [Evaluation Metrics Explained](#evaluation-metrics-explained)
9. [Beginner Path (30 Minutes)](#beginner-path-30-minutes)
10. [Customization Guide](#customization-guide)
11. [Troubleshooting](#troubleshooting)
12. [Glossary](#glossary)

## What This Project Does

This project builds a complete machine learning workflow to predict customer churn (whether a telecom customer will leave).

It does this in one reusable `scikit-learn` pipeline that includes:

- Data preprocessing (numeric imputation, scaling, categorical imputation, one-hot encoding)
- Model training (Logistic Regression and Random Forest)
- Hyperparameter search (`GridSearchCV`)
- Evaluation (`ROC-AUC`, `accuracy`, classification report)
- Exporting the best full pipeline to disk with `joblib`

Why this is important:

- You train once, save once, and later run predictions with the exact same preprocessing logic.
- This reduces training/serving skew (mismatches between train-time and predict-time transformations).

## Key Concepts Explained

### 1) Churn Prediction

- **Goal**: binary classification (`0` = No churn, `1` = Churn).
- **Target column**: `Churn` in the dataset.
- **Features**: customer demographic and account/service fields.

### 2) Pipeline

`Pipeline` in scikit-learn chains steps in a strict order.

In this project:

1. `preprocessor` transforms raw mixed-type data into model-ready numeric arrays.
2. `model` receives transformed data and learns patterns.

### 3) ColumnTransformer

`ColumnTransformer` lets you apply different transformations to different column groups.

- Numeric columns -> median imputation + scaling
- Categorical columns -> most-frequent imputation + one-hot encoding

### 4) GridSearchCV

`GridSearchCV` tries many combinations of model + hyperparameters and picks the best based on a score.

This project uses:

- `scoring="roc_auc"`
- `cv=5` (5-fold cross-validation)

#### What `cv=5` means (5-fold cross-validation)

`cv=5` means the training data is split into 5 equal parts called **folds**.

For each model/hyperparameter combination, scikit-learn trains and validates 5 times:

1. Train on folds 2,3,4,5 and validate on fold 1.
2. Train on folds 1,3,4,5 and validate on fold 2.
3. Train on folds 1,2,4,5 and validate on fold 3.
4. Train on folds 1,2,3,5 and validate on fold 4.
5. Train on folds 1,2,3,4 and validate on fold 5.

Then it averages the 5 validation scores. That average score is used to compare hyperparameter settings.

Tiny example with 10 rows (for intuition):

- Suppose rows are `[1,2,3,4,5,6,7,8,9,10]`.
- With 5 folds, each fold has 2 rows.
- Example fold split:
  - Fold 1: `[1,2]`
  - Fold 2: `[3,4]`
  - Fold 3: `[5,6]`
  - Fold 4: `[7,8]`
  - Fold 5: `[9,10]`
- Run 1: validate on `[1,2]`, train on all others.
- Run 2: validate on `[3,4]`, train on all others.
- ... until each fold has been validation once.

ASCII mini-diagram:

```text
Rows:   [1,2,3,4,5,6,7,8,9,10]
Folds:   F1  F1  F2  F2  F3  F3  F4  F4  F5  F5

Run 1:  [ V   V | T   T | T   T | T   T | T   T ]  validate=F1
Run 2:  [ T   T | V   V | T   T | T   T | T   T ]  validate=F2
Run 3:  [ T   T | T   T | V   V | T   T | T   T ]  validate=F3
Run 4:  [ T   T | T   T | T   T | V   V | T   T ]  validate=F4
Run 5:  [ T   T | T   T | T   T | T   T | V   V ]  validate=F5

Legend: V = validation fold, T = training folds
```

Why this helps:

- Less dependent on one lucky/unlucky split.
- Every row is used for validation exactly once.
- Gives a more stable estimate when selecting the best model.

### 5) Serialization with joblib

`joblib.dump` saves Python objects efficiently. Here it saves the full best pipeline (preprocessor + model), not just model weights.

## Project Structure

```text
End-to-End-ML-Pipeline-with-Scikit-learn-Pipeline-API/
  Dataset/
    Telco-Customer-Churn.csv
    About_Dataset.md
  docs/
    DEVELOPER_GUIDE.md
  models/
    best_pipeline.joblib
  reports/
    metrics.json
    classification_report.txt
  src/
    train.py
    predict.py
  README.md
  requirements.txt
  Task.md
```

## How the Data Flows

1. `src/train.py` reads `Dataset/Telco-Customer-Churn.csv`.
2. `load_data()` cleans columns:
   - `TotalCharges` converted to numeric (`errors="coerce"`)
   - `customerID` dropped
   - `Churn` mapped from `Yes/No` to `1/0`
3. Features/labels split (`X`, `y`).
4. Train/test split with stratification.
5. Preprocessing and model are wrapped into one `Pipeline`.
6. `GridSearchCV` tests logistic regression and random forest settings.
7. Best pipeline predicts on test set.
8. Metrics and report are generated.
9. Artifacts saved:
   - `models/best_pipeline.joblib`
   - `reports/metrics.json`
   - `reports/classification_report.txt`
10. `src/predict.py` loads saved pipeline and predicts on new sample rows.

### ASCII Flowchart

```text
START
  |
  v
Load CSV (Dataset/Telco-Customer-Churn.csv)
  |
  v
Clean Data
  - TotalCharges -> numeric
  - Drop customerID
  - Map Churn Yes/No -> 1/0
  |
  v
Split Features/Target (X, y)
  |
  v
Train/Test Split (stratified, 80/20)
  |
  v
Build Preprocessor
  |-- Numeric: median imputer -> scaler
  `-- Categorical: mode imputer -> one-hot encoder
  |
  v
Build Full Pipeline (preprocessor + model)
  |
  v
GridSearchCV
  |-- Logistic Regression grid
  `-- Random Forest grid
  |
  v
Select best_estimator_ by ROC-AUC
  |
  v
Evaluate on test set
  - ROC-AUC
  - Accuracy
  - Classification report
  |
  v
Save Outputs
  - models/best_pipeline.joblib
  - reports/metrics.json
  - reports/classification_report.txt
  |
  v
Predict on new rows with src/predict.py
  |
  v
END
```

### ColumnTransformer Branch Diagram (ASCII)

```text
Input DataFrame X
  |
  v
ColumnTransformer
  |
  +--> Branch 1: Numeric Columns (exclude object)
  |       |
  |       +--> SimpleImputer(strategy="median")
  |       |
  |       `--> StandardScaler()
  |
  `--> Branch 2: Categorical Columns (include object)
          |
          +--> SimpleImputer(strategy="most_frequent")
          |
          `--> OneHotEncoder(handle_unknown="ignore")

Merged transformed feature matrix
  |
  v
Model step (LogisticRegression or RandomForestClassifier)
```

## File-by-File Breakdown

This section explains each important file, then gives line-by-line explanation for code files.

---

### `README.md`

- Declares project purpose.
- Gives quick start commands (`pip install -r requirements.txt`, `python src/train.py`).
- Lists outputs and points readers to this developer guide.

### `requirements.txt`

Packages required:

- `pandas`: DataFrame operations and CSV loading.
- `numpy`: Numeric backend used by pandas/sklearn.
- `scikit-learn`: preprocessing, models, search, metrics.
- `joblib`: saving/loading trained pipeline.

### `Dataset/About_Dataset.md`

- Documents meaning of each dataset column.
- Defines `Churn` as target.

### `Task.md`

- Captures original project objectives: end-to-end reusable ML pipeline.

---

### `src/train.py` (line-by-line + syntax)

Below, each line is explained with both purpose and Python/scikit-learn syntax.

1. `from __future__ import annotations`
   - Enables postponed evaluation of type annotations.
   - Syntax: `from ... import ...` imports names from a module.
   - `__future__` toggles modern behavior in current Python versions.

2. *(blank line)*
   - Used for readability; no runtime effect.

3. `import json`
   - Imports standard library module for JSON serialization.

4. `from pathlib import Path`
   - Imports `Path` class for cross-platform path handling.

5. *(blank line)*

6. `import joblib`
   - Imports `joblib` for model persistence.

7. `import pandas as pd`
   - Imports pandas and aliases as `pd` (conventional alias).

8. `from sklearn.compose import ColumnTransformer`
   - Imports transformer that applies pipelines by column subset.

9. `from sklearn.ensemble import RandomForestClassifier`
   - Imports tree ensemble classifier.

10. `from sklearn.impute import SimpleImputer`
    - Imports missing-value imputer.

11. `from sklearn.linear_model import LogisticRegression`
    - Imports linear classifier for binary/multiclass tasks.

12. `from sklearn.metrics import accuracy_score, classification_report, roc_auc_score`
    - Imports evaluation functions.
    - Syntax: comma-separated imports in one statement.

13. `from sklearn.model_selection import GridSearchCV, train_test_split`
    - Imports data split utility and hyperparameter search tool.

14. `from sklearn.pipeline import Pipeline`
    - Imports `Pipeline` class.

15. `from sklearn.preprocessing import OneHotEncoder, StandardScaler`
    - Imports categorical and numeric preprocessors.

16-17. *(blank lines)*

18. `DATA_PATH = Path(__file__).resolve().parents[1] / "Dataset" / "Telco-Customer-Churn.csv"`
    - Defines absolute path to CSV file.
    - `__file__`: current script path.
    - `.resolve()`: absolute canonical path.
    - `.parents[1]`: project root from `src/`.
    - `/` operator on `Path` joins path components.

19. `MODEL_DIR = Path(__file__).resolve().parents[1] / "models"`
    - Directory where model artifact is saved.

20. `REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"`
    - Directory where metric/report files are saved.

21-22. *(blank lines)*

23. `def load_data() -> pd.DataFrame:`
    - Defines function with return type hint.
    - `-> pd.DataFrame` is a type annotation.

24. `df = pd.read_csv(DATA_PATH)`
    - Reads CSV into DataFrame.

25. `df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")`
    - Converts column to numeric.
    - `errors="coerce"` converts invalid strings to `NaN`.

26. `df = df.drop(columns=["customerID"])`
    - Removes non-informative identifier feature.
    - `columns=[...]` explicitly specifies columns axis.

27. `df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})`
    - Maps target labels from strings to integers.

28. `return df`
    - Returns cleaned DataFrame.

29-30. *(blank lines)*

31. `def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:`
    - Function receives feature DataFrame and returns `ColumnTransformer`.

32. `categorical_features = features.select_dtypes(include=["object"]).columns`
    - Finds object/string columns.
    - `.columns` returns Index of column names.

33. `numeric_features = features.select_dtypes(exclude=["object"]).columns`
    - Finds non-object columns (numbers).

34. *(blank line)*

35. `numeric_pipeline = Pipeline(`
    - Starts numeric preprocessing pipeline.

36. `steps=[`
    - `steps` is an ordered list of `(name, transformer)` tuples.

37. `("imputer", SimpleImputer(strategy="median")),`
    - Missing numeric values replaced by median.

38. `("scaler", StandardScaler()),`
    - Standardizes numeric features (zero mean, unit variance).

39. `]`
    - Ends list.

40. `)`
    - Ends `Pipeline(...)` call.

41. `categorical_pipeline = Pipeline(`
    - Starts categorical preprocessing pipeline.

42. `steps=[`

43. `("imputer", SimpleImputer(strategy="most_frequent")),`
    - Replaces missing category values with mode.

44. `("onehot", OneHotEncoder(handle_unknown="ignore")),`
    - Converts categories into binary indicator columns.
    - `handle_unknown="ignore"` avoids crash for unseen categories at inference.

45. `]`

46. `)`

47. *(blank line)*

48. `return ColumnTransformer(`
    - Returns combined transformer.

49. `transformers=[`
    - List of `(name, transformer, columns)` entries.

50. `("num", numeric_pipeline, numeric_features),`
    - Apply numeric pipeline to numeric columns.

51. `("cat", categorical_pipeline, categorical_features),`
    - Apply categorical pipeline to categorical columns.

52. `]`

53. `)`

54-55. *(blank lines)*

56. `def train_model() -> tuple[Pipeline, dict, str]:`
    - Returns trained pipeline, metrics dict, and text report.
    - `tuple[...]` is a parameterized type annotation.

57. `df = load_data()`
    - Loads cleaned dataset.

58. `X = df.drop(columns=["Churn"])`
    - Feature matrix.

59. `y = df["Churn"]`
    - Target vector.

60. *(blank line)*

61. `preprocessor = build_preprocessor(X)`
    - Builds preprocessing object from feature dtypes.

62. *(blank line)*

63. `X_train, X_test, y_train, y_test = train_test_split(`
    - Unpacks four outputs from split function.

64. `X, y, test_size=0.2, random_state=42, stratify=y`
    - 80/20 split, reproducible seed, class balance preserved.

65. `)`

66. *(blank line)*

67. `pipeline = Pipeline(`
    - Creates end-to-end processing + model object.

68. `steps=[`

69. `("preprocessor", preprocessor),`
    - First step transforms raw tabular data.

70. `("model", LogisticRegression(max_iter=1000)),`
    - Placeholder/default estimator; grid can replace it.

71. `]`

72. `)`

73. *(blank line)*

74. `param_grid = [`
    - Starts list of search spaces.

75. `{`
    - First dict: logistic regression space.

76. `"model": [LogisticRegression(max_iter=1000, solver="liblinear")],`
    - Explicit estimator object as candidate.

77. `"model__C": [0.1, 1.0, 10.0],`
    - Regularization strengths.
    - `model__C` uses double underscore syntax: step name + parameter name.

78. `"model__penalty": ["l1", "l2"],`
    - Try both L1 and L2 regularization.

79. `},`
    - Ends first search dict.

80. `{`
    - Second dict: random forest space.

81. `"model": [RandomForestClassifier(random_state=42)],`
    - Candidate model object.

82. `"model__n_estimators": [200, 500],`
    - Number of trees.

83. `"model__max_depth": [None, 10, 20],`
    - Tree depth limits.

84. `"model__min_samples_split": [2, 5],`
    - Minimum samples to split an internal node.

85. `},`

86. `]`

87. *(blank line)*

88. `search = GridSearchCV(`
    - Initializes exhaustive search object.

89. `pipeline,`
    - Base estimator is the full pipeline.

90. `param_grid=param_grid,`
    - Search space defined above.

91. `scoring="roc_auc",`
    - Objective metric for selecting best model.

92. `cv=5,`
    - 5-fold cross-validation.

93. `n_jobs=-1,`
    - Use all available CPU cores.

94. `verbose=1,`
    - Print progress logs.

95. `)`

96. `search.fit(X_train, y_train)`
    - Runs all grid combinations on training data.

97. *(blank line)*

98. `best_model = search.best_estimator_`
    - Best pipeline selected by CV ROC-AUC.

99. `y_pred = best_model.predict(X_test)`
    - Predicted classes for test data.

100. `y_proba = best_model.predict_proba(X_test)[:, 1]`
     - Positive-class probabilities.
     - `[:, 1]` means all rows, second column (class 1 probability).

101. *(blank line)*

102. `best_params = dict(search.best_params_)`
     - Copies best-params mapping into plain dict.

103. `model_param = best_params.pop("model", None)`
     - Removes raw model object from dict if present.
     - `pop(key, default)` avoids KeyError.

104. `if model_param is not None:`
     - Guard clause for optional value.

105. `best_params["model"] = model_param.__class__.__name__`
     - Replaces estimator object with readable class name.

106. *(blank line)*

107. `metrics = {`
     - Builds metrics dictionary.

108. `"best_params": best_params,`
     - Stores selected hyperparameters.

109. `"roc_auc": float(roc_auc_score(y_test, y_proba)),`
     - Converts NumPy scalar to Python `float` for JSON compatibility.

110. `"accuracy": float(accuracy_score(y_test, y_pred)),`
     - Accuracy metric as float.

111. `}`

112. `report = classification_report(y_test, y_pred)`
     - Text table with precision/recall/f1/support.

113. *(blank line)*

114. `return best_model, metrics, report`
     - Function output tuple.

115-116. *(blank lines)*

117. `def save_outputs(best_model: Pipeline, metrics: dict, report: str) -> None:`
     - Function to persist all artifacts.

118. `MODEL_DIR.mkdir(parents=True, exist_ok=True)`
     - Creates `models/`; no error if already exists.

119. `REPORT_DIR.mkdir(parents=True, exist_ok=True)`
     - Creates `reports/` similarly.

120. *(blank line)*

121. `joblib.dump(best_model, MODEL_DIR / "best_pipeline.joblib")`
     - Saves full trained pipeline object.

122. `(REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")`
     - Serializes metrics dict as pretty JSON.

123. `(REPORT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")`
     - Saves text classification report.

124-125. *(blank lines)*

126. `def main() -> None:`
     - Script entry function.

127. `best_model, metrics, report = train_model()`
     - Runs training and evaluation.

128. `save_outputs(best_model, metrics, report)`
     - Persists artifacts.

129. *(blank line)*

130. `print("Best parameters:", metrics["best_params"])`
     - Prints chosen hyperparameters.

131. `print("ROC-AUC:", metrics["roc_auc"])`
     - Prints ROC-AUC.

132. `print("Accuracy:", metrics["accuracy"])`
     - Prints accuracy.

133. `print(report)`
     - Prints classification report text.

134-135. *(blank lines)*

136. `if __name__ == "__main__":`
     - Python idiom: only run when file executed directly.

137. `main()`
     - Launches workflow.

---

### `src/predict.py` (line-by-line + syntax)

1. `import joblib`
   - Imports serialization tool to load trained artifact.

2. `import pandas as pd`
   - Imports pandas for DataFrame construction.

3. *(blank line)*

4. `pipeline = joblib.load("models/best_pipeline.joblib")`
   - Loads persisted end-to-end pipeline from disk.

5. *(blank line)*

6. `# Example: build a DataFrame with the same feature columns used in training`
   - Comment explains why this DataFrame exists.

7. `sample = pd.DataFrame(`
   - Creates a DataFrame from list of row dictionaries.

8. `[`
   - Starts list of rows.

9. `{`
   - Starts first row as dict.

10-29. key-value pairs for each feature
   - Each key must match training feature names.
   - Values are realistic sample values.

30. `]`
   - Ends rows list.

31. `)`
   - Ends DataFrame constructor.

32. *(blank line)*

33. `proba_churn = pipeline.predict_proba(sample)[:, 1]`
   - Gets probability of class 1 (churn) for each row.

34. `pred_churn = pipeline.predict(sample)`
   - Gets hard predicted class labels.

35. *(blank line)*

36. `print("probability:", float(proba_churn[0]))`
   - Prints first row probability as scalar float.

37. `print("predicted_label:", int(pred_churn[0]))`
   - Prints first row predicted class as int.

## Understanding the Libraries

### pandas

- `pd.read_csv(...)` loads CSV into DataFrame.
- `DataFrame.drop(...)` removes columns.
- `Series.map(...)` transforms labels.
- `pd.to_numeric(..., errors="coerce")` handles dirty numeric strings.

### scikit-learn

- `Pipeline`: chains preprocessing + estimator.
- `ColumnTransformer`: branch preprocessing by column dtype.
- `SimpleImputer`: missing value handling.
- `StandardScaler`: normalization for numeric stability.
- `OneHotEncoder`: categorical to machine-readable matrix.
- `LogisticRegression` and `RandomForestClassifier`: supervised models.
- `GridSearchCV`: model/hyperparameter selection with CV.

### joblib

- Efficiently saves/loading large NumPy/sklearn objects.

### pathlib

- Safer, cleaner path handling than manual string concatenation.

## Training Process Explained

1. Load raw data.
2. Clean/convert columns (`TotalCharges`, target mapping, drop ID).
3. Split data with stratification.
4. Build preprocessing graph for numeric/categorical features.
5. Wrap preprocessor + model in one pipeline.
6. Define two model families in grid (logistic and forest).
7. Run 5-fold CV search maximizing ROC-AUC.
8. Evaluate best pipeline on held-out test set.
9. Save model and reports.

## Evaluation Metrics Explained

### ROC-AUC

- Measures ranking quality across all classification thresholds.
- 0.5 is random; 1.0 is perfect.
- Good choice when class imbalance exists and probability ranking matters.

### Accuracy

- Fraction of correct predictions.
- Can be misleading if dataset is imbalanced.

### Classification Report

Provides per class:

- **Precision**: among predicted positives, how many are truly positive.
- **Recall**: among true positives, how many were found.
- **F1-score**: harmonic mean of precision and recall.
- **Support**: number of true samples in that class.

## Beginner Path (30 Minutes)

This is a fast, practical path for someone who knows Python but is new to ML pipelines.

### Minute 0-5: Understand the big picture

1. Read `README.md` for project goal and outputs.
2. Open `docs/DEVELOPER_GUIDE.md` and read:
   - What This Project Does
   - Key Concepts Explained
   - How the Data Flows + ASCII Flowchart

### Minute 5-10: Understand the data

1. Read `Dataset/About_Dataset.md`.
2. Confirm target variable is `Churn`.
3. Identify which columns are numeric vs categorical.

### Minute 10-18: Understand training code

1. Read `src/train.py` from top to bottom.
2. Focus on these functions in order:
   - `load_data()`
   - `build_preprocessor(...)`
   - `train_model()`
   - `save_outputs(...)`
3. Connect each function to one pipeline stage in the flowchart.

### Minute 18-24: Run and inspect outputs

1. Install deps: `pip install -r requirements.txt`.
2. Run training: `python src/train.py`.
3. Open generated files:
   - `reports/metrics.json`
   - `reports/classification_report.txt`
4. Observe which model/hyperparameters were selected.

### Minute 24-30: Run inference and modify one thing

1. Run prediction script: `python src/predict.py`.
2. Change one sample value in `src/predict.py` (for example `Contract` or `tenure`) and rerun.
3. Optional quick experiment: in `src/train.py`, change `test_size` or one grid value, retrain, compare metrics.

If you complete this sequence, you will understand the full lifecycle: data -> preprocessing -> model search -> evaluation -> saved pipeline -> prediction.

## Customization Guide

### Change train/test split

In `src/train.py`, edit `test_size` in `train_test_split(...)`.

### Add/replace model candidates

- Add a new dict inside `param_grid`.
- Set `"model": [YourEstimator(...)]`.
- Add estimator params as `"model__param_name"`.

### Change optimization metric

In `GridSearchCV`, change `scoring="roc_auc"` to another scorer (for example `"f1"`, `"accuracy"`).

### Add feature engineering

- Add new preprocessing step in `numeric_pipeline` or `categorical_pipeline`.
- Or add a custom transformer class and place it before model step.

### Use class weights for imbalance

For logistic regression, try `class_weight="balanced"`.

### Improve reproducibility

- Keep fixed `random_state` values.
- Pin package versions in `requirements.txt`.

## Troubleshooting

### Error: `FileNotFoundError` for dataset

- Ensure `Dataset/Telco-Customer-Churn.csv` exists.
- Run command from project root.

### Error: `No module named ...`

- Install dependencies: `pip install -r requirements.txt`.
- Ensure active virtual environment is correct.

### Error during `predict.py`: missing columns

- Input DataFrame must include all training feature names with compatible types.

### Warning about convergence (logistic regression)

- Increase `max_iter`.
- Scale numeric data (already done here).

### Very slow training

- Grid search is combinational; reduce grid size.
- Keep `n_jobs=-1` for multi-core usage.

## Glossary

- **Artifact**: a saved output file from ML workflow (model, metrics, report).
- **Binary Classification**: predicting one of two classes.
- **Cross-validation (CV)**: repeated train/validation splits for robust scoring.
- **Estimator**: scikit-learn model object implementing `fit` and `predict`.
- **Feature**: input variable used by model.
- **Hyperparameter**: setting chosen before training (for example `C`, `max_depth`).
- **Imputation**: filling missing values.
- **Inference**: using trained model to make predictions on new data.
- **One-hot Encoding**: converting categories to binary columns.
- **Pipeline**: ordered chain of preprocessors + model.
- **ROC curve**: plot of TPR vs FPR across thresholds.
- **Stratification**: preserving class ratio while splitting dataset.
- **Target**: label to predict (`Churn`).
