# End-to-End ML Pipeline with Scikit-learn

This project trains and evaluates a customer churn classifier using a single, reusable
Scikit-learn `Pipeline` wrapped in `GridSearchCV`.

If you want full project documentation aimed at Python developers, read:

- `docs/DEVELOPER_GUIDE.md`

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train and evaluate:

```bash
python src/train.py
```

## Outputs

- `models/best_pipeline.joblib`: Trained preprocessing + model pipeline
- `reports/metrics.json`: Best hyperparameters + ROC-AUC + accuracy
- `reports/classification_report.txt`: Per-class precision/recall/F1 report
