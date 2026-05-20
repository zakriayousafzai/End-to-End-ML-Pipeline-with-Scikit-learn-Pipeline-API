from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).resolve().parents[1] / "Dataset" / "Telco-Customer-Churn.csv"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_features = features.select_dtypes(include=["object"]).columns
    numeric_features = features.select_dtypes(exclude=["object"]).columns

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def train_model() -> tuple[Pipeline, dict, str]:
    df = load_data()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    param_grid = [
        {
            "model": [LogisticRegression(max_iter=1000, solver="liblinear")],
            "model__C": [0.1, 1.0, 10.0],
            "model__penalty": ["l1", "l2"],
        },
        {
            "model": [RandomForestClassifier(random_state=42)],
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
        },
    ]

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    best_params = dict(search.best_params_)
    model_param = best_params.pop("model", None)
    if model_param is not None:
        best_params["model"] = model_param.__class__.__name__

    metrics = {
        "best_params": best_params,
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }
    report = classification_report(y_test, y_pred)

    return best_model, metrics, report


def save_outputs(best_model: Pipeline, metrics: dict, report: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_DIR / "best_pipeline.joblib")
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")


def main() -> None:
    best_model, metrics, report = train_model()
    save_outputs(best_model, metrics, report)

    print("Best parameters:", metrics["best_params"])
    print("ROC-AUC:", metrics["roc_auc"])
    print("Accuracy:", metrics["accuracy"])
    print(report)


if __name__ == "__main__":
    main()