import joblib
import pandas as pd

pipeline = joblib.load("models/best_pipeline.joblib")

# Example: build a DataFrame with the same feature columns used in training
sample = pd.DataFrame(
    [
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 89.1,
            "TotalCharges": 1069.2,
        }
    ]
)

proba_churn = pipeline.predict_proba(sample)[:, 1]
pred_churn = pipeline.predict(sample)

print("probability:", float(proba_churn[0]))
print("predicted_label:", int(pred_churn[0]))