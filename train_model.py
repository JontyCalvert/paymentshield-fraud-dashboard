import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load the generated transaction data
df = pd.read_csv("transactions.csv")

features = [
    "amount",
    "hour",
    "account_age_days",
    "is_new_device"
]

X = df[features]
y = df["is_fraud"]

# Split data while preserving the fraud proportion
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train a fraud-classification model
model = RandomForestClassifier(
    n_estimators=150,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Classification report:")
print(classification_report(y_test, predictions))

print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")

# Save the model and its feature list
joblib.dump(
    {
        "model": model,
        "features": features
    },
    "fraud_model.joblib"
)

print("Saved model to fraud_model.joblib")