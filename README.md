# PaymentShield: Explainable Payment Fraud Detection

PaymentShield is a demonstration fintech application that uses machine learning to estimate payment-fraud risk and present the results in an interactive Streamlit dashboard.

> **Disclaimer:** This project uses synthetic data. It is an educational prototype and is not suitable for making real payment decisions.

## Features

- Generates 10,000 synthetic payment transactions
- Trains a class-weighted Random Forest fraud classifier
- Produces fraud probabilities and 0–100 risk scores
- Assigns transactions to Approve, Review, or Block decisions
- Displays summary metrics and interactive Plotly charts
- Filters transactions by risk decision and country

## Technologies

- Python
- pandas and NumPy
- scikit-learn
- Streamlit
- Plotly
- joblib

## Model results

The model was evaluated on a stratified test set containing 2,000 transactions:

| Metric | Result |
|---|---:|
| Fraud precision | 0.85 |
| Fraud recall | 0.88 |
| Fraud F1-score | 0.86 |
| ROC-AUC | 0.998 |

These results should not be interpreted as real-world performance because the labels were generated using artificial rules.

## Risk decisions

| Risk score | Decision | Meaning |
|---:|---|---|
| 0–30 | Approve | Low predicted risk |
| 31–70 | Review | Needs additional checking |
| 71–100 | Block | High predicted risk |

## Project structure

```text
fintech-fraud-dashboard/
├── app.py
├── generate_data.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

The generated files `transactions.csv` and `fraud_model.joblib` are excluded from GitHub and are recreated locally by the setup instructions below.

## Setup

Open a terminal in the project folder and run:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python generate_data.py
python train_model.py
python -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## How it works

1. `generate_data.py` creates synthetic transactions and artificial fraud labels.
2. `train_model.py` trains and evaluates the Random Forest model, then saves `fraud_model.joblib`.
3. `app.py` loads the data and model, calculates risk scores, assigns decisions, and displays the dashboard.

## Future improvements

- Use a real anonymised fraud dataset
- Add time-based train/test splitting
- Tune decision thresholds using the cost of false positives and false negatives
- Add model explainability with SHAP
- Add authentication and audit logs
- Monitor model drift and performance over time
