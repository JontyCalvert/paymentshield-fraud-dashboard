import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="PaymentShield Fraud Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ PaymentShield")
st.subheader("Payment Fraud Detection Dashboard")

# Load data and trained model
df = pd.read_csv("transactions.csv")
saved_model = joblib.load("fraud_model.joblib")

model = saved_model["model"]
features = saved_model["features"]

# Generate fraud probabilities
df["fraud_probability"] = model.predict_proba(df[features])[:, 1]
df["risk_score"] = (df["fraud_probability"] * 100).round(1)

# Assign business decisions
df["decision"] = pd.cut(
    df["risk_score"],
    bins=[-1, 30, 70, 100],
    labels=["Approve", "Review", "Block"]
)

# Summary metrics
total_transactions = len(df)
fraud_rate = df["is_fraud"].mean()
average_amount = df["amount"].mean()
blocked_transactions = (df["decision"] == "Block").sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Transactions", f"{total_transactions:,}")
col2.metric("Actual Fraud Rate", f"{fraud_rate:.2%}")
col3.metric("Average Amount", f"£{average_amount:,.2f}")
col4.metric("Blocked Transactions", f"{blocked_transactions:,}")

st.divider()

# Charts
left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Risk Decisions")
    decision_counts = (
        df["decision"]
        .value_counts()
        .reindex(["Approve", "Review", "Block"])
        .fillna(0)
        .reset_index()
    )
    decision_counts.columns = ["decision", "count"]

    chart = px.bar(
        decision_counts,
        x="decision",
        y="count",
        color="decision",
        color_discrete_map={
            "Approve": "green",
            "Review": "orange",
            "Block": "red"
        }
    )
    st.plotly_chart(chart, use_container_width=True)

with right_column:
    st.subheader("Fraud by Merchant Category")

    fraud_by_category = (
        df.groupby("merchant_category")["is_fraud"]
        .mean()
        .mul(100)
        .reset_index(name="fraud_rate")
    )

    chart = px.bar(
        fraud_by_category,
        x="merchant_category",
        y="fraud_rate",
        labels={"fraud_rate": "Fraud Rate (%)"}
    )
    st.plotly_chart(chart, use_container_width=True)

# Filters
st.divider()
st.subheader("Transaction Explorer")

selected_decision = st.selectbox(
    "Filter by decision",
    ["All", "Approve", "Review", "Block"]
)

selected_country = st.selectbox(
    "Filter by country",
    ["All"] + sorted(df["country"].unique().tolist())
)

filtered_df = df.copy()

if selected_decision != "All":
    filtered_df = filtered_df[
        filtered_df["decision"] == selected_decision
    ]

if selected_country != "All":
    filtered_df = filtered_df[
        filtered_df["country"] == selected_country
    ]

display_columns = [
    "amount",
    "hour",
    "account_age_days",
    "is_new_device",
    "country",
    "merchant_category",
    "risk_score",
    "decision",
    "is_fraud"
]

st.dataframe(
    filtered_df[display_columns]
    .sort_values("risk_score", ascending=False)
    .head(100),
    use_container_width=True
)