import numpy as np
import pandas as pd

np.random.seed(42)

n = 10000

df = pd.DataFrame({
    "amount": np.round(np.random.lognormal(mean=3.5, sigma=1.0, size=n), 2),
    "hour": np.random.randint(0, 24, size=n),
    "account_age_days": np.random.randint(1, 2000, size=n),
    "is_new_device": np.random.binomial(1, 0.15, size=n),
    "country": np.random.choice(
        ["UK", "US", "FR", "DE", "NG", "BR"],
        size=n,
        p=[0.35, 0.25, 0.12, 0.10, 0.10, 0.08]
    ),
    "merchant_category": np.random.choice(
        ["Retail", "Travel", "Gaming", "Food", "Electronics"],
        size=n
    )
})

df["is_fraud"] = (
    (df["amount"] > 250) |
    ((df["is_new_device"] == 1) & (df["amount"] > 100)) |
    (df["country"].isin(["NG", "BR"]) & (df["amount"] > 150))
).astype(int)

df.to_csv("transactions.csv", index=False)

print(f"Created {len(df)} transactions")
print(f"Fraud rate: {df['is_fraud'].mean():.2%}")