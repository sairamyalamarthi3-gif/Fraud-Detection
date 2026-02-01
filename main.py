import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

np.random.seed(42)

n_samples = 10000

data = pd.DataFrame({
    "amount": np.random.exponential(scale=100, size=n_samples),
    "hour": np.random.randint(0, 24, n_samples),
    "is_foreign_transaction": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    "is_high_risk_country": np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    "transaction_velocity": np.random.poisson(2, n_samples),
})

# Fraud logic (hidden truth)
data["fraud"] = (
    (data["amount"] > 500) &
    (data["is_foreign_transaction"] == 1) &
    (data["transaction_velocity"] > 3)
).astype(int)

print(data.head())
