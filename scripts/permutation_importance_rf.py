import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# =====================================
# Load data
# =====================================

train = pd.read_csv(
    "data/processed/train_dataset.csv"
)

trade = pd.read_csv(
    "data/processed/trade_dataset.csv"
)

# =====================================
# Reduced feature set
# =====================================

features = [
    "stock_price",
    "strike",
    "volume",
    "openInterest",
    "impliedVolatility",
    "dte",
    "mid_price",
    "spread",
    "relative_spread",
    "moneyness",
    "is_call",
    "volume_oi_ratio",
    "Volume",
    "ret_1d",
    "vol20"
]

target = "target_return"

# =====================================
# Missing values
# =====================================

train[features] = train[features].fillna(0)
trade[features] = trade[features].fillna(0)

X_train = train[features]
y_train = train[target]

X_test = trade[features]
y_test = trade[target]

# =====================================
# Train winning random forest
# =====================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

# =====================================
# Permutation importance
# =====================================

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# =====================================
# Results
# =====================================

importance = pd.DataFrame(
    {
        "Feature": features,
        "Mean_Importance": result.importances_mean,
        "Std_Importance": result.importances_std
    }
)

importance = importance.sort_values(
    "Mean_Importance",
    ascending=False
)

print()
print("PERMUTATION IMPORTANCE")
print("-" * 60)

print(
    importance.round(6)
)

print()
print("-" * 60)
print(
    "Total Features:",
    len(features)
)