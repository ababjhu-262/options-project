import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

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
# Handle missing values
# =====================================

train[features] = train[features].fillna(0)
trade[features] = trade[features].fillna(0)

X_train = train[features]
y_train = train[target]

X_test = trade[features]
y_test = trade[target]

# =====================================
# Random Forest
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
# Predictions
# =====================================

predictions = model.predict(
    X_test
)

# =====================================
# Metrics
# =====================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

direction_accuracy = (
    (y_test > 0)
    ==
    (predictions > 0)
).mean()

# =====================================
# Results
# =====================================

print("\nRANDOM FOREST (REDUCED FEATURES)")
print("-" * 50)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print(
    f"Direction Accuracy: {direction_accuracy:.4%}"
)

# =====================================
# Feature Importance
# =====================================

importance = pd.DataFrame(
    {
        "Feature": features,
        "Importance": model.feature_importances_
    }
)

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print()
print("FEATURE IMPORTANCE")
print("-" * 50)

print(importance)