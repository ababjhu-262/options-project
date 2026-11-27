import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================
# Load training data only
# =====================================

df = pd.read_csv(
    "data/processed/train_dataset.csv"
)

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

df[features] = df[features].fillna(0)

# Must be sorted chronologically

df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"]
)

df = df.sort_values(
    "snapshot_date"
)

X = df[features]
y = df[target]

# =====================================
# Time Series Split
# =====================================

tscv = TimeSeriesSplit(
    n_splits=5
)

results = []

print("\nTIME SERIES CROSS VALIDATION")
print("=" * 70)

for fold, (train_idx, test_idx) in enumerate(
    tscv.split(X),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

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

    predictions = model.predict(
        X_test
    )

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

    results.append(
        [fold, mae, rmse, r2, direction_accuracy]
    )

    print(
        f"Fold {fold}: "
        f"MAE={mae:.4f}  "
        f"RMSE={rmse:.4f}  "
        f"R2={r2:.4f}  "
        f"Direction={direction_accuracy:.4%}"
    )

# =====================================
# Summary
# =====================================

results = pd.DataFrame(
    results,
    columns=[
        "Fold",
        "MAE",
        "RMSE",
        "R2",
        "Direction Accuracy"
    ]
)

print("\n")
print("=" * 70)
print("AVERAGE PERFORMANCE")
print("=" * 70)

print(
    results.mean(numeric_only=True)
)