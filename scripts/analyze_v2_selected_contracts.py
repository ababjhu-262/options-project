import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

# =====================================
# Load Data
# =====================================

train = pd.read_csv(
    "data/processed/train_dataset_v2.csv"
)

trade = pd.read_csv(
    "data/processed/trade_dataset_v2.csv"
)

# =====================================
# No-Spread Feature Set
# =====================================

features = [
    "stock_price",
    "strike",
    "volume",
    "openInterest",
    "impliedVolatility",
    "dte",
    "mid_price",
    "moneyness",
    "is_call",
    "volume_oi_ratio",
    "Volume",
    "ret_1d",
    "vol20"
]

target = "target_return_exec"

# =====================================
# Fill Missing Values
# =====================================

train[features] = train[features].fillna(0)
trade[features] = trade[features].fillna(0)

# =====================================
# Train RF
# =====================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)

model.fit(
    train[features],
    train[target]
)

# =====================================
# Predict
# =====================================

trade["predicted_return"] = model.predict(
    trade[features]
)

# =====================================
# Collect Top 5 Per Day
# =====================================

selected = []

dates = sorted(
    trade["snapshot_date"].unique()
)

for date in dates:

    top5 = (
        trade[
            trade["snapshot_date"] == date
        ]
        .sort_values(
            "predicted_return",
            ascending=False
        )
        .head(5)
    )

    selected.append(top5)

selected = pd.concat(
    selected,
    ignore_index=True
)

# =====================================
# Summary Statistics
# =====================================

print("\nSELECTED CONTRACT SUMMARY")
print("=" * 80)

print(
    selected[
        [
            "target_return_exec",
            "predicted_return",
            "dte",
            "mid_price",
            "volume",
            "openInterest",
            "impliedVolatility"
        ]
    ]
    .describe()
)

# =====================================
# Calls vs Puts
# =====================================

print("\nPUT PERCENTAGE")
print("=" * 80)

puts = (
    (selected["is_call"] == 0)
    .mean()
)

print(
    f"Puts Selected: {puts:.2%}"
)

# =====================================
# Compare Universe vs Selected
# =====================================

print("\nUNIVERSE VS SELECTED")
print("=" * 80)

comparison = pd.DataFrame({
    "Universe_DTE": [trade["dte"].mean()],
    "Selected_DTE": [selected["dte"].mean()],
    "Universe_MidPrice": [trade["mid_price"].mean()],
    "Selected_MidPrice": [selected["mid_price"].mean()],
    "Universe_Volume": [trade["volume"].mean()],
    "Selected_Volume": [selected["volume"].mean()],
    "Universe_OI": [trade["openInterest"].mean()],
    "Selected_OI": [selected["openInterest"].mean()],
    "Universe_Return": [trade["target_return_exec"].mean()],
    "Selected_Return": [selected["target_return_exec"].mean()]
})

print(
    comparison
    .round(2)
    .to_string(index=False)
)

# =====================================
# Top Realized Winners Selected
# =====================================

print("\nTOP REALIZED RETURNS AMONG SELECTED")
print("=" * 80)

print(
    selected[
        [
            "snapshot_date",
            "contractSymbol",
            "predicted_return",
            "target_return_exec",
            "dte",
            "mid_price",
            "volume",
            "openInterest",
            "impliedVolatility"
        ]
    ]
    .sort_values(
        "target_return_exec",
        ascending=False
    )
    .head(25)
    .to_string(index=False)
)

# =====================================
# Daily Portfolio Returns
# =====================================

print("\nDAILY TOP-5 RETURNS")
print("=" * 80)

daily_summary = []

for date in dates:

    top5 = (
        trade[
            trade["snapshot_date"] == date
        ]
        .sort_values(
            "predicted_return",
            ascending=False
        )
        .head(5)
    )

    daily_summary.append({
        "date": date,
        "portfolio_return": top5["target_return_exec"].mean(),
        "avg_dte": top5["dte"].mean(),
        "avg_mid_price": top5["mid_price"].mean()
    })

daily_summary = pd.DataFrame(
    daily_summary
)

print(
    daily_summary
    .sort_values(
        "portfolio_return",
        ascending=False
    )
    .to_string(index=False)
)