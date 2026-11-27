import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# =====================================
# Load Data
# =====================================

train = pd.read_csv(
    "data/processed/train_dataset.csv"
)

trade = pd.read_csv(
    "data/processed/trade_dataset.csv"
)

# =====================================
# Use same reduced RF feature set
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
# Fill missing values
# =====================================

train[features] = train[features].fillna(0)
trade[features] = trade[features].fillna(0)

# =====================================
# Train winning RF model
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
# Collect Top-5 per day
# =====================================

dates = sorted(
    trade["snapshot_date"].unique()
)

selected = []

for date in dates:

    day_data = trade[
        trade["snapshot_date"] == date
    ].copy()

    top5 = (
        day_data
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
# Selected contracts
# =====================================

print("\nSELECTED CONTRACTS SUMMARY")
print("=" * 70)

print(
    selected[
        [
            "dte",
            "mid_price",
            "relative_spread",
            "volume",
            "openInterest",
            "target_return"
        ]
    ].describe()
)

print("\nPUT PERCENTAGE")
print("=" * 70)

puts = (
    (selected["is_call"] == 0)
    .mean()
)

print(
    f"Puts Selected: {puts:.2%}"
)

# =====================================
# Compare to entire universe
# =====================================

print("\n")
print("=" * 70)
print("SELECTED VS ENTIRE UNIVERSE")
print("=" * 70)

comparison = pd.DataFrame({
    "Universe_DTE": [trade["dte"].mean()],
    "Selected_DTE": [selected["dte"].mean()],
    "Universe_MidPrice": [trade["mid_price"].mean()],
    "Selected_MidPrice": [selected["mid_price"].mean()],
    "Universe_Spread": [trade["relative_spread"].mean()],
    "Selected_Spread": [selected["relative_spread"].mean()],
    "Universe_Volume": [trade["volume"].mean()],
    "Selected_Volume": [selected["volume"].mean()],
    "Universe_OI": [trade["openInterest"].mean()],
    "Selected_OI": [selected["openInterest"].mean()],
    "Universe_Return": [trade["target_return"].mean()],
    "Selected_Return": [selected["target_return"].mean()]
})

print(
    comparison.round(2)
    .to_string(index=False)
)

# =====================================
# Show picks
# =====================================

print("\n")
print("=" * 70)
print("TOP 20 SELECTED CONTRACTS")
print("=" * 70)

print(
    selected[
        [
            "snapshot_date",
            "contractSymbol",
            "predicted_return",
            "target_return",
            "mid_price",
            "dte",
            "relative_spread"
        ]
    ]
    .sort_values(
        "predicted_return",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)