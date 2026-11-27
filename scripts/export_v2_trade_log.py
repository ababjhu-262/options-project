import pandas as pd

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
# DTE Filter
# =====================================

trade = trade[
    trade["dte"] >= 14
].copy()

# =====================================
# Features (No Spread Model)
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
# Train Model
# =====================================

train[features] = train[features].fillna(0)
trade[features] = trade[features].fillna(0)

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
# Predictions
# =====================================

trade["predicted_return"] = model.predict(
    trade[features]
)

# =====================================
# Build Trade Log
# =====================================

trade_log = []

POSITION_SIZE = 5000

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

    for _, row in top5.iterrows():

        profit = (
            POSITION_SIZE
            * row["target_return_exec"]
        )

        trade_log.append(
            {
                "Date": row["snapshot_date"],
                "Contract": row["contractSymbol"],
                "Predicted_Return": row["predicted_return"],
                "Actual_Return": row["target_return_exec"],
                "Position_Size": POSITION_SIZE,
                "Profit": profit,
                "DTE": row["dte"],
                "Mid_Price": row["mid_price"],
                "Volume": row["volume"],
                "Open_Interest": row["openInterest"],
                "Implied_Volatility": row["impliedVolatility"]
            }
        )

# =====================================
# Export
# =====================================

trade_log = pd.DataFrame(
    trade_log
)

trade_log = trade_log.sort_values(
    ["Date", "Predicted_Return"],
    ascending=[True, False]
)

trade_log.to_csv(
    "data/processed/v2_trade_log.csv",
    index=False
)

# =====================================
# Summary
# =====================================

print("\nTRADE LOG SUMMARY")
print("-" * 60)

print(
    f"Trades: {len(trade_log)}"
)

print(
    f"Total Profit: ${trade_log['Profit'].sum():,.2f}"
)

print(
    f"Average Profit per Trade: "
    f"${trade_log['Profit'].mean():,.2f}"
)

print()

print(
    trade_log.head(20)
)

print(
    "\nSaved: data/processed/v2_trade_log.csv"
)