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
# Optional Restriction
# =====================================

trade = trade[
    trade["dte"] >= 14
].copy()

# =====================================
# Features (No Spread Version)
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
# Missing Values
# =====================================

train[features] = train[features].fillna(0)
trade[features] = trade[features].fillna(0)

# =====================================
# Train Model
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
# Fixed-Dollar Simulation
# =====================================

starting_capital = 100000

portfolio_value = starting_capital

position_size = 5000
positions_per_day = 5

history = []

dates = sorted(
    trade["snapshot_date"].unique()
)

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
        .head(positions_per_day)
    )

    trade_profits = (
        position_size
        * top5["target_return_exec"]
    )

    daily_profit = trade_profits.sum()

    portfolio_value += daily_profit

    history.append(
        {
            "date": date,
            "daily_profit": daily_profit,
            "portfolio_value": portfolio_value,
            "average_return": top5["target_return_exec"].mean()
        }
    )

# =====================================
# Results
# =====================================

results = pd.DataFrame(
    history
)

# =====================================
# Summary Metrics
# =====================================

total_profit = (
    portfolio_value -
    starting_capital
)

roi = (
    total_profit /
    starting_capital
)

winning_days = (
    results["daily_profit"] > 0
).mean()

avg_daily_profit = (
    results["daily_profit"]
    .mean()
)

best_day = (
    results["daily_profit"]
    .max()
)

worst_day = (
    results["daily_profit"]
    .min()
)

# =====================================
# Output
# =====================================

print("\nRF V2 NO-SPREAD FIXED-$5,000 SIMULATION")
print("-" * 60)

print(
    f"Starting Capital: ${starting_capital:,.2f}"
)

print(
    f"Position Size: ${position_size:,.0f}"
)

print(
    f"Positions Per Day: {positions_per_day}"
)

print(
    f"Maximum Daily Exposure: "
    f"${position_size * positions_per_day:,.0f}"
)

print()

print(
    f"Final Portfolio Value: "
    f"${portfolio_value:,.2f}"
)

print(
    f"Total Profit: "
    f"${total_profit:,.2f}"
)

print(
    f"ROI: {roi:.2%}"
)

print(
    f"Average Daily Profit: "
    f"${avg_daily_profit:,.2f}"
)

print(
    f"Winning Days: "
    f"{winning_days:.2%}"
)

print(
    f"Best Day: "
    f"${best_day:,.2f}"
)

print(
    f"Worst Day: "
    f"${worst_day:,.2f}"
)

print()

print(
    results["daily_profit"]
    .describe()
)

# =====================================
# Save Results
# =====================================

results.to_csv(
    "data/processed/trading_results_rf_v2_no_spread_fixed5000.csv",
    index=False
)

print(
    "\nSaved: trading_results_rf_v2_no_spread_fixed5000.csv"
)