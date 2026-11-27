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
# Fill missing values
# =====================================

train[features] = train[features].fillna(0)
trade[features] = trade[features].fillna(0)

# =====================================
# Train Final RF
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
# Simulation
# =====================================

starting_capital = 100000

portfolio_value = starting_capital

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
        .head(5)
    )

    realized_return = (
        top5["target_return_exec"]
        .mean()
    )

    portfolio_value *= (
        1 + realized_return
    )

    history.append(
        {
            "date": date,
            "daily_return": realized_return,
            "portfolio_value": portfolio_value
        }
    )

# =====================================
# Results
# =====================================

results = pd.DataFrame(
    history
)

# =====================================
# Summary Statistics
# =====================================

final_value = portfolio_value

total_return = (
    final_value -
    starting_capital
) / starting_capital

win_rate = (
    results["daily_return"] > 0
).mean()

avg_daily_return = (
    results["daily_return"]
    .mean()
)

best_day = (
    results["daily_return"]
    .max()
)

worst_day = (
    results["daily_return"]
    .min()
)

# =====================================
# Output
# =====================================

print("\nRF V2 NO-SPREAD TRADING SIMULATION")
print("-" * 60)

print(
    f"Starting Capital: ${starting_capital:,.2f}"
)

print(
    f"Final Value: ${final_value:,.2f}"
)

print(
    f"Total Return: {total_return:.2%}"
)

print(
    f"Average Daily Return: {avg_daily_return:.2%}"
)

print(
    f"Winning Days: {win_rate:.2%}"
)

print(
    f"Best Day: {best_day:.2%}"
)

print(
    f"Worst Day: {worst_day:.2%}"
)

print(
    f"Trading Days: {len(results)}"
)

print("\nDAILY RETURN SUMMARY")
print("-" * 60)

print(
    results["daily_return"]
    .describe()
)

# =====================================
# Save
# =====================================

results.to_csv(
    "data/processed/trading_results_rf_v2_no_spread.csv",
    index=False
)

print(
    "\nSaved: trading_results_rf_v2_no_spread.csv"
)