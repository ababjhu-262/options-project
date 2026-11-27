import pandas as pd
import numpy as np

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
# Features (Reduced RF)
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
# Missing Values
# =====================================

train[features] = train[features].fillna(0)
trade[features] = trade[features].fillna(0)

# =====================================
# REALISTIC TRADING FILTERS
# =====================================

trade = trade[
    (trade["volume"] >= 10)
    &
    (trade["openInterest"] >= 100)
    &
    (trade["relative_spread"] <= 0.50)
    &
    (trade["dte"] >= 14)
].copy()

print("\nTRADABLE UNIVERSE (DTE >= 14)")
print("-" * 60)

print(
    f"Remaining contracts: {len(trade):,}"
)

print(
    f"Unique contracts: "
    f"{trade['contractSymbol'].nunique():,}"
)

print(
    f"Average DTE: "
    f"{trade['dte'].mean():.1f}"
)

print(
    f"Average Relative Spread: "
    f"{trade['relative_spread'].mean():.3f}"
)

# =====================================
# Train Random Forest
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
# Predictions
# =====================================

trade["predicted_return"] = model.predict(
    trade[features]
)

# =====================================
# Trading Simulation
# =====================================

portfolio_value = 100000

history = []

dates = sorted(
    trade["snapshot_date"].unique()
)

for date in dates:

    day_data = trade[
        trade["snapshot_date"] == date
    ].copy()

    if len(day_data) < 5:
        continue

    top5 = (
        day_data
        .sort_values(
            "predicted_return",
            ascending=False
        )
        .head(5)
    )

    realized_return = (
        top5["target_return"]
        .mean()
    )

    portfolio_value *= (
        1 + realized_return
    )

    history.append(
        {
            "date": date,
            "portfolio_value": portfolio_value,
            "daily_return": realized_return
        }
    )

# =====================================
# Results
# =====================================

results = pd.DataFrame(
    history
)

final_value = portfolio_value

total_return = (
    final_value - 100000
) / 100000

win_rate = (
    results["daily_return"] > 0
).mean()

avg_daily_return = (
    results["daily_return"]
    .mean()
)

# =====================================
# Diagnostics
# =====================================

print("\nDAILY RETURN SUMMARY")
print("-" * 60)

print(
    results["daily_return"]
    .describe()
)

# =====================================
# Final Results
# =====================================

print("\nREALISTIC TRADING SIMULATION")
print("-" * 60)

print(
    f"Starting Value: $100,000"
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
    f"Trading Days: {len(results)}"
)

# =====================================
# Save Results
# =====================================

results.to_csv(
    "data/processed/trading_results_rf_realistic_dte14.csv",
    index=False
)

print()
print(
    "Saved: data/processed/trading_results_rf_realistic_dte14.csv"
)