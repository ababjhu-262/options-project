import pandas as pd

# =====================================
# Load datasets
# =====================================

train = pd.read_csv(
    "data/processed/train_dataset.csv"
)

trade = pd.read_csv(
    "data/processed/trade_dataset.csv"
)

# =====================================
# Basic shape information
# =====================================

print("=" * 60)
print("TRAIN DATASET")
print("=" * 60)

print(f"Rows: {len(train):,}")
print(f"Columns: {len(train.columns)}")
print(f"Unique Contracts: {train['contractSymbol'].nunique():,}")
print(f"Unique Dates: {train['snapshot_date'].nunique()}")

print()

print("=" * 60)
print("TRADE DATASET")
print("=" * 60)

print(f"Rows: {len(trade):,}")
print(f"Columns: {len(trade.columns)}")
print(f"Unique Contracts: {trade['contractSymbol'].nunique():,}")
print(f"Unique Dates: {trade['snapshot_date'].nunique()}")

# =====================================
# Missing values
# =====================================

print()
print("=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = (
    train.isna()
         .sum()
         .sort_values(ascending=False)
)

missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing)

# =====================================
# Target distribution
# =====================================

print()
print("=" * 60)
print("TARGET RETURN SUMMARY")
print("=" * 60)

print(
    train["target_return"].describe()
)

print()

print("Quantiles")

print(
    train["target_return"].quantile(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )
)

# =====================================
# Candidate modeling features
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
    "ret_5d",
    "ret_10d",
    "dist_ma20",
    "vol20"
]

print()
print("=" * 60)
print("FEATURE SUMMARY")
print("=" * 60)

print(
    train[features]
    .describe()
    .round(4)
)

# =====================================
# Correlations with target
# =====================================

print()
print("=" * 60)
print("CORRELATION WITH TARGET")
print("=" * 60)

corr = (
    train[features + ["target_return"]]
    .corr(numeric_only=True)["target_return"]
    .sort_values(ascending=False)
)

print(
    corr.round(4)
)

print()
print("=" * 60)
print("DATASET EXPLORATION COMPLETE")
print("=" * 60)