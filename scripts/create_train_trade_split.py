import pandas as pd

# ===========================
# Load final dataset
# ===========================

df = pd.read_csv(
    "data/processed/final_modeling_dataset.csv"
)

df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"]
)

# ===========================
# Time-based split
# ===========================

train = df[
    df["snapshot_date"] < "2026-10-19"
].copy()

trade = df[
    df["snapshot_date"] >= "2026-10-19"
].copy()

# ===========================
# Save
# ===========================

train.to_csv(
    "data/processed/train_dataset.csv",
    index=False
)

trade.to_csv(
    "data/processed/trade_dataset.csv",
    index=False
)

# ===========================
# Diagnostics
# ===========================

print("\nTRAIN DATASET")
print("Rows:", len(train))
print(
    "Contracts:",
    train["contractSymbol"].nunique()
)

print(
    "Dates:",
    train["snapshot_date"].nunique()
)

print("\nTRADE DATASET")
print("Rows:", len(trade))
print(
    "Contracts:",
    trade["contractSymbol"].nunique()
)

print(
    "Dates:",
    trade["snapshot_date"].nunique()
)

print(
    "\nSaved train_dataset.csv and trade_dataset.csv"
)