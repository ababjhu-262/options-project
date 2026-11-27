import pandas as pd

# ===========================
# Load final dataset V2
# ===========================

df = pd.read_csv(
    "data/processed/final_modeling_dataset_v2.csv"
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
    "data/processed/train_dataset_v2.csv",
    index=False
)

trade.to_csv(
    "data/processed/trade_dataset_v2.csv",
    index=False
)

# ===========================
# Diagnostics
# ===========================

print("\nTRAIN DATASET V2")
print("-" * 50)

print("Rows:", len(train))

print(
    "Contracts:",
    train["contractSymbol"].nunique()
)

print(
    "Dates:",
    train["snapshot_date"].nunique()
)

print(
    "\nExecutable Return Summary"
)

print(
    train["target_return_exec"]
    .describe()
)

print("\nTRADE DATASET V2")
print("-" * 50)

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
    "\nExecutable Return Summary"
)

print(
    trade["target_return_exec"]
    .describe()
)

print(
    "\nSaved train_dataset_v2.csv and trade_dataset_v2.csv"
)