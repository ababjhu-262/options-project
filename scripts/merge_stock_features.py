import pandas as pd

# =====================================
# Load datasets
# =====================================

options = pd.read_csv(
    "data/processed/options_features.csv"
)

stock = pd.read_csv(
    "data/stock/nvda_stock_features.csv"
)

# =====================================
# Convert dates
# =====================================

options["snapshot_date"] = pd.to_datetime(
    options["snapshot_date"]
)

stock["Date"] = pd.to_datetime(
    stock["Date"]
)

# Remove time components

options["snapshot_date"] = (
    options["snapshot_date"]
      .dt.normalize()
)

stock["Date"] = (
    stock["Date"]
      .dt.normalize()
)

# =====================================
# Sort for merge_asof
# =====================================

options = options.sort_values(
    "snapshot_date"
)

stock = stock.sort_values(
    "Date"
)

# =====================================
# Merge
# =====================================

merged = pd.merge_asof(
    options,
    stock,
    left_on="snapshot_date",
    right_on="Date",
    direction="backward"
)

# =====================================
# Diagnostics
# =====================================

print("\nMerged Shape:")
print(merged.shape)

print("\nMissing Values")

for col in [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ma20",
    "dist_ma20",
    "vol20"
]:
    print(
        f"{col}:",
        merged[col].isna().sum()
    )

# =====================================
# Save
# =====================================

output_file = (
    "data/processed/final_modeling_dataset.csv"
)

merged.to_csv(
    output_file,
    index=False
)

print()
print(f"Saved: {output_file}")