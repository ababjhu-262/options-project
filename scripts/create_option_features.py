from pathlib import Path
import pandas as pd
import numpy as np

# ==========================
# Load master dataset
# ==========================

df = pd.read_csv(
    "data/processed/options_master.csv"
)

print(f"Rows loaded: {len(df):,}")

# ==========================
# Convert dates
# ==========================

df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"]
)

df["expiration"] = pd.to_datetime(
    df["expiration"]
)

# ==========================
# Basic Feature Engineering
# ==========================

# Days to expiration
df["dte"] = (
    df["expiration"]
    - df["snapshot_date"]
).dt.days

# Midpoint price
df["mid_price"] = (
    df["bid"]
    + df["ask"]
) / 2

# Bid/Ask spread
df["spread"] = (
    df["ask"]
    - df["bid"]
)

# Relative spread
df["relative_spread"] = (
    df["spread"]
    / df["mid_price"]
)

# Moneyness
df["moneyness"] = (
    df["stock_price"]
    / df["strike"]
)

# Call indicator
df["is_call"] = (
    df["option_type"] == "CALL"
).astype(int)

# Volume/Open Interest
df["volume_oi_ratio"] = (
    df["volume"]
    / df["openInterest"].replace(0, np.nan)
)

# ==========================
# Build Target Variable
# ==========================

df = df.sort_values(
    ["contractSymbol", "snapshot_date"]
)

# Tomorrow's mid-price
df["next_mid_price"] = (
    df.groupby("contractSymbol")
      ["mid_price"]
      .shift(-1)
)

# Target return
df["target_return"] = (
    df["next_mid_price"]
    - df["mid_price"]
) / df["mid_price"]

# ==========================
# Basic Cleaning
# ==========================

# Remove very low-priced contracts.
# Contracts below $0.10 generated unstable percentage returns
# due to bid-ask quote noise and illiquidity.

df = df[
    df["mid_price"] >= 0.10
]


df = df[
    df["dte"] > 0
]

df = df[
    df["impliedVolatility"] > 0
]

df = df.dropna(
    subset=["target_return"]
)

print()
print("Shape after cleaning:")
print(df.shape)

# ==========================
# Save Output
# ==========================

output_file = (
    "data/processed/options_features.csv"
)

df.to_csv(
    output_file,
    index=False
)

print()
print(f"Saved: {output_file}")