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
# Quote Quality Filters (V2)
# ==========================

print("\nApplying quote quality filters...")

before_rows = len(df)

df = df[
    (df["ask"] > 0)
    &
    (df["bid"] >= 0)
    &
    (df["bid"] <= df["ask"])
].copy()

print(
    f"Removed {before_rows - len(df):,} rows "
    f"with invalid bid/ask quotes"
)

# ==========================
# Remove Penny Options
# ==========================

before_rows = len(df)

df = df[
    df["ask"] >= 0.10
].copy()

print(
    f"Removed {before_rows - len(df):,} rows "
    f"with ask < $0.10"
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
    /
    df["openInterest"].replace(
        0,
        np.nan
    )
)

# ==========================
# Build Executable Target
# ==========================

df = df.sort_values(
    ["contractSymbol", "snapshot_date"]
)

# Future bid price
df["next_bid"] = (
    df.groupby("contractSymbol")
      ["bid"]
      .shift(-1)
)

# Future ask price (diagnostic only)
df["next_ask"] = (
    df.groupby("contractSymbol")
      ["ask"]
      .shift(-1)
)

# Future midpoint (diagnostic only)
df["next_mid_price"] = (
    df.groupby("contractSymbol")
      ["mid_price"]
      .shift(-1)
)

# Executable return
#
# Buy today at ask
# Sell next period at bid
#
df["target_return_exec"] = (
    df["next_bid"]
    -
    df["ask"]
) / df["ask"]

# ==========================
# Additional Cleaning
# ==========================

df = df[
    df["dte"] > 0
]

df = df[
    df["impliedVolatility"] > 0
]

df = df.dropna(
    subset=["target_return_exec"]
)

# ==========================
# Diagnostics
# ==========================

print("\n")
print("=" * 60)
print("EXECUTABLE TARGET SUMMARY")
print("=" * 60)

print(
    df["target_return_exec"]
    .describe()
)

print("\nQUANTILES")

print(
    df["target_return_exec"]
    .quantile(
        [
            0.90,
            0.95,
            0.99,
            0.999
        ]
    )
)

print("\nTOP 10 RETURNS")

print(
    df[
        [
            "contractSymbol",
            "snapshot_date",
            "bid",
            "ask",
            "next_bid",
            "volume",
            "openInterest",
            "target_return_exec"
        ]
    ]
    .sort_values(
        "target_return_exec",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

# ==========================
# Save Output
# ==========================

output_file = (
    "data/processed/options_features_v2.csv"
)

df.to_csv(
    output_file,
    index=False
)

print()
print(f"Saved: {output_file}")