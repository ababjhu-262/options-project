import pandas as pd
import numpy as np

# =====================================
# Load options data
# =====================================

df = pd.read_csv(
    "data/processed/options_features.csv"
)

# =====================================
# Sort by contract/date
# =====================================

df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"]
)

df = df.sort_values(
    ["contractSymbol", "snapshot_date"]
)

# =====================================
# Create future bid/ask values
# =====================================

df["next_bid"] = (
    df.groupby("contractSymbol")["bid"]
      .shift(-1)
)

df["next_ask"] = (
    df.groupby("contractSymbol")["ask"]
      .shift(-1)
)

df["next_mid"] = (
    df.groupby("contractSymbol")["mid_price"]
      .shift(-1)
)

# =====================================
# Return Definitions
# =====================================

# Current project target
df["mid_return"] = (
    (df["next_mid"] - df["mid_price"])
    / df["mid_price"]
)

# Conservative
df["bid_return"] = np.where(
    df["bid"] > 0,
    (df["next_bid"] - df["bid"])
    / df["bid"],
    np.nan
)

# Quote-to-quote
df["ask_return"] = np.where(
    df["ask"] > 0,
    (df["next_ask"] - df["ask"])
    / df["ask"],
    np.nan
)

# Realistic execution
df["ask_to_bid_return"] = np.where(
    df["ask"] > 0,
    (df["next_bid"] - df["ask"])
    / df["ask"],
    np.nan
)

# =====================================
# Summary Statistics
# =====================================

returns = [
    "mid_return",
    "bid_return",
    "ask_return",
    "ask_to_bid_return"
]

print("\nRETURN COMPARISON")
print("=" * 80)

for col in returns:

    print(f"\n{col.upper()}")
    print("-" * 40)

    print(
        df[col]
        .describe()
    )

# =====================================
# Extreme Winners Comparison
# =====================================

print("\n")
print("=" * 80)
print("TOP 20 MIDPOINT RETURNS")
print("=" * 80)

top20 = (
    df[
        [
            "contractSymbol",
            "snapshot_date",
            "bid",
            "ask",
            "next_bid",
            "next_ask",
            "mid_price",
            "next_mid",
            "mid_return",
            "bid_return",
            "ask_return",
            "ask_to_bid_return",
            "volume",
            "openInterest"
        ]
    ]
    .sort_values(
        "mid_return",
        ascending=False
    )
    .head(20)
)

print(
    top20.to_string(index=False)
)

# =====================================
# Correlation Between Targets
# =====================================

print("\n")
print("=" * 80)
print("TARGET CORRELATIONS")
print("=" * 80)

corr = df[
    [
        "mid_return",
        "bid_return",
        "ask_return",
        "ask_to_bid_return"
    ]
].corr()

print(
    corr.round(4)
)