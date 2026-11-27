import pandas as pd
import numpy as np

df = pd.read_csv(
    "data/processed/options_features.csv"
)

df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"]
)

df = df.sort_values(
    ["contractSymbol", "snapshot_date"]
)

# Future bid

df["next_bid"] = (
    df.groupby("contractSymbol")["bid"]
      .shift(-1)
)

# Executable return

df["target_return_exec"] = np.where(
    df["ask"] > 0,
    (df["next_bid"] - df["ask"])
    /
    df["ask"],
    np.nan
)

print("\nEXECUTABLE TARGET")
print("-" * 50)

print(
    df["target_return_exec"]
    .describe()
)

print("\nQUANTILES")
print(
    df["target_return_exec"]
    .quantile(
        [0.90, 0.95, 0.99, 0.999]
    )
)