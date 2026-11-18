from pathlib import Path
import pandas as pd

# location of raw files
option_dir = Path("data/options")

# get all csvs
files = sorted(option_dir.glob("*.csv"))

print(f"Found {len(files)} files")

dfs = []

for file in files:
    print(f"Reading {file.name}")
    dfs.append(pd.read_csv(file))

master = pd.concat(
    dfs,
    ignore_index=True
)

print()
print("Combined Shape:")
print(master.shape)

print()
print("Columns:")
print(master.columns.tolist())

master.to_csv(
    "data/processed/options_master.csv",
    index=False
)

print()
print("Saved data/processed/options_master.csv")