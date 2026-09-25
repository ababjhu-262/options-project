# download_options.py

import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ----------------------------
# Configuration
# ----------------------------

TICKER = "NVDA"

# Create output directory if it doesn't exist
output_dir = BASE_DIR / "data" / "options"
output_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Download data
# ----------------------------

ticker = yf.Ticker(TICKER)

snapshot_date = datetime.now().strftime("%Y-%m-%d")

# Current stock price
stock_price = ticker.history(period="1d")["Close"].iloc[-1]

print(f"Downloading option chains for {TICKER}")
print(f"Current stock price: {stock_price:.2f}")

all_options = []

# Loop through every expiration date
for expiration in ticker.options:

    print(f"  Processing expiration: {expiration}")

    try:
        chain = ticker.option_chain(expiration)

        # Calls
        calls = chain.calls.copy()
        calls["option_type"] = "CALL"

        # Puts
        puts = chain.puts.copy()
        puts["option_type"] = "PUT"

        # Add metadata
        for df in (calls, puts):
            df["snapshot_date"] = snapshot_date
            df["expiration"] = expiration
            df["stock_symbol"] = TICKER
            df["stock_price"] = stock_price

        all_options.append(calls)
        all_options.append(puts)

    except Exception as e:
        print(f"  Error loading {expiration}: {e}")

# Combine everything
options_df = pd.concat(all_options, ignore_index=True)

# Save file
filename = output_dir / f"{TICKER}_options_{snapshot_date}.csv"

options_df.to_csv(filename, index=False)

print()
print(f"Saved {len(options_df):,} option contracts")
print(f"Output file: {filename}")

print("\nColumns:")
print(options_df.columns.tolist())