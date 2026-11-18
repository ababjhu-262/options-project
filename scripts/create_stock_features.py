import pandas as pd
import numpy as np

# Load stock history
df = pd.read_csv(
    "data/stock/nvda_history.csv"
)

# Convert date

df["Date"] = (
    pd.to_datetime(
        df["Date"],
        utc=True
    )
    .dt.tz_localize(None)
    .dt.normalize()
)


# Sort just in case
df = df.sort_values("Date")

# ==========================
# Returns
# ==========================

# Previous 1-day return
df["ret_1d"] = df["Close"].pct_change(1)

# Previous 5-day return
df["ret_5d"] = df["Close"].pct_change(5)

# Previous 10-day return
df["ret_10d"] = df["Close"].pct_change(10)

# ==========================
# Trend
# ==========================

df["ma20"] = (
    df["Close"]
      .rolling(20)
      .mean()
)

df["dist_ma20"] = (
    (df["Close"] - df["ma20"])
    / df["ma20"]
)

# ==========================
# Volatility
# ==========================

daily_returns = df["Close"].pct_change()

df["vol20"] = (
    daily_returns
      .rolling(20)
      .std()
)

# ==========================
# Save
# ==========================

output_file = (
    "data/stock/nvda_stock_features.csv"
)

df.to_csv(
    output_file,
    index=False
)

print(df.head(25))

print()
print("Saved:", output_file)