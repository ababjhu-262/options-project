import yfinance as yf
from pathlib import Path

ticker = yf.Ticker("NVDA")

history = ticker.history(
    start="2026-01-01",
    end="2026-11-18"
)

output_file = Path(
    "data/stock/nvda_history.csv"
)

history.to_csv(output_file)

print()
print(f"Rows: {len(history)}")
print()
print(history.head())
print()
print(f"Saved: {output_file}")