import pandas as pd
import numpy as np

# =====================================
# Load Training Data
# =====================================

train = pd.read_csv(
    "data/processed/train_dataset.csv"
)

features = [
    "stock_price",
    "strike",
    "volume",
    "openInterest",
    "impliedVolatility",
    "dte",
    "mid_price",
    "spread",
    "relative_spread",
    "moneyness",
    "is_call",
    "volume_oi_ratio",
    "Volume",
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "dist_ma20",
    "vol20"
]

# Fix missing values
train[features] = train[features].fillna(0)

# =====================================
# Correlation Matrix
# =====================================

corr = train[features].corr(numeric_only=True)

# =====================================
# High Correlations
# =====================================

print("\nHIGH FEATURE CORRELATIONS")
print("-" * 60)

high_corrs = []

for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):

        var1 = corr.columns[i]
        var2 = corr.columns[j]

        value = corr.iloc[i, j]

        if abs(value) >= 0.70:
            high_corrs.append(
                (var1, var2, value)
            )

high_corrs = sorted(
    high_corrs,
    key=lambda x: abs(x[2]),
    reverse=True
)

for var1, var2, value in high_corrs:
    print(
        f"{var1:20s} {var2:20s} {value:.4f}"
    )

# =====================================
# Top Correlations by Feature
# =====================================

print()
print("TOP CORRELATIONS PER FEATURE")
print("-" * 60)

for feature in features:

    correlations = (
        corr[feature]
        .drop(feature)
        .abs()
        .sort_values(
            ascending=False
        )
    )

    top_feature = correlations.index[0]
    top_corr = correlations.iloc[0]

    print(
        f"{feature:20s} -> "
        f"{top_feature:20s} "
        f"({top_corr:.4f})"
    )