import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================
# Load data
# =====================================

trade = pd.read_csv(
    "data/processed/trade_dataset.csv"
)

y_test = trade["target_return"]

# =====================================
# Baseline prediction
# =====================================

predictions = np.zeros(
    len(y_test)
)

# =====================================
# Metrics
# =====================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

direction_accuracy = (
    (y_test > 0)
    ==
    (predictions > 0)
).mean()

# =====================================
# Results
# =====================================

print("\nBASELINE MODEL")
print("-" * 40)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print(
    f"Direction Accuracy: {direction_accuracy:.4%}"
)