import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

train = pd.read_csv(
    "data/processed/train_dataset.csv"
)

trade = pd.read_csv(
    "data/processed/trade_dataset.csv"
)

features = [
    "is_call"
]

target = "target_return"

X_train = train[features]
y_train = train[target]

X_test = trade[features]
y_test = trade[target]

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

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

print("\nIS_CALL ONLY MODEL")
print("-" * 40)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print(
    f"Direction Accuracy: {direction_accuracy:.4%}"
)