
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Read the CCPP dataset
data = np.genfromtxt(
    "ccpp.csv",
    delimiter=",",
    names=True,
    dtype=float,
    filling_values=np.nan
)

# Define features and target
features = ["AT", "V", "AP", "RH"]

X = np.column_stack([data[name] for name in features])
y = data["PE"]

# Split the dataset
split_index = 7654

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# Baseline model
baseline_mean = np.mean(y_train)

baseline_predictions = np.full(
    len(y_test),
    baseline_mean
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    max_features=2,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
rf_mae = mean_absolute_error(
    y_test,
    predictions
)

# Display results
print("EEE213 - RANDOM FOREST REGRESSION")
print("----------------------------------")

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

print("Features:", features)
print("Target: PE (MW)")

print("\nModel Parameters")
print("n_estimators: 100")
print("max_features: 2")
print("random_state: 42")

print("\nModel Performance")
print(f"Baseline MAE: {baseline_mae:.2f} MW")
print(f"Random Forest MAE: {rf_mae:.2f} MW")
