
import numpy as np

# Read the power plant dataset
data = np.genfromtxt(
    "ccpp.csv",
    delimiter=",",
    names=True,
    dtype=float,
    filling_values=np.nan
)

# Display basic dataset information
print("EEE213 - WEEK 09")
print("POWER PLANT DATASET")
print("------------------------------")

print("Number of rows:", len(data))
print("Number of columns:", len(data.dtype.names))
print("Column names:", data.dtype.names)

print("\nFirst observation:")
print(data[0])
# Check missing values

print("\nMissing Value Check")
print("------------------------------")

for column in data.dtype.names:
    missing_count = np.count_nonzero(
        np.isnan(data[column])
    )

    print(column, ":", missing_count)

# Check minimum and maximum values

print("\nPhysical Range Check")
print("------------------------------")

for column in data.dtype.names:
    minimum = np.min(data[column])
    maximum = np.max(data[column])

    print(f"{column}: {minimum:.2f} - {maximum:.2f}")

# Check exact duplicate rows

print("\nDuplicate Row Check")
print("------------------------------")

unique_rows, counts = np.unique(
    data,
    return_counts=True
)

duplicate_groups = np.count_nonzero(counts > 1)

duplicate_rows = np.sum(counts[counts > 1] - 1)

print("Unique rows:", len(unique_rows))
print("Duplicate groups:", duplicate_groups)
print("Extra duplicate rows:", duplicate_rows)

# Summary Statistics

print("\nSummary Statistics")
print("------------------------------")

for column in ["AT", "PE"]:
    mean_value = np.mean(data[column])
    median_value = np.median(data[column])
    std_value = np.std(data[column])

    print(f"\n{column}")
    print(f"Mean: {mean_value:.2f}")
    print(f"Median: {median_value:.2f}")
    print(f"Standard Deviation: {std_value:.2f}")

# Correlation Analysis

print("\nCorrelation with PE")
print("------------------------------")

features = ["AT", "V", "AP", "RH"]

for column in features:
    correlation = np.corrcoef(
        data[column],
        data["PE"]
    )[0, 1]

    print(f"{column}: {correlation:.3f}")

# Split the dataset into training and test sets

split_index = 7654

train_data = data[:split_index]
test_data = data[split_index:]

print("\nTraining and Test Split")
print("------------------------------")

print("Training rows:", len(train_data))
print("Test rows:", len(test_data))
print("Total rows:", len(train_data) + len(test_data))

# Baseline Model

print("\nBaseline Model")
print("------------------------------")

# Calculate the training mean of PE
training_mean = np.mean(train_data["PE"])

# Predict the same value for every test row
baseline_predictions = np.full(
    len(test_data),
    training_mean
)

# Calculate test MAE
baseline_mae = np.mean(
    np.abs(test_data["PE"] - baseline_predictions)
)

print(f"Training mean of PE: {training_mean:.2f} MW")
print(f"Baseline test MAE: {baseline_mae:.2f} MW")

# Linear Regression using Exhaust Vacuum (V)

print("\nLinear Regression with V")
print("------------------------------")

# Fit the linear model using training data
slope, intercept = np.polyfit(
    train_data["V"],
    train_data["PE"],
    1
)

# Predict PE using test data
v_predictions = (
    slope * test_data["V"] + intercept
)

# Calculate test MAE
v_mae = np.mean(
    np.abs(test_data["PE"] - v_predictions)
)

print(f"Feature: Exhaust Vacuum (cm Hg)")
print(f"Slope: {slope:.4f} MW per cm Hg")
print(f"Intercept: {intercept:.2f} MW")
print(f"Test MAE: {v_mae:.2f} MW")

# Linear Regression using Ambient Temperature (AT)

print("\nLinear Regression with AT")
print("------------------------------")

# Fit the linear model using training data
at_slope, at_intercept = np.polyfit(
    train_data["AT"],
    train_data["PE"],
    1
)

# Predict PE using test data
at_predictions = (
    at_slope * test_data["AT"] + at_intercept
)

# Calculate test MAE
at_mae = np.mean(
    np.abs(test_data["PE"] - at_predictions)
)

print(f"Slope: {at_slope:.4f} MW per °C")
print(f"Intercept: {at_intercept:.2f} MW")
print(f"Test MAE: {at_mae:.2f} MW")

# Model Comparison

print("\nModel Comparison")
print("------------------------------")

print(f"Baseline: {baseline_mae:.2f} MW")
print(f"Linear Regression (AT): {at_mae:.2f} MW")
print(f"Linear Regression (V): {v_mae:.2f} MW")

# Manual Prediction Verification

print("\nManual Prediction Verification")
print("------------------------------")

row_number = 7656

feature_value = test_data["V"][0]
true_pe = test_data["PE"][0]

predicted_pe = slope * feature_value + intercept
absolute_error = abs(true_pe - predicted_pe)

print("Row number:", row_number)
print(f"Feature value: {feature_value:.2f} cm Hg")
print(f"Predicted PE: {predicted_pe:.2f} MW")
print(f"True PE: {true_pe:.2f} MW")
print(f"Absolute error: {absolute_error:.2f} MW")
