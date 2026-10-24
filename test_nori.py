from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from synthefy_nori import NoriRegressor
import numpy as np

# Load the same dataset
X, y = load_diabetes(return_X_y=True)

# Use the same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=21
)

# Use the downloaded Nori 1.5 model
print("Loading Nori 1.5...", flush=True)

nori = NoriRegressor(model="nori-1-5")

# Provide training examples
nori.fit(X_train, y_train)

# Predict unseen data
print("Predicting with Nori...", flush=True)

predictions = nori.predict(X_test)

# Evaluate results
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n--- NORI 1.5 RESULTS ---")
print("Test MAE:", round(mae, 2))
print("Test RMSE:", round(rmse, 2))
print("Test R2:", round(r2, 3))

# STEP 26: Nori 5-Fold Cross-Validation

from sklearn.model_selection import KFold, cross_val_score

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nRunning Nori 5-Fold CV...", flush=True)

nori_cv_scores = cross_val_score(
    NoriRegressor(model="nori-1-5"),
    X_train,
    y_train,
    cv=cv,
    scoring="neg_mean_absolute_error",
    n_jobs=1,
    error_score="raise"
)

print("\n--- NORI CROSS-VALIDATION ---")
print("Fold MAEs:", -nori_cv_scores)
print("Average CV MAE:", round(-nori_cv_scores.mean(), 2))