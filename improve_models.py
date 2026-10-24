
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    KFold,
    cross_val_score
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=21
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# STEP 2: SET UP CROSS-VALIDATION
# ==========================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# STEP 3: CREATE EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )
    r2 = r2_score(y_test, predictions)

    print(f"\n--- {name} ---")
    print("Test MAE:", round(mae, 2))
    print("Test RMSE:", round(rmse, 2))
    print("Test R2:", round(r2, 3))

    return {
        "Model": name,
        "Test_MAE": mae,
        "Test_RMSE": rmse,
        "Test_R2": r2
    }


# ==========================================
# STEP 4: ORIGINAL RANDOM FOREST
# ==========================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    n_jobs=1
)

rf_cv_scores = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv=cv,
    scoring="neg_mean_absolute_error"
)

print(
    "\nOriginal Random Forest CV MAE:",
    round(-rf_cv_scores.mean(), 2)
)

rf_model.fit(X_train, y_train)

evaluate_model(
    "Original Random Forest",
    rf_model,
    X_test,
    y_test
)


# ==========================================
# STEP 5: HYPERPARAMETER TUNING
# ==========================================

parameters = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 8, None],
    "min_samples_leaf": [1, 3, 5]
}

rf_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(
        random_state=42,
        n_jobs=1
    ),
    param_distributions=parameters,
    n_iter=12,
    cv=cv,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=2
)

print("\nTuning Random Forest...", flush=True)

rf_search.fit(X_train, y_train)

print("\n--- IMPROVED RANDOM FOREST ---")
print("Best parameters:", rf_search.best_params_)
print(
    "Best CV MAE:",
    round(-rf_search.best_score_, 2)
)

best_rf = rf_search.best_estimator_

evaluate_model(
    "Improved Random Forest",
    best_rf,
    X_test,
    y_test
)


# ==========================================
# STEP 6: XGBOOST
# ==========================================

xgb_model = XGBRegressor(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=2
)

xgb_cv_scores = cross_val_score(
    xgb_model,
    X_train,
    y_train,
    cv=cv,
    scoring="neg_mean_absolute_error"
)

print(
    "\nXGBoost CV MAE:",
    round(-xgb_cv_scores.mean(), 2)
)

xgb_model.fit(X_train, y_train)

evaluate_model(
    "XGBoost",
    xgb_model,
    X_test,
    y_test
)


# ==========================================
# STEP 7: LIGHTGBM
# ==========================================

lgbm_model = LGBMRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    num_leaves=7,
    min_child_samples=10,
    random_state=42,
    n_jobs=2,
    verbosity=-1
)

lgbm_cv_scores = cross_val_score(
    lgbm_model,
    X_train,
    y_train,
    cv=cv,
    scoring="neg_mean_absolute_error"
)

print(
    "\nLightGBM CV MAE:",
    round(-lgbm_cv_scores.mean(), 2)
)

lgbm_model.fit(X_train, y_train)

evaluate_model(
    "LightGBM",
    lgbm_model,
    X_test,
    y_test
)


# ==========================================
# STEP 8: FINISH
# ==========================================

print("\n====================================")
print("Traditional ML Benchmark Completed!")
print("====================================")
