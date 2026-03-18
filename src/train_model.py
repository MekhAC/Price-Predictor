import os, glob
import numpy as np
import joblib
import pandas as pd
from math import sqrt
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor

from data_preprocessing import load_and_clean, build_preprocessor, prepare_model_input, is_native_lightgbm_preprocessor

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def price_space_mape_from_log(y_true_log: np.ndarray, y_pred_log: np.ndarray):
    """Compute MAPE in original rupee space while training on log1p(target)."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    denom = np.clip(np.abs(y_true), 1.0, None)
    mape = np.mean(np.abs(y_pred - y_true) / denom)
    return 'mape_price_space', float(mape), False

files = (
    glob.glob(os.path.join(DATA_DIR, 'Cars24_*.*')) +
    glob.glob(os.path.join(DATA_DIR, 'Spinny_*.*'))
)
if not files:
    raise RuntimeError(f"No data files found in {DATA_DIR}")

print(f"🔍 Training on {len(files)} files…")
df = pd.concat([load_and_clean(f) for f in files], ignore_index=True)

pre_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
if os.path.exists(pre_path):
    meta = joblib.load(pre_path)
    if not is_native_lightgbm_preprocessor(meta):
        build_preprocessor(df)
        meta = joblib.load(pre_path)
    numeric_feats = meta['numeric_feats']
    categorical_feats = meta['categorical_feats']
else:
    build_preprocessor(df)  # also saves
    meta = joblib.load(pre_path)
    numeric_feats = meta['numeric_feats']
    categorical_feats = meta['categorical_feats']

FEATURES = numeric_feats + categorical_feats
X = prepare_model_input(df[FEATURES], meta)
y = df['Price (₹)'].astype(float)

# ---- Split indices first to keep alignment for X, y ----
# Keep a true holdout test split separate from early-stopping validation.
idx = np.arange(len(y))
idx_train, idx_holdout = train_test_split(idx, test_size=0.2, random_state=42)
idx_val, idx_test = train_test_split(idx_holdout, test_size=0.5, random_state=42)

X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
y_train, y_val, y_test = y.iloc[idx_train], y.iloc[idx_val], y.iloc[idx_test]

# ---- Model ----
base_lgb = LGBMRegressor(
    n_estimators=4000,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=127,
    min_child_samples=15,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=0.3,
    min_split_gain=0.0,
    max_bin=255,
    metric='None',
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)
model = TransformedTargetRegressor(
    regressor=base_lgb, func=np.log1p, inverse_func=np.expm1
)

# model.fit(X_train, y_train, regressor__sample_weight=sw_train)
# Eval set y must be log-transformed to match TransformedTargetRegressor's internal target space.
y_val_log = np.log1p(y_val.values)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val_log)],
    eval_metric=price_space_mape_from_log,
    callbacks=[
        lgb_early_stopping(stopping_rounds=250, first_metric_only=True, verbose=False),
        lgb_log_evaluation(period=-1),
    ]
)

# ---- Eval ----
y_pred = model.predict(X_test)
mse  = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2   = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print(f"🔎 MSE: {mse:,.0f} ₹")
print(f"🔎 RMSE: {rmse:,.0f} ₹")
print(f"🔎 R²  : {r2:.4f}")
print(f"🔎 MAPE: {mape:.2f}%")

out_model = os.path.join(MODELS_DIR, 'price_model.joblib')
joblib.dump(model, out_model)
print(f"✅ Model saved to {out_model}")
