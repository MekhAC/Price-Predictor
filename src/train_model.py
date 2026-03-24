from __future__ import annotations
import os, glob
import numpy as np
import joblib
import pandas as pd
from math import sqrt
from typing import Optional
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, KFold

from data_preprocessing import load_and_clean, build_preprocessor, prepare_model_input, is_native_lightgbm_preprocessor
from ensemble import EnsembleModel

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def price_space_mape_from_log(y_true_log: Optional[np.ndarray], y_pred_log: np.ndarray):
    """Compute MAPE in original rupee space while training on log1p(target)."""
    assert y_true_log is not None
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    denom = np.clip(np.abs(y_true), 1.0, None)
    mape = np.mean(np.abs(y_pred - y_true) / denom)
    return 'mape_price_space', float(mape), False

files = (
    glob.glob(os.path.join(DATA_DIR, 'normalized_table_*.*')) +
    glob.glob(os.path.join(DATA_DIR, 'Cars24_*.*')) +
    glob.glob(os.path.join(DATA_DIR, 'Spinny_*.*'))
)
# Exclude auto-generated .parquet caches when the .xlsx source also exists
files = [f for f in files if not (f.lower().endswith('.parquet') and os.path.exists(f.rsplit('.', 1)[0] + '.xlsx'))]
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

# ---- Split: hold out 10% as a true test set, use rest for training ----
idx = np.arange(len(y))
idx_trainval, idx_test = train_test_split(idx, test_size=0.1, random_state=42)
X_test, y_test = X.iloc[idx_test], y.iloc[idx_test]

# Split trainval into train + early-stop validation
idx_train, idx_val = train_test_split(idx_trainval, test_size=0.1, random_state=42)
X_train, X_val = X.iloc[idx_train], X.iloc[idx_val]
y_train, y_val = y.iloc[idx_train], y.iloc[idx_val]

# All base models are trained on log1p(y) directly (no TransformedTargetRegressor)
y_train_log = np.log1p(y_train.values)
y_val_log   = np.log1p(y_val.values)

# ---- 1. LightGBM ----
print("\n🌲 Training LightGBM …")
lgb = LGBMRegressor(
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
lgb.fit(
    X_train, y_train_log,
    eval_set=[(X_val, y_val_log)],
    eval_metric=price_space_mape_from_log,
    callbacks=[
        lgb_early_stopping(stopping_rounds=250, first_metric_only=True, verbose=False),
        lgb_log_evaluation(period=-1),
    ],
)
lgb_pred = np.expm1(np.asarray(lgb.predict(X_test)))
lgb_mape = mean_absolute_percentage_error(y_test, lgb_pred) * 100
lgb_r2   = r2_score(y_test, lgb_pred)
print(f"   LightGBM  → MAPE: {lgb_mape:.2f}%  R²: {lgb_r2:.4f}")

# ---- 2. XGBoost ----
print("\n🌲 Training XGBoost …")
xgb = XGBRegressor(
    n_estimators=4000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.5,
    tree_method='hist',
    device='cuda',
    enable_categorical=True,
    early_stopping_rounds=250,
    eval_metric='rmse',
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)
xgb.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)
xgb_pred = np.expm1(xgb.predict(X_test))
xgb_mape = mean_absolute_percentage_error(y_test, xgb_pred) * 100
xgb_r2   = r2_score(y_test, xgb_pred)
print(f"   XGBoost   → MAPE: {xgb_mape:.2f}%  R²: {xgb_r2:.4f}")

# ---- 3. CatBoost ----
# CatBoost needs string categoricals, not pd.Categorical
print("\n🌲 Training CatBoost …")
cat_cols = meta['categorical_feats']

def _to_catboost_df(df):
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

X_train_cb = _to_catboost_df(X_train)
X_val_cb   = _to_catboost_df(X_val)

cb = CatBoostRegressor(
    iterations=4000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3.0,
    bootstrap_type='MVS',
    subsample=0.85,
    cat_features=cat_cols,
    task_type='GPU',
    devices='0',
    early_stopping_rounds=250,
    eval_metric='RMSE',
    random_seed=42,
    verbose=0,
)
cb.fit(X_train_cb, y_train_log, eval_set=(X_val_cb, y_val_log))
cb_pred = np.expm1(cb.predict(_to_catboost_df(X_test)))
cb_mape = mean_absolute_percentage_error(y_test, cb_pred) * 100
cb_r2   = r2_score(y_test, cb_pred)
print(f"   CatBoost  → MAPE: {cb_mape:.2f}%  R²: {cb_r2:.4f}")

# ---- Optimise ensemble weights on the validation set ----
print("\n⚖️  Optimising ensemble weights …")
lgb_val = np.expm1(np.asarray(lgb.predict(X_val)))
xgb_val = np.expm1(np.asarray(xgb.predict(X_val)))
cb_val  = np.expm1(cb.predict(X_val_cb))

best_weights, best_mape_w = None, float('inf')
steps = np.arange(0.0, 1.05, 0.05)
for w1 in steps:
    for w2 in steps:
        w3 = 1.0 - w1 - w2
        if w3 < -0.01:
            continue
        w3 = max(w3, 0.0)
        blended = w1 * lgb_val + w2 * xgb_val + w3 * cb_val
        m = mean_absolute_percentage_error(y_val, blended) * 100
        if m < best_mape_w:
            best_mape_w = m
            best_weights = [w1, w2, w3]

# Normalise
ws = np.array(best_weights)
ws = ws / ws.sum()
best_weights = ws.tolist()
print(f"   Best weights → LGB: {best_weights[0]:.2f}  XGB: {best_weights[1]:.2f}  CB: {best_weights[2]:.2f}  (val MAPE: {best_mape_w:.2f}%)")

# ---- Build ensemble ----
model = EnsembleModel(
    models=[('lightgbm', lgb), ('xgboost', xgb), ('catboost', cb)],
    cat_features=cat_cols,
    weights=best_weights,
)

# ---- Eval on held-out test set ----
y_pred = model.predict(X_test)
mse  = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2   = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print(f"\n🏆 Ensemble test metrics:")
print(f"🔎 MSE:  {mse:,.0f} ₹")
print(f"🔎 RMSE: {rmse:,.0f} ₹")
print(f"🔎 R²  : {r2:.4f}")
print(f"🔎 MAPE: {mape:.2f}%")

out_model = os.path.join(MODELS_DIR, 'price_model.joblib')
joblib.dump(model, out_model)
print(f"\n✅ Ensemble model saved to {out_model}")
