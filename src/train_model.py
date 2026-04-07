from __future__ import annotations
import os, glob
import numpy as np
import joblib
import pandas as pd
from math import sqrt
from typing import Optional
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from data_preprocessing import load_and_clean, build_preprocessor, prepare_model_input, keep_first_snapshot_per_listing
from ensemble import EnsembleModel
from market_dynamics import DYNAMICS_FEATURES, build_listing_dynamics_targets, build_market_dynamics_features
from price_model_training import main as _selected_training_main

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

RECENCY_HALF_LIFE_DAYS = 90.0
RECENCY_MIN_WEIGHT = 0.35

def price_space_mape_from_log(y_true_log: Optional[np.ndarray], y_pred_log: np.ndarray):
    """Compute MAPE in original rupee space while training on log1p(target)."""
    assert y_true_log is not None
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    denom = np.clip(np.abs(y_true), 1.0, None)
    mape = np.mean(np.abs(y_pred - y_true) / denom)
    return 'mape_price_space', float(mape), False


def build_recency_weights(df: pd.DataFrame) -> pd.Series:
    """Down-weight older snapshots while keeping all rows in training."""
    if 'Fetched On' not in df.columns:
        return pd.Series(np.ones(len(df), dtype=float), index=df.index)

    fetched_on = pd.to_datetime(df['Fetched On'], errors='coerce')
    if not fetched_on.notna().any():
        return pd.Series(np.ones(len(df), dtype=float), index=df.index)

    latest_snapshot = fetched_on.max()
    assert latest_snapshot is not None

    age_days = (latest_snapshot - fetched_on).dt.days.clip(lower=0)
    oldest_age = float(age_days[fetched_on.notna()].max())
    age_days = age_days.fillna(oldest_age)

    weights = np.exp(-np.log(2.0) * age_days / RECENCY_HALF_LIFE_DAYS)
    weights = np.clip(weights, RECENCY_MIN_WEIGHT, 1.0)
    return pd.Series(weights.astype(float), index=df.index)
def main() -> None:
    _selected_training_main()
    return
    files = (
        glob.glob(os.path.join(DATA_DIR, 'normalized_table.*'))
        + glob.glob(os.path.join(DATA_DIR, 'normalized_table_*.*'))
        + glob.glob(os.path.join(DATA_DIR, 'Cars24_*.*'))
        + glob.glob(os.path.join(DATA_DIR, 'Spinny_*.*'))
    )
    files = [
        path
        for path in files
        if not (path.lower().endswith('.parquet') and os.path.exists(path.rsplit('.', 1)[0] + '.xlsx'))
    ]
    if not files:
        raise RuntimeError(f"No data files found in {DATA_DIR}")

    print(f"Training on {len(files)} files...")
    snapshot_df = pd.concat([load_and_clean(path) for path in files], ignore_index=True)
    original_rows = len(snapshot_df)
    df = keep_first_snapshot_per_listing(snapshot_df)
    removed_rows = original_rows - len(df)
    if removed_rows > 0:
        print(f"Using earliest snapshot per listing: kept {len(df):,} rows, removed {removed_rows:,} repeats")
    price_col = 'Price (₹)' if 'Price (₹)' in df.columns else next(c for c in df.columns if c.startswith('Price'))
    if 'Price (â‚¹)' not in df.columns:
        df['Price (â‚¹)'] = df[price_col]

    y = df[price_col].astype(float)
    sample_weight = build_recency_weights(df)
    dynamics_targets = build_listing_dynamics_targets(snapshot_df)
    target_df = pd.DataFrame(index=df.index, columns=DYNAMICS_FEATURES, dtype=float)
    if 'ID' in df.columns and not dynamics_targets.empty:
        target_df = df[['ID']].copy()
        target_df['_row_order'] = np.arange(len(target_df))
        target_df['ID'] = target_df['ID'].astype('string').str.strip()
        target_df = (
            target_df.merge(dynamics_targets, on='ID', how='left')
            .sort_values('_row_order', kind='mergesort')
            .drop(columns=['ID', '_row_order'], errors='ignore')
            .reset_index(drop=True)
        )
    known_dynamics = int(target_df.notna().all(axis=1).sum())
    print(f"History-derived dynamics targets available for {known_dynamics:,} of {len(df):,} listings")

    if 'Fetched On' in df.columns and pd.to_datetime(df['Fetched On'], errors='coerce').notna().any():
        latest_snapshot = pd.to_datetime(df['Fetched On'], errors='coerce').max()
        print(
            "Recency weighting enabled: "
            f"half-life={RECENCY_HALF_LIFE_DAYS:.0f}d, min_weight={RECENCY_MIN_WEIGHT:.2f}, "
            f"latest_snapshot={latest_snapshot.date()}"
        )
    print(
        "Sample weight range: "
        f"{sample_weight.min():.3f} to {sample_weight.max():.3f} "
        f"(mean {sample_weight.mean():.3f})"
    )

    idx = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(idx, test_size=0.1, random_state=42)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.1, random_state=42)
    dynamics_features, dynamics_bundle = build_market_dynamics_features(df, target_df, idx_train)
    df = df.copy()
    for column in DYNAMICS_FEATURES:
        df[column] = dynamics_features[column]

    pre_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    build_preprocessor(df)
    meta = joblib.load(pre_path)
    numeric_feats = meta['numeric_feats']
    categorical_feats = meta['categorical_feats']
    features = numeric_feats + categorical_feats
    X = prepare_model_input(df[features], meta)

    X_train, X_val = X.iloc[idx_train], X.iloc[idx_val]
    X_test = X.iloc[idx_test]
    y_train, y_val = y.iloc[idx_train], y.iloc[idx_val]
    y_test = y.iloc[idx_test]
    w_train, w_val = sample_weight.iloc[idx_train], sample_weight.iloc[idx_val]

    y_train_log = np.log1p(y_train.values)
    y_val_log = np.log1p(y_val.values)

    print("\nTraining LightGBM...")
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
        X_train,
        y_train_log,
        sample_weight=w_train,
        eval_set=[(X_val, y_val_log)],
        eval_sample_weight=[w_val],
        eval_metric=price_space_mape_from_log,
        callbacks=[
            lgb_early_stopping(stopping_rounds=250, first_metric_only=True, verbose=False),
            lgb_log_evaluation(period=-1),
        ],
    )
    lgb_pred = np.expm1(np.asarray(lgb.predict(X_test)))
    lgb_mape = mean_absolute_percentage_error(y_test, lgb_pred) * 100
    lgb_r2 = r2_score(y_test, lgb_pred)
    print(f"   LightGBM  -> MAPE: {lgb_mape:.2f}%  R2: {lgb_r2:.4f}")

    print("\nTraining XGBoost...")
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
    xgb.fit(
        X_train,
        y_train_log,
        sample_weight=w_train,
        eval_set=[(X_val, y_val_log)],
        sample_weight_eval_set=[w_val],
        verbose=False,
    )
    xgb_pred = np.expm1(xgb.predict(X_test))
    xgb_mape = mean_absolute_percentage_error(y_test, xgb_pred) * 100
    xgb_r2 = r2_score(y_test, xgb_pred)
    print(f"   XGBoost   -> MAPE: {xgb_mape:.2f}%  R2: {xgb_r2:.4f}")

    print("\nOptimising ensemble weights...")
    lgb_val = np.expm1(np.asarray(lgb.predict(X_val)))
    xgb_val = np.expm1(np.asarray(xgb.predict(X_val)))

    best_weights, best_mape_w = None, float('inf')
    steps = np.arange(0.0, 1.05, 0.05)
    for w_lgb in steps:
        w_xgb = 1.0 - w_lgb
        blended = w_lgb * lgb_val + w_xgb * xgb_val
        mape_val = mean_absolute_percentage_error(y_val, blended) * 100
        if mape_val < best_mape_w:
            best_mape_w = mape_val
            best_weights = [w_lgb, w_xgb]

    ws = np.array(best_weights)
    ws = ws / ws.sum()
    best_weights = ws.tolist()
    print(
        "   Best weights -> "
        f"LGB: {best_weights[0]:.2f}  XGB: {best_weights[1]:.2f}  "
        f"(val MAPE: {best_mape_w:.2f}%)"
    )

    model = EnsembleModel(
        models=[('lightgbm', lgb), ('xgboost', xgb)],
        weights=best_weights,
    )

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print("\nEnsemble test metrics:")
    print(f"MSE :  {mse:,.0f} Rs")
    print(f"RMSE: {rmse:,.0f} Rs")
    print(f"R2  : {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    out_model = os.path.join(MODELS_DIR, 'price_model.joblib')
    out_dynamics = os.path.join(MODELS_DIR, 'market_dynamics_bundle.joblib')
    joblib.dump(model, out_model)
    joblib.dump(dynamics_bundle, out_dynamics)
    print(f"\nEnsemble model saved to {out_model}")
    print(f"Market dynamics bundle saved to {out_dynamics}")


if __name__ == '__main__':
    main()
