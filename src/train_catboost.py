from __future__ import annotations

import argparse
import glob
import os
from math import sqrt

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_preprocessing import build_preprocessor, load_and_clean, prepare_model_input, keep_first_snapshot_per_listing
from market_dynamics import DYNAMICS_FEATURES, build_listing_dynamics_targets, build_market_dynamics_features

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

RECENCY_HALF_LIFE_DAYS = 90.0
RECENCY_MIN_WEIGHT = 0.35
DEFAULT_TRIALS = 25
DEFAULT_SAMPLE_SIZE = 200_000
TARGET_TRANSFORM = 'log1p_price'


def _to_log_price(values) -> np.ndarray:
    return np.log1p(np.asarray(values, dtype=float))


def _from_log_price(values) -> np.ndarray:
    return np.expm1(np.asarray(values, dtype=float))


def build_recency_weights(df: pd.DataFrame) -> pd.Series:
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


def _resolve_price_column(df: pd.DataFrame) -> str:
    if 'Price (₹)' in df.columns:
        return 'Price (₹)'
    return next(column for column in df.columns if str(column).startswith('Price'))


def _to_catboost_df(frame: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for col in cat_cols:
        if col in frame.columns:
            frame[col] = frame[col].astype(str)
    return frame


def _build_catboost_params(trial: optuna.Trial, task_type: str, devices: str) -> dict:
    bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS'])
    params = {
        'iterations': trial.suggest_int('iterations', 1500, 4500, step=500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 3.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 64),
        'border_count': trial.suggest_categorical('border_count', [128, 254]),
        'one_hot_max_size': trial.suggest_categorical('one_hot_max_size', [2, 8, 16]),
        'bootstrap_type': bootstrap_type,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 200,
        'random_seed': 42,
        'verbose': 0,
        'allow_writing_files': False,
        'task_type': task_type,
    }
    if task_type == 'GPU':
        params['devices'] = devices
    else:
        params['thread_count'] = -1
        params['rsm'] = trial.suggest_float('rsm', 0.6, 1.0)

    if bootstrap_type == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 10.0)
    else:
        params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)

    return params


def _load_training_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
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
        raise RuntimeError(f'No data files found in {DATA_DIR}')

    print(f'Loading {len(files)} files...')
    snapshot_df = pd.concat([load_and_clean(path) for path in files], ignore_index=True)
    original_rows = len(snapshot_df)
    df = keep_first_snapshot_per_listing(snapshot_df)
    removed_rows = original_rows - len(df)
    if removed_rows > 0:
        print(f'Using earliest snapshot per listing: kept {len(df):,} rows, removed {removed_rows:,} repeats')

    price_col = _resolve_price_column(df)
    y = df[price_col].astype(float)
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
    print(f'History-derived dynamics targets available for {int(target_df.notna().all(axis=1).sum()):,} of {len(df):,} listings')
    return df, y, target_df, build_recency_weights(df)


def main() -> None:
    parser = argparse.ArgumentParser(description='Tune and train a standalone CatBoost price model.')
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS, help='Optuna trials for CatBoost tuning')
    parser.add_argument(
        '--sample-size',
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help='Maximum train/validation rows used during tuning',
    )
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu', help='CatBoost training device')
    parser.add_argument('--gpu-devices', default='0', help='GPU device string passed to CatBoost when using GPU')
    args = parser.parse_args()

    task_type = 'GPU' if args.device == 'gpu' else 'CPU'
    df, y, target_df, sample_weight = _load_training_data()

    fetched = pd.to_datetime(df['Fetched On'], errors='coerce') if 'Fetched On' in df.columns else pd.Series(pd.NaT, index=df.index)
    if fetched.notna().any():
        latest_snapshot = fetched.max()
        assert latest_snapshot is not None
        print(
            'Recency weighting enabled: '
            f'half-life={RECENCY_HALF_LIFE_DAYS:.0f}d, min_weight={RECENCY_MIN_WEIGHT:.2f}, '
            f'latest_snapshot={latest_snapshot.date()}'
        )
    print(
        'Sample weight range: '
        f'{sample_weight.min():.3f} to {sample_weight.max():.3f} '
        f'(mean {sample_weight.mean():.3f})'
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
    cat_cols = meta['categorical_feats']
    features = meta['numeric_feats'] + meta['categorical_feats']
    X = prepare_model_input(df[features], meta)

    X_train = X.iloc[idx_train]
    X_val = X.iloc[idx_val]
    X_test = X.iloc[idx_test]
    y_train = y.iloc[idx_train]
    y_val = y.iloc[idx_val]
    y_test = y.iloc[idx_test]
    w_train = sample_weight.iloc[idx_train]
    w_val = sample_weight.iloc[idx_val]

    y_train_log = _to_log_price(y_train.values)
    y_val_log = _to_log_price(y_val.values)

    tune_idx = idx_trainval
    if len(tune_idx) > args.sample_size:
        rng = np.random.RandomState(42)
        tune_idx = rng.choice(tune_idx, size=args.sample_size, replace=False)

    tune_train_idx, tune_val_idx = train_test_split(tune_idx, test_size=0.15, random_state=42)
    X_tune_train = _to_catboost_df(X.iloc[tune_train_idx], cat_cols)
    X_tune_val = _to_catboost_df(X.iloc[tune_val_idx], cat_cols)
    y_tune_train_log = _to_log_price(y.iloc[tune_train_idx].values)
    y_tune_val = y.iloc[tune_val_idx].values
    y_tune_val_log = _to_log_price(y_tune_val)
    w_tune_train = sample_weight.iloc[tune_train_idx].to_numpy()
    w_tune_val = sample_weight.iloc[tune_val_idx].to_numpy()

    print(
        f'Tuning CatBoost on {len(tune_train_idx):,} train rows and {len(tune_val_idx):,} validation rows '
        f'for {args.trials} trials...'
    )

    def objective(trial: optuna.Trial) -> float:
        params = _build_catboost_params(trial, task_type=task_type, devices=args.gpu_devices)
        model = CatBoostRegressor(**params)
        train_pool = Pool(X_tune_train, y_tune_train_log, cat_features=cat_cols, weight=w_tune_train)
        val_pool = Pool(X_tune_val, y_tune_val_log, cat_features=cat_cols, weight=w_tune_val)
        model.fit(train_pool, eval_set=val_pool)
        pred = _from_log_price(model.predict(X_tune_val))
        return float(mean_absolute_percentage_error(y_tune_val, pred))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', study_name='catboost_price')
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best_params = _build_catboost_params(study.best_trial, task_type=task_type, devices=args.gpu_devices)
    print(f'\nBest CatBoost validation MAPE: {study.best_value * 100:.2f}%')
    print('Best CatBoost params:')
    for key, value in sorted(best_params.items()):
        print(f'  {key}: {value}')

    final_model = CatBoostRegressor(**best_params)
    X_train_cb = _to_catboost_df(X_train, cat_cols)
    X_val_cb = _to_catboost_df(X_val, cat_cols)
    X_test_cb = _to_catboost_df(X_test, cat_cols)
    train_pool = Pool(X_train_cb, y_train_log, cat_features=cat_cols, weight=w_train.to_numpy())
    val_pool = Pool(X_val_cb, y_val_log, cat_features=cat_cols, weight=w_val.to_numpy())
    final_model.fit(train_pool, eval_set=val_pool)

    pred = _from_log_price(final_model.predict(X_test_cb))
    mse = mean_squared_error(y_test, pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, pred)
    mape = mean_absolute_percentage_error(y_test, pred) * 100

    print('\nCatBoost test metrics:')
    print(f'MSE :  {mse:,.0f} Rs')
    print(f'RMSE: {rmse:,.0f} Rs')
    print(f'R2  : {r2:.4f}')
    print(f'MAPE: {mape:.2f}%')

    model_path = os.path.join(MODELS_DIR, 'catboost_price_model.joblib')
    params_path = os.path.join(MODELS_DIR, 'catboost_tuned_params.joblib')
    dynamics_path = os.path.join(MODELS_DIR, 'market_dynamics_bundle.joblib')
    importance_path = os.path.join(MODELS_DIR, 'catboost_feature_importance.csv')
    summary_path = os.path.join(MODELS_DIR, 'catboost_training_summary.joblib')
    importance = pd.DataFrame(
        {
            'feature': features,
            'importance': final_model.get_feature_importance(train_pool),
        }
    ).sort_values('importance', ascending=False, kind='mergesort')
    joblib.dump(final_model, model_path)
    joblib.dump(best_params, params_path)
    joblib.dump(dynamics_bundle, dynamics_path)
    joblib.dump(
        {
            'target_transform': TARGET_TRANSFORM,
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
        },
        summary_path,
    )
    importance.to_csv(importance_path, index=False, encoding='utf-8-sig')
    print(f'\nStandalone CatBoost model saved to {model_path}')
    print(f'CatBoost tuned params saved to {params_path}')
    print(f'Market dynamics bundle saved to {dynamics_path}')
    print(f'CatBoost feature importance saved to {importance_path}')
    print(f'CatBoost training summary saved to {summary_path}')


if __name__ == '__main__':
    main()
