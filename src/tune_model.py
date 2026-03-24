"""
Optuna-based hyperparameter tuning for LightGBM + XGBoost ensemble.
Run:  python src/tune_model.py
"""
import os, glob, sys, json
import numpy as np
import pandas as pd
import joblib
import optuna
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import load_and_clean, build_preprocessor, prepare_model_input, is_native_lightgbm_preprocessor

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

N_TRIALS = 30
N_FOLDS  = 2          # 2-fold for speed on large datasets
SAMPLE_SIZE = 300_000  # subsample for tuning speed; final train uses all data

# ── Load data ──────────────────────────────────────────────────────────────────
files = (
    glob.glob(os.path.join(DATA_DIR, 'normalized_table_*.*')) +
    glob.glob(os.path.join(DATA_DIR, 'Cars24_*.*')) +
    glob.glob(os.path.join(DATA_DIR, 'Spinny_*.*'))
)
# Exclude auto-generated .parquet caches when the .xlsx source also exists
files = [f for f in files if not (f.lower().endswith('.parquet') and os.path.exists(f.rsplit('.', 1)[0] + '.xlsx'))]
if not files:
    raise RuntimeError(f"No data files found in {DATA_DIR}")

print(f"Loading {len(files)} files...")
df = pd.concat([load_and_clean(f) for f in files], ignore_index=True)

pre_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
if os.path.exists(pre_path):
    meta = joblib.load(pre_path)
    if not is_native_lightgbm_preprocessor(meta):
        build_preprocessor(df)
        meta = joblib.load(pre_path)
else:
    build_preprocessor(df)
    meta = joblib.load(pre_path)

FEATURES = meta['numeric_feats'] + meta['categorical_feats']
X_full = prepare_model_input(df[FEATURES], meta)
y_full = df['Price (\u20b9)'].astype(float).values
print(f"Full dataset: {len(y_full):,} rows, {len(FEATURES)} features")

# Subsample for tuning speed
if len(y_full) > SAMPLE_SIZE:
    rng = np.random.RandomState(42)
    idx = rng.choice(len(y_full), size=SAMPLE_SIZE, replace=False)
    X = X_full.iloc[idx].reset_index(drop=True)
    y = y_full[idx]
    print(f"Subsampled to {SAMPLE_SIZE:,} rows for tuning")
else:
    X = X_full
    y = y_full
print()


def _mape_eval_lgb(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    denom = np.clip(np.abs(y_true), 1.0, None)
    mape = np.mean(np.abs(y_pred - y_true) / denom)
    return 'mape_price_space', float(mape), False


def objective(trial):
    # ── LightGBM params ──
    lgb_params = {
        'n_estimators':      4000,
        'learning_rate':     trial.suggest_float('lgb_lr', 0.01, 0.08, log=True),
        'num_leaves':        trial.suggest_int('lgb_num_leaves', 63, 255),
        'max_depth':         trial.suggest_int('lgb_max_depth', 5, 12),
        'min_child_samples': trial.suggest_int('lgb_min_child', 5, 40),
        'subsample':         trial.suggest_float('lgb_subsample', 0.7, 1.0),
        'colsample_bytree':  trial.suggest_float('lgb_colsample', 0.6, 1.0),
        'reg_alpha':         trial.suggest_float('lgb_alpha', 1e-3, 5, log=True),
        'reg_lambda':        trial.suggest_float('lgb_lambda', 1e-3, 5, log=True),
        'min_split_gain':    trial.suggest_float('lgb_split_gain', 0.0, 0.5),
        'max_bin':           trial.suggest_categorical('lgb_max_bin', [127, 255, 511]),
        'metric':            'None',
        'n_jobs':            -1,
        'random_state':      42,
        'verbose':           -1,
    }

    # ── XGBoost params ──
    xgb_params = {
        'n_estimators':      4000,
        'learning_rate':     trial.suggest_float('xgb_lr', 0.01, 0.08, log=True),
        'max_depth':         trial.suggest_int('xgb_max_depth', 5, 10),
        'subsample':         trial.suggest_float('xgb_subsample', 0.7, 1.0),
        'colsample_bytree':  trial.suggest_float('xgb_colsample', 0.6, 1.0),
        'reg_alpha':         trial.suggest_float('xgb_alpha', 1e-3, 5, log=True),
        'reg_lambda':        trial.suggest_float('xgb_lambda', 1e-3, 5, log=True),
        'tree_method':       'hist',
        'device':            'cuda',
        'enable_categorical': True,
        'early_stopping_rounds': 150,
        'eval_metric':       'rmse',
        'n_jobs':            -1,
        'random_state':      42,
        'verbosity':         0,
    }

    # ── Ensemble weight ──
    lgb_weight = trial.suggest_float('lgb_weight', 0.2, 0.8)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    mapes = []

    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        y_tr_log, y_va_log = np.log1p(y_tr), np.log1p(y_va)

        # LightGBM
        lgb_model = LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_tr, y_tr_log,
            eval_set=[(X_va, y_va_log)],
            eval_metric=_mape_eval_lgb,
            callbacks=[
                lgb_early_stopping(stopping_rounds=150, first_metric_only=True, verbose=False),
                lgb_log_evaluation(period=-1),
            ],
        )

        # XGBoost
        xgb_model = XGBRegressor(**xgb_params)
        xgb_model.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)

        # Blend
        lgb_pred = np.expm1(np.asarray(lgb_model.predict(X_va)))
        xgb_pred = np.expm1(np.asarray(xgb_model.predict(X_va)))
        blended = lgb_weight * lgb_pred + (1 - lgb_weight) * xgb_pred

        mapes.append(mean_absolute_percentage_error(y_va, blended))

    return float(np.mean(mapes))


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', study_name='ensemble_price')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n{'='*50}")
    print(f"Best ensemble MAPE: {study.best_value * 100:.2f}%")
    print(f"{'='*50}")
    print("Best params:")
    for k, v in sorted(study.best_params.items()):
        print(f"   {k}: {v}")

    # Save to JSON so train_model.py can optionally read them
    out = os.path.join(MODELS_DIR, 'tuned_params.json')
    joblib.dump(study.best_params, out)
    print(f"\nParams saved to {out}")
    print("Copy the LGB/XGB params into train_model.py and retrain.")

