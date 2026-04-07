from __future__ import annotations

import glob
import os
from math import sqrt
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from data_preprocessing import (
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    PREPROCESSOR_VERSION,
    build_feature_meta,
    keep_first_snapshot_per_listing,
    load_and_clean,
    prepare_model_input,
)
from ensemble import EnsembleModel
from market_dynamics import DYNAMICS_FEATURES, build_listing_dynamics_targets, build_market_dynamics_features

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

RECENCY_HALF_LIFE_DAYS = 90.0
RECENCY_MIN_WEIGHT = 0.35
TARGET_TRANSFORM = 'log1p_price'
SPECIALIST_SEGMENT_CONFIG = [
    {
        'column': 'Fuel',
        'values': ['DIESEL'],
        'min_train_rows': 2500,
        'min_val_rows': 150,
        'min_improvement': 0.10,
    },
    {
        'column': 'BodyType',
        'values': ['SUV', 'Sedan'],
        'min_train_rows': 2500,
        'min_val_rows': 150,
        'min_improvement': 0.05,
    },
]
SPECIALIST_BLEND_STEPS = np.arange(0.10, 1.00, 0.10)


def price_space_mape_from_log(y_true_log: Optional[np.ndarray], y_pred_log: np.ndarray):
    """Compute MAPE in original rupee space while training on log1p(target)."""
    assert y_true_log is not None
    y_true = _from_log_price(y_true_log)
    y_pred = _from_log_price(y_pred_log)
    denom = np.clip(np.abs(y_true), 1.0, None)
    mape = np.mean(np.abs(y_pred - y_true) / denom)
    return 'mape_price_space', float(mape), False


def _to_log_price(values) -> np.ndarray:
    return np.log1p(np.asarray(values, dtype=float))


def _from_log_price(values) -> np.ndarray:
    return np.expm1(np.asarray(values, dtype=float))


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


def _resolve_price_column(df: pd.DataFrame) -> str:
    if 'Price (â‚¹)' in df.columns:
        return 'Price (â‚¹)'
    return next(column for column in df.columns if str(column).startswith('Price'))


def _segment_mask(X: pd.DataFrame, column: str, value: str) -> np.ndarray:
    if column not in X.columns:
        return np.zeros(len(X), dtype=bool)
    series = pd.Series(X[column], index=X.index).fillna('').astype(str).str.casefold()
    return (series == str(value).casefold()).to_numpy(dtype=bool)


def _build_specialist_model() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=0.5,
        min_split_gain=0.0,
        max_bin=255,
        metric='None',
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )


def _train_segment_specialists(
    *,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    sample_weight_train: pd.Series,
    global_ensemble: EnsembleModel,
) -> list[dict]:
    specialists: list[dict] = []
    global_val_pred = global_ensemble.predict(X_val)

    print("\nEvaluating specialist segment models...")
    for config in SPECIALIST_SEGMENT_CONFIG:
        column = str(config['column'])
        min_train_rows = int(config['min_train_rows'])
        min_val_rows = int(config['min_val_rows'])
        min_improvement = float(config['min_improvement'])

        for value in config['values']:
            train_mask = _segment_mask(X_train, column, str(value))
            val_mask = _segment_mask(X_val, column, str(value))
            train_rows = int(train_mask.sum())
            val_rows = int(val_mask.sum())

            if train_rows < min_train_rows or val_rows < min_val_rows:
                print(
                    f"Skipping specialist {column}={value}: "
                    f"train_rows={train_rows:,}, val_rows={val_rows:,}"
                )
                continue

            segment_actual = y_val.iloc[val_mask]
            global_segment_pred = global_val_pred[val_mask]
            global_segment_mape = mean_absolute_percentage_error(segment_actual, global_segment_pred) * 100

            specialist = _build_specialist_model()
            y_train_seg_log = _to_log_price(y_train.iloc[train_mask].values)
            y_val_seg_log = _to_log_price(y_val.iloc[val_mask].values)

            specialist.fit(
                X_train.iloc[train_mask],
                y_train_seg_log,
                sample_weight=sample_weight_train.iloc[train_mask],
                eval_set=[(X_val.iloc[val_mask], y_val_seg_log)],
                eval_metric=price_space_mape_from_log,
                callbacks=[
                    lgb_early_stopping(stopping_rounds=150, first_metric_only=True, verbose=False),
                    lgb_log_evaluation(period=-1),
                ],
            )

            specialist_segment_pred = _from_log_price(specialist.predict(X_val.iloc[val_mask]))
            best_blend_weight = 0.0
            best_segment_mape = float(global_segment_mape)

            for blend_weight in SPECIALIST_BLEND_STEPS:
                blended = (
                    (1.0 - float(blend_weight)) * global_segment_pred
                    + float(blend_weight) * specialist_segment_pred
                )
                blended_mape = mean_absolute_percentage_error(segment_actual, blended) * 100
                if blended_mape < best_segment_mape:
                    best_segment_mape = float(blended_mape)
                    best_blend_weight = float(blend_weight)

            improvement = float(global_segment_mape - best_segment_mape)
            if best_blend_weight > 0 and improvement >= min_improvement:
                specialists.append(
                    {
                        'column': column,
                        'value': str(value),
                        'model_name': 'lightgbm',
                        'model': specialist,
                        'blend_weight': best_blend_weight,
                        'global_segment_val_mape': float(global_segment_mape),
                        'specialist_segment_val_mape': float(best_segment_mape),
                        'improvement': improvement,
                        'train_rows': train_rows,
                        'val_rows': val_rows,
                    }
                )
                print(
                    f"Keeping specialist {column}={value}: "
                    f"val MAPE {global_segment_mape:.2f}% -> {best_segment_mape:.2f}% "
                    f"(blend_weight={best_blend_weight:.2f}, rows train/val={train_rows:,}/{val_rows:,})"
                )
            else:
                print(
                    f"Skipping specialist {column}={value}: "
                    f"no material gain over global model "
                    f"({global_segment_mape:.2f}% -> {best_segment_mape:.2f}%)"
                )

    return specialists


def _feature_importance_frame(model: EnsembleModel, meta: dict) -> pd.DataFrame:
    feature_names = list(meta['numeric_feats']) + list(meta['categorical_feats'])
    table = pd.DataFrame({'feature': feature_names})

    combined = np.zeros(len(feature_names), dtype=float)
    for name, submodel in model.models:
        raw_values = getattr(submodel, 'feature_importances_', None)
        if raw_values is None:
            raw_values = np.zeros(len(feature_names), dtype=float)
        raw_values = np.asarray(raw_values, dtype=float).ravel()

        if raw_values.size != len(feature_names) and name == 'xgboost':
            raw_values = np.zeros(len(feature_names), dtype=float)
            booster = submodel.get_booster()
            for feature_key, value in booster.get_score(importance_type='gain').items():
                if feature_key.startswith('f') and feature_key[1:].isdigit():
                    idx = int(feature_key[1:])
                    if 0 <= idx < len(feature_names):
                        raw_values[idx] = float(value)

        if raw_values.size != len(feature_names):
            raw_values = np.resize(raw_values, len(feature_names))

        total = float(raw_values.sum())
        normalized = raw_values / total if total > 0 else np.zeros(len(feature_names), dtype=float)
        combined += normalized
        table[f'{name}_importance'] = raw_values
        table[f'{name}_importance_norm'] = normalized

    table['combined_score'] = combined
    return table.sort_values('combined_score', ascending=False, kind='mergesort').reset_index(drop=True)


def _save_feature_importance(model: EnsembleModel, meta: dict, out_path: str) -> pd.DataFrame:
    importance = _feature_importance_frame(model, meta)
    importance.to_csv(out_path, index=False, encoding='utf-8-sig')
    return importance


def _collect_training_files() -> list[str]:
    files = (
        glob.glob(os.path.join(DATA_DIR, 'normalized_table.*'))
        + glob.glob(os.path.join(DATA_DIR, 'normalized_table_*.*'))
        + glob.glob(os.path.join(DATA_DIR, 'Cars24_*.*'))
        + glob.glob(os.path.join(DATA_DIR, 'Spinny_*.*'))
    )
    return [
        path
        for path in files
        if not (path.lower().endswith('.parquet') and os.path.exists(path.rsplit('.', 1)[0] + '.xlsx'))
    ]


def _build_variant_inputs(
    df: pd.DataFrame,
    *,
    use_dynamics: bool,
    target_df: pd.DataFrame,
    idx_train: np.ndarray,
):
    work = df.copy()
    numeric_feats = list(BASE_NUMERIC_FEATURES)
    dynamics_bundle = None

    if use_dynamics:
        dynamics_features, dynamics_bundle = build_market_dynamics_features(work, target_df, idx_train)
        for column in DYNAMICS_FEATURES:
            work[column] = dynamics_features[column]
        numeric_feats = numeric_feats + list(DYNAMICS_FEATURES)

    meta = build_feature_meta(
        work,
        numeric_feats=numeric_feats,
        categorical_feats=list(CATEGORICAL_FEATURES),
        preprocessor_type='lightgbm_native',
        preprocessor_version=PREPROCESSOR_VERSION,
    )
    X = prepare_model_input(work, meta)
    return meta, X, dynamics_bundle


def _train_variant(
    *,
    label: str,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
) -> dict:
    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    y_train, y_val, y_test = y.iloc[idx_train], y.iloc[idx_val], y.iloc[idx_test]
    w_train, w_val = sample_weight.iloc[idx_train], sample_weight.iloc[idx_val]

    y_train_log = _to_log_price(y_train.values)
    y_val_log = _to_log_price(y_val.values)

    print(f"\nTraining {label} LightGBM...")
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

    print(f"Training {label} XGBoost...")
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

    print(f"Optimising {label} ensemble weights...")
    lgb_val = _from_log_price(lgb.predict(X_val))
    xgb_val = _from_log_price(xgb.predict(X_val))

    best_weights, best_val_mape = None, float('inf')
    steps = np.arange(0.0, 1.05, 0.05)
    for w_lgb in steps:
        w_xgb = 1.0 - w_lgb
        blended = w_lgb * lgb_val + w_xgb * xgb_val
        mape_val = mean_absolute_percentage_error(y_val, blended) * 100
        if mape_val < best_val_mape:
            best_val_mape = mape_val
            best_weights = [w_lgb, w_xgb]

    weights = np.array(best_weights, dtype=float)
    weights = weights / weights.sum()
    ensemble = EnsembleModel(
        models=[('lightgbm', lgb), ('xgboost', xgb)],
        weights=weights.tolist(),
    )
    specialists = _train_segment_specialists(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        sample_weight_train=w_train,
        global_ensemble=ensemble,
    )
    if specialists:
        ensemble.segment_specialists = specialists

    final_val_pred = ensemble.predict(X_val)
    final_val_mape = mean_absolute_percentage_error(y_val, final_val_pred) * 100
    y_pred = ensemble.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"{label} validation MAPE: {final_val_mape:.2f}%")
    print(f"{label} test MAPE: {test_mape:.2f}%  R2: {r2:.4f}")
    if specialists:
        print(
            f"{label} specialists enabled: "
            + ", ".join(
                f"{spec['column']}={spec['value']}@{spec['blend_weight']:.2f}"
                for spec in specialists
            )
        )
    else:
        print(f"{label} specialists enabled: none")

    return {
        'label': label,
        'model': ensemble,
        'val_mape': float(final_val_mape),
        'test_mape': float(test_mape),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'weights': weights.tolist(),
        'segment_specialists': specialists,
    }


def main() -> None:
    files = _collect_training_files()
    if not files:
        raise RuntimeError(f"No data files found in {DATA_DIR}")

    print(f"Training on {len(files)} files...")
    snapshot_df = pd.concat([load_and_clean(path) for path in files], ignore_index=True)
    original_rows = len(snapshot_df)
    df = keep_first_snapshot_per_listing(snapshot_df)
    removed_rows = original_rows - len(df)
    if removed_rows > 0:
        print(f"Using earliest snapshot per listing: kept {len(df):,} rows, removed {removed_rows:,} repeats")

    price_col = _resolve_price_column(df)
    if 'Price (Ã¢â€šÂ¹)' not in df.columns:
        df['Price (Ã¢â€šÂ¹)'] = df[price_col]

    y = df[price_col].astype(float)
    sample_weight = build_recency_weights(df)

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

    idx = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(idx, test_size=0.1, random_state=42)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.1, random_state=42)

    baseline_meta, baseline_X, _ = _build_variant_inputs(
        df,
        use_dynamics=False,
        target_df=target_df,
        idx_train=idx_train,
    )
    baseline_result = _train_variant(
        label='Baseline',
        X=baseline_X,
        y=y,
        sample_weight=sample_weight,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )
    baseline_result['meta'] = baseline_meta
    baseline_result['uses_dynamics'] = False
    baseline_result['dynamics_bundle'] = None

    candidates = [baseline_result]

    if known_dynamics > 0:
        dynamics_meta, dynamics_X, dynamics_bundle = _build_variant_inputs(
            df,
            use_dynamics=True,
            target_df=target_df,
            idx_train=idx_train,
        )
        dynamics_result = _train_variant(
            label='WithDynamics',
            X=dynamics_X,
            y=y,
            sample_weight=sample_weight,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
        )
        dynamics_result['meta'] = dynamics_meta
        dynamics_result['uses_dynamics'] = True
        dynamics_result['dynamics_bundle'] = dynamics_bundle
        candidates.append(dynamics_result)
    else:
        print("Skipping dynamics variant because no listings have usable history-derived targets.")

    print("\nModel comparison:")
    for result in candidates:
        print(
            f"{result['label']:>12} -> "
            f"val MAPE: {result['val_mape']:.2f}%  "
            f"test MAPE: {result['test_mape']:.2f}%  "
            f"R2: {result['r2']:.4f}"
        )

    winner = min(candidates, key=lambda result: result['val_mape'])
    print(f"\nSelected variant: {winner['label']} (validation MAPE {winner['val_mape']:.2f}%)")

    pre_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    out_model = os.path.join(MODELS_DIR, 'price_model.joblib')
    out_summary = os.path.join(MODELS_DIR, 'training_comparison.joblib')
    out_dynamics = os.path.join(MODELS_DIR, 'market_dynamics_bundle.joblib')
    out_importance = os.path.join(MODELS_DIR, 'feature_importance.csv')

    joblib.dump(winner['meta'], pre_path)
    joblib.dump(winner['model'], out_model)
    importance = _save_feature_importance(winner['model'], winner['meta'], out_importance)
    joblib.dump(
        {
            result['label']: {
                'val_mape': result['val_mape'],
                'test_mape': result['test_mape'],
                'r2': result['r2'],
                'rmse': result['rmse'],
                'uses_dynamics': result['uses_dynamics'],
                'target_transform': TARGET_TRANSFORM,
                'segment_specialists': [
                    {
                        'column': spec['column'],
                        'value': spec['value'],
                        'blend_weight': spec['blend_weight'],
                        'global_segment_val_mape': spec['global_segment_val_mape'],
                        'specialist_segment_val_mape': spec['specialist_segment_val_mape'],
                        'improvement': spec['improvement'],
                        'train_rows': spec['train_rows'],
                        'val_rows': spec['val_rows'],
                    }
                    for spec in result.get('segment_specialists', [])
                ],
            }
            for result in candidates
        },
        out_summary,
    )

    if winner['uses_dynamics'] and winner['dynamics_bundle'] is not None:
        joblib.dump(winner['dynamics_bundle'], out_dynamics)
        print(f"Market dynamics bundle saved to {out_dynamics}")
    else:
        print("Saved baseline model without market dynamics features.")

    print(f"Preprocessor saved to {pre_path}")
    print(f"Ensemble model saved to {out_model}")
    print(f"Feature importance saved to {out_importance}")
    print(f"Training comparison saved to {out_summary}")
    print("\nTop feature importance:")
    print(importance[['feature', 'combined_score']].head(15).to_string(index=False))


if __name__ == '__main__':
    main()
