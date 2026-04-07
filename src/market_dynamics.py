from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from model_preprocessing import (
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    DYNAMICS_FEATURES,
    build_feature_meta,
    prepare_model_input,
)

SECONDARY_RANDOM_STATE = 42

_TARGET_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    'Expected Avg Drop 7d': (0.0, None),
    'Expected Avg Drop 30d': (0.0, None),
    'Expected Price Drop Rate': (0.0, None),
    'Expected Time To First Drop': (0.0, None),
    'Expected Time To Sell': (0.0, None),
    'Expected Price Volatility': (0.0, None),
    'Expected Market Liquidity Score': (0.0, 100.0),
}


class ConstantValueModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.value, dtype=float)


class MarketDynamicsBundle:
    def __init__(
        self,
        input_meta: dict,
        models: dict[str, object],
        default_values: dict[str, float],
    ) -> None:
        self.input_meta = input_meta
        self.models = models
        self.default_values = default_values
        self.target_columns = list(DYNAMICS_FEATURES)

    def predict_features(self, df: pd.DataFrame) -> pd.DataFrame:
        input_cols = self.input_meta['numeric_feats'] + self.input_meta['categorical_feats']
        X = prepare_model_input(df[input_cols] if set(input_cols).issubset(df.columns) else df, self.input_meta)
        preds = {}
        for column in self.target_columns:
            model = self.models[column]
            values = np.asarray(model.predict(X), dtype=float).ravel()
            preds[column] = _clip_target_values(column, values)
        return pd.DataFrame(preds, index=df.index)


def _resolve_price_column(df: pd.DataFrame) -> str:
    if 'Price (â‚¹)' in df.columns:
        return 'Price (â‚¹)'
    return next(column for column in df.columns if str(column).startswith('Price'))


def _clip_target_values(column: str, values: np.ndarray) -> np.ndarray:
    low, high = _TARGET_BOUNDS.get(column, (None, None))
    if low is not None:
        values = np.maximum(values, low)
    if high is not None:
        values = np.minimum(values, high)
    return values


def build_listing_dynamics_targets(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if 'ID' not in snapshot_df.columns or 'Fetched On' not in snapshot_df.columns:
        return pd.DataFrame(columns=['ID'] + list(DYNAMICS_FEATURES))

    price_col = _resolve_price_column(snapshot_df)
    work = snapshot_df.copy()
    work['_listing_id'] = work['ID'].astype('string').str.strip()
    work['_fetched_on'] = pd.to_datetime(work['Fetched On'], errors='coerce')
    work[price_col] = pd.to_numeric(work[price_col], errors='coerce')
    work = work[
        work['_listing_id'].notna()
        & (work['_listing_id'] != '')
        & work['_fetched_on'].notna()
        & work[price_col].notna()
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=['ID'] + list(DYNAMICS_FEATURES))

    work['_day'] = work['_fetched_on'].dt.floor('D')
    work['_row_order'] = np.arange(len(work))
    work = work.sort_values(
        ['_listing_id', '_day', '_fetched_on', '_row_order'],
        kind='mergesort',
    )
    work = work.drop_duplicates(subset=['_listing_id', '_day'], keep='last')

    rows: list[dict[str, float | str]] = []
    for listing_id, group in work.groupby('_listing_id', sort=False):
        group = group.sort_values(['_day', '_fetched_on', '_row_order'], kind='mergesort')
        prices = group[price_col].astype(float).to_numpy()
        first_price = float(prices[0])
        if not np.isfinite(first_price) or first_price <= 0:
            continue
        log_prices = np.log1p(prices)
        first_log_price = float(log_prices[0])

        days = (group['_day'] - group['_day'].iloc[0]).dt.days.astype(float).to_numpy()
        observed_days = float(max(days[-1], 0.0))
        effective_days = max(observed_days, 1.0)

        min_log_price_7d = float(log_prices[days <= 7].min()) if np.any(days <= 7) else first_log_price
        min_log_price_30d = float(log_prices[days <= 30].min()) if np.any(days <= 30) else first_log_price
        total_log_drop = max(first_log_price - float(log_prices[-1]), 0.0)

        first_drop_points = np.flatnonzero(log_prices < (first_log_price - 0.01))
        if len(first_drop_points):
            time_to_first_drop = float(days[first_drop_points[0]])
            first_drop_score = min(time_to_first_drop / 30.0, 1.0)
        else:
            time_to_first_drop = float(observed_days)
            first_drop_score = 1.0

        log_price_offsets = log_prices - first_log_price
        price_volatility = float(np.std(log_price_offsets, ddof=0))
        monthly_log_drop = max(first_log_price - min_log_price_30d, 0.0)
        daily_log_drop_rate = total_log_drop / effective_days

        sell_speed_score = 1.0 / (1.0 + observed_days / 30.0)
        resilience_score = 1.0 / (1.0 + 8.0 * monthly_log_drop + 30.0 * daily_log_drop_rate)
        volatility_score = 1.0 / (1.0 + 10.0 * price_volatility)
        liquidity = 100.0 * np.clip(
            0.45 * sell_speed_score
            + 0.20 * first_drop_score
            + 0.20 * resilience_score
            + 0.15 * volatility_score,
            0.0,
            1.0,
        )

        rows.append(
            {
                'ID': str(listing_id),
                'Expected Avg Drop 7d': max(first_log_price - min_log_price_7d, 0.0) / 7.0,
                'Expected Avg Drop 30d': max(first_log_price - min_log_price_30d, 0.0) / 30.0,
                'Expected Price Drop Rate': daily_log_drop_rate,
                'Expected Time To First Drop': time_to_first_drop,
                'Expected Time To Sell': observed_days,
                'Expected Price Volatility': price_volatility,
                'Expected Market Liquidity Score': liquidity,
            }
        )

    return pd.DataFrame(rows, columns=['ID'] + list(DYNAMICS_FEATURES))


def _fit_target_models(X: pd.DataFrame, targets: pd.DataFrame) -> tuple[dict[str, object], dict[str, float]]:
    models: dict[str, object] = {}
    default_values: dict[str, float] = {}

    for column in DYNAMICS_FEATURES:
        y = pd.to_numeric(targets[column], errors='coerce')
        fit_index = y.index[y.notna()]
        if len(fit_index) == 0:
            default_values[column] = 0.0
            models[column] = ConstantValueModel(0.0)
            continue

        y_fit = y.loc[fit_index].astype(float)
        default = float(y_fit.median())
        default_values[column] = default

        if len(fit_index) < 25 or float(y_fit.max() - y_fit.min()) < 1e-9:
            models[column] = ConstantValueModel(default)
            continue

        model = LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.1,
            random_state=SECONDARY_RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X.loc[fit_index], y_fit.to_numpy())
        models[column] = model

    return models, default_values


def build_market_dynamics_features(
    first_snapshot_df: pd.DataFrame,
    target_df: pd.DataFrame,
    train_idx: np.ndarray,
) -> tuple[pd.DataFrame, MarketDynamicsBundle]:
    input_meta = build_feature_meta(
        first_snapshot_df,
        numeric_feats=list(BASE_NUMERIC_FEATURES),
        categorical_feats=list(CATEGORICAL_FEATURES),
        preprocessor_type='listing_dynamics_input',
        preprocessor_version=1,
    )
    input_cols = input_meta['numeric_feats'] + input_meta['categorical_feats']
    X_all = prepare_model_input(first_snapshot_df[input_cols] if set(input_cols).issubset(first_snapshot_df.columns) else first_snapshot_df, input_meta)

    preds = pd.DataFrame(index=first_snapshot_df.index, columns=DYNAMICS_FEATURES, dtype=float)
    target_mask = target_df[DYNAMICS_FEATURES].notna().all(axis=1).to_numpy()
    train_idx = np.asarray(train_idx, dtype=int)
    train_target_idx = train_idx[target_mask[train_idx]]

    if len(train_target_idx) == 0:
        models = {column: ConstantValueModel(0.0) for column in DYNAMICS_FEATURES}
        default_values = {column: 0.0 for column in DYNAMICS_FEATURES}
        bundle = MarketDynamicsBundle(input_meta, models, default_values)
        return bundle.predict_features(first_snapshot_df), bundle

    n_splits = min(5, len(train_target_idx))
    if n_splits >= 2:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SECONDARY_RANDOM_STATE)
        for fit_pos, hold_pos in kf.split(train_target_idx):
            fold_fit_idx = train_target_idx[fit_pos]
            fold_hold_idx = train_target_idx[hold_pos]
            fold_models, fold_defaults = _fit_target_models(
                X_all.loc[fold_fit_idx],
                target_df.loc[fold_fit_idx, DYNAMICS_FEATURES],
            )
            fold_bundle = MarketDynamicsBundle(input_meta, fold_models, fold_defaults)
            preds.loc[fold_hold_idx, DYNAMICS_FEATURES] = fold_bundle.predict_features(
                first_snapshot_df.loc[fold_hold_idx]
            )

    final_models, final_defaults = _fit_target_models(
        X_all.loc[train_target_idx],
        target_df.loc[train_target_idx, DYNAMICS_FEATURES],
    )
    bundle = MarketDynamicsBundle(input_meta, final_models, final_defaults)

    missing_index = preds.index[preds.isna().any(axis=1)]
    if len(missing_index) > 0:
        preds.loc[missing_index, DYNAMICS_FEATURES] = bundle.predict_features(first_snapshot_df.loc[missing_index])

    return preds, bundle


def add_market_dynamics_features(df: pd.DataFrame, bundle: MarketDynamicsBundle) -> pd.DataFrame:
    work = df.copy()
    dynamic_features = bundle.predict_features(work)
    for column in DYNAMICS_FEATURES:
        work[column] = dynamic_features[column]
    return work


__all__ = [
    'DYNAMICS_FEATURES',
    'MarketDynamicsBundle',
    'build_listing_dynamics_targets',
    'build_market_dynamics_features',
    'add_market_dynamics_features',
]
