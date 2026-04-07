from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, train_test_split

from train_last_2weeks_baseline import (
    BASE_DIR,
    RANDOM_STATE,
    clean_features,
    evaluate_model,
    filter_last_2weeks,
    get_first_occurrence,
    load_data,
)


PREDICTIONS_PATH = Path(BASE_DIR) / "last_2weeks_with_dynamics_predictions.csv"
DYNAMICS_FEATURES = [
    "Expected Avg Drop 7d",
    "Expected Avg Drop 30d",
    "Expected Price Drop Rate",
    "Expected Time To First Drop",
    "Expected Time To Sell",
    "Expected Price Volatility",
    "Expected Market Liquidity Score",
]

TARGET_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "Expected Avg Drop 7d": (0.0, None),
    "Expected Avg Drop 30d": (0.0, None),
    "Expected Price Drop Rate": (0.0, None),
    "Expected Time To First Drop": (0.0, None),
    "Expected Time To Sell": (0.0, None),
    "Expected Price Volatility": (0.0, None),
    "Expected Market Liquidity Score": (0.0, 100.0),
}


@dataclass
class FeatureSchema:
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_fill_values: dict[str, float]
    categorical_levels: dict[str, list[str]]


class ConstantValueModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value, dtype=float)


def build_listing_dynamics_targets(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert repeated listing history into per-listing future-behaviour targets.

    These are derived from the full available listing history, but they are later
    predicted from listing-time static features only, so the main model can use
    them without directly consuming future rows.
    """
    required_columns = {"car_id", "date_posted", "price"}
    if not required_columns.issubset(snapshot_df.columns):
        return pd.DataFrame(columns=["car_id"] + DYNAMICS_FEATURES)

    work = snapshot_df.copy()
    work["car_id"] = work["car_id"].astype("string").str.strip()
    work["date_posted"] = pd.to_datetime(work["date_posted"], errors="coerce")
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work = work[
        work["car_id"].notna()
        & (work["car_id"] != "")
        & work["date_posted"].notna()
        & work["price"].notna()
        & (work["price"] > 0)
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=["car_id"] + DYNAMICS_FEATURES)

    work["_day"] = work["date_posted"].dt.floor("D")
    work["_row_order"] = np.arange(len(work))
    work = work.sort_values(["car_id", "_day", "date_posted", "_row_order"], kind="mergesort")
    work = work.drop_duplicates(subset=["car_id", "_day"], keep="last")

    rows: list[dict[str, float | str]] = []
    for car_id, group in work.groupby("car_id", sort=False):
        group = group.sort_values(["_day", "date_posted", "_row_order"], kind="mergesort")
        prices = group["price"].astype(float).to_numpy()
        first_price = float(prices[0])
        if not np.isfinite(first_price) or first_price <= 0:
            continue

        log_prices = np.log(prices)
        first_log_price = float(log_prices[0])
        days = (group["_day"] - group["_day"].iloc[0]).dt.days.astype(float).to_numpy()
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
                "car_id": str(car_id),
                "Expected Avg Drop 7d": max(first_log_price - min_log_price_7d, 0.0) / 7.0,
                "Expected Avg Drop 30d": max(first_log_price - min_log_price_30d, 0.0) / 30.0,
                "Expected Price Drop Rate": daily_log_drop_rate,
                "Expected Time To First Drop": time_to_first_drop,
                "Expected Time To Sell": observed_days,
                "Expected Price Volatility": price_volatility,
                "Expected Market Liquidity Score": liquidity,
            }
        )

    return pd.DataFrame(rows, columns=["car_id"] + DYNAMICS_FEATURES)


def fit_feature_schema(df: pd.DataFrame) -> FeatureSchema:
    numeric_columns = [column for column in ["kms_driven", "ownership", "car_age"] if column in df.columns]
    categorical_columns = [column for column in ["make", "model", "variant", "city"] if column in df.columns]

    numeric_fill_values: dict[str, float] = {}
    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce")
        median = series.median()
        numeric_fill_values[column] = float(median) if pd.notna(median) else 0.0

    categorical_levels: dict[str, list[str]] = {}
    for column in categorical_columns:
        values = df[column].fillna("UNKNOWN").astype(str).str.strip().replace({"": "UNKNOWN"})
        levels = sorted(values.unique().tolist())
        if "UNKNOWN" not in levels:
            levels.append("UNKNOWN")
        if column == "variant" and "OTHER" not in levels:
            levels.append("OTHER")
        categorical_levels[column] = levels

    return FeatureSchema(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_fill_values=numeric_fill_values,
        categorical_levels=categorical_levels,
    )


def apply_feature_schema(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    work = df.copy()
    for column in schema.numeric_columns:
        work[column] = (
            pd.to_numeric(work[column], errors="coerce")
            .fillna(schema.numeric_fill_values[column])
            .astype(float)
        )

    for column in schema.categorical_columns:
        values = work[column].fillna("UNKNOWN").astype(str).str.strip().replace({"": "UNKNOWN"})
        fallback = "OTHER" if column == "variant" else "UNKNOWN"
        values = values.where(values.isin(schema.categorical_levels[column]), fallback)
        work[column] = pd.Categorical(values, categories=schema.categorical_levels[column])

    ordered_columns = schema.numeric_columns + schema.categorical_columns
    return work[ordered_columns].copy()


def clip_dynamic_values(column: str, values: np.ndarray) -> np.ndarray:
    low, high = TARGET_BOUNDS.get(column, (None, None))
    clipped = np.asarray(values, dtype=float).ravel()
    if low is not None:
        clipped = np.maximum(clipped, low)
    if high is not None:
        clipped = np.minimum(clipped, high)
    return clipped


def make_dynamic_model() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        lambda_l1=1.0,
        lambda_l2=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def make_main_model() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        metric="mape",
        num_leaves=64,
        learning_rate=0.05,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        lambda_l1=2.0,
        lambda_l2=2.0,
        n_estimators=1200,
        importance_type="gain",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def build_dynamic_feature_predictions(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    target_train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Predict listing dynamics from static listing-time features only.

    The training rows get out-of-fold dynamic predictions so the primary model
    does not see the true history-derived targets for the same row.
    """
    train_pred = pd.DataFrame(index=X_train.index, columns=DYNAMICS_FEATURES, dtype=float)
    valid_pred = pd.DataFrame(index=X_valid.index, columns=DYNAMICS_FEATURES, dtype=float)

    n_splits = min(5, len(X_train))
    if n_splits >= 2:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for fit_idx, hold_idx in kf.split(X_train):
            fold_train_idx = X_train.index[fit_idx]
            fold_hold_idx = X_train.index[hold_idx]
            for column in DYNAMICS_FEATURES:
                y_fit = pd.to_numeric(target_train.loc[fold_train_idx, column], errors="coerce")
                usable_idx = y_fit.index[y_fit.notna()]
                if len(usable_idx) < 25 or float(y_fit.loc[usable_idx].max() - y_fit.loc[usable_idx].min()) < 1e-9:
                    default = float(y_fit.loc[usable_idx].median()) if len(usable_idx) else 0.0
                    train_pred.loc[fold_hold_idx, column] = default
                    continue

                model = make_dynamic_model()
                model.fit(X_train.loc[usable_idx], y_fit.loc[usable_idx].to_numpy())
                pred = clip_dynamic_values(column, model.predict(X_train.loc[fold_hold_idx]))
                train_pred.loc[fold_hold_idx, column] = pred

    final_models: dict[str, ConstantValueModel | LGBMRegressor] = {}
    defaults: dict[str, float] = {}
    for column in DYNAMICS_FEATURES:
        y_train = pd.to_numeric(target_train[column], errors="coerce")
        usable_idx = y_train.index[y_train.notna()]
        default = float(y_train.loc[usable_idx].median()) if len(usable_idx) else 0.0
        defaults[column] = default

        if len(usable_idx) < 25 or float(y_train.loc[usable_idx].max() - y_train.loc[usable_idx].min()) < 1e-9:
            final_models[column] = ConstantValueModel(default)
        else:
            model = make_dynamic_model()
            model.fit(X_train.loc[usable_idx], y_train.loc[usable_idx].to_numpy())
            final_models[column] = model

    for column in DYNAMICS_FEATURES:
        missing_idx = train_pred.index[train_pred[column].isna()]
        if len(missing_idx):
            fallback_pred = final_models[column].predict(X_train.loc[missing_idx])
            train_pred.loc[missing_idx, column] = clip_dynamic_values(column, fallback_pred)
        valid_pred[column] = clip_dynamic_values(column, final_models[column].predict(X_valid))

    return train_pred.astype(float), valid_pred.astype(float)


def train_model_with_dynamics(df: pd.DataFrame, dynamics_targets: pd.DataFrame) -> dict[str, object]:
    feature_columns = ["make", "model", "variant", "kms_driven", "ownership", "city", "car_age"]
    feature_columns = [column for column in feature_columns if column in df.columns]

    work = df.merge(dynamics_targets, on="car_id", how="left")
    idx_train, idx_valid = train_test_split(
        np.arange(len(work)),
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    train_df = work.iloc[idx_train].copy()
    valid_df = work.iloc[idx_valid].copy()

    schema = fit_feature_schema(train_df[feature_columns])
    X_train_base = apply_feature_schema(train_df[feature_columns], schema)
    X_valid_base = apply_feature_schema(valid_df[feature_columns], schema)

    dynamics_train, dynamics_valid = build_dynamic_feature_predictions(
        X_train_base,
        X_valid_base,
        train_df[DYNAMICS_FEATURES],
    )

    X_train = X_train_base.copy()
    X_valid = X_valid_base.copy()
    for column in DYNAMICS_FEATURES:
        X_train[column] = dynamics_train[column].astype(float)
        X_valid[column] = dynamics_valid[column].astype(float)

    y_train = np.log(train_df["price"].astype(float).to_numpy())
    y_valid = np.log(valid_df["price"].astype(float).to_numpy())

    model = make_main_model()
    model.fit(
        X_train,
        y_train,
        categorical_feature=schema.categorical_columns,
    )

    pred_price = np.exp(np.asarray(model.predict(X_valid), dtype=float).ravel())
    actual_price = np.exp(y_valid)
    metrics = evaluate_model(actual_price, pred_price)

    predictions = pd.DataFrame(
        {
            "car_id": valid_df["car_id"].astype(str).to_numpy(),
            "actual_price": actual_price,
            "predicted_price": pred_price,
        }
    )
    for column in DYNAMICS_FEATURES:
        predictions[f"Pred {column}"] = dynamics_valid[column].to_numpy()

    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved predictions to: {PREDICTIONS_PATH}")

    return {
        "model": model,
        "metrics": metrics,
        "predictions": predictions,
        "schema": schema,
    }


def main() -> None:
    print("Loading data...")
    raw_df = load_data()
    print(f"Loaded {len(raw_df):,} raw rows")

    print("Building listing-level dynamics targets from full history...")
    dynamics_targets = build_listing_dynamics_targets(raw_df)
    print(f"Dynamics targets built for {len(dynamics_targets):,} unique cars")

    print("Filtering last 2 weeks...")
    recent_df = filter_last_2weeks(raw_df)
    print(f"Rows after last-2-weeks filter: {len(recent_df):,}")

    print("Keeping first occurrence per car...")
    first_df = get_first_occurrence(recent_df)
    print(f"Rows after first-occurrence dedupe: {len(first_df):,}")

    print("Cleaning static features...")
    clean_df = clean_features(first_df)
    print(f"Rows after cleaning: {len(clean_df):,}")

    print("Training last-2-weeks model with market dynamics features...")
    train_model_with_dynamics(clean_df, dynamics_targets)


if __name__ == "__main__":
    main()
