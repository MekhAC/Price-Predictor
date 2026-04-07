from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold

from model_preprocessing import (
    DATA_DIR,
    MODELS_DIR,
    _TRANSMISSION_MAP,
    _normalize_bodytype_value,
    _normalize_city_value,
    _normalize_fuel_value,
    _normalize_reg_state_value,
    _normalize_text_series,
    _parse_kms,
    _parse_ownership,
    _reg_state,
    _split_name_to_make_model,
    standardize_columns,
)

CURRENT_YEAR = datetime.now().year
TARGET_TRANSFORM = "log_price"
LISTING_ID_COLUMN = "Listing ID"
LISTING_DATE_COLUMN = "Listing Date"
DAYS_SINCE_LISTING_COLUMN = "Days Since Listing"
IS_LISTING_DAY_COLUMN = "Is Listing Day"
TARGET_PRICE_COLUMN = "Target Price"
VARIANT_GROUPED_COLUMN = "Variant Grouped"
MODEL_TE_COLUMN = "Model Target Encode"
VARIANT_TE_COLUMN = "Variant Target Encode"
PRESSURE_FEATURE_COLUMN = "Price Pressure Score"
CAR_AGE_BUCKET_COLUMN = "Car Age Bucket"
MARKET_COUNT_FEATURE_COLUMN = "Market Listings Count Log"
MARKET_MEDIAN_PRICE_FEATURE_COLUMN = "Market Median Price"
MARKET_PRESSURE_FEATURE_COLUMN = "Market Pressure Mean"
MARKET_DROP_30D_FEATURE_COLUMN = "Market Avg Drop 30d"
MARKET_SELL_SPEED_FEATURE_COLUMN = "Market Sell Speed"
MARKET_VOLATILITY_FEATURE_COLUMN = "Market Volatility"

TOP_N_VARIANTS = 250
TARGET_ENCODING_SMOOTHING = 50.0
OUTER_FOLDS = 5
INNER_FOLDS = 4

PRESSURE_COMPONENT_COLUMNS = [
    "Avg Drop 30d Raw",
    "Time To Sell Inverse Raw",
    "Volatility Raw",
]
PRESSURE_NUMERIC_COLUMNS = [
    "Car Age",
    "KMs Driven",
    "KMs/Year",
    "Log KMs",
    "Ownership",
    "Age x Ownership",
    "KMs x Ownership",
]
PRESSURE_CATEGORICAL_COLUMNS = [
    "Make",
    "Model",
    VARIANT_GROUPED_COLUMN,
    "Transmission",
    "Fuel",
    "BodyType",
    "City",
    "Reg State",
]
MAIN_NUMERIC_COLUMNS = [
    "Car Age",
    "KMs Driven",
    "KMs/Year",
    "Log KMs",
    "Ownership",
    "Age x Ownership",
    "KMs x Ownership",
    PRESSURE_FEATURE_COLUMN,
    MARKET_COUNT_FEATURE_COLUMN,
    MARKET_MEDIAN_PRICE_FEATURE_COLUMN,
    MARKET_PRESSURE_FEATURE_COLUMN,
    MARKET_DROP_30D_FEATURE_COLUMN,
    MARKET_SELL_SPEED_FEATURE_COLUMN,
    MARKET_VOLATILITY_FEATURE_COLUMN,
    MODEL_TE_COLUMN,
    VARIANT_TE_COLUMN,
]
MAIN_CATEGORICAL_COLUMNS = [
    "Make",
    VARIANT_GROUPED_COLUMN,
    "Transmission",
    "Fuel",
    "BodyType",
    "City",
    "Reg State",
]
MARKET_HISTORY_FEATURE_COLUMNS = [
    MARKET_COUNT_FEATURE_COLUMN,
    MARKET_MEDIAN_PRICE_FEATURE_COLUMN,
    MARKET_PRESSURE_FEATURE_COLUMN,
    MARKET_DROP_30D_FEATURE_COLUMN,
    MARKET_SELL_SPEED_FEATURE_COLUMN,
    MARKET_VOLATILITY_FEATURE_COLUMN,
]
MARKET_HISTORY_LEVELS: list[tuple[list[str], int]] = [
    (["Make", "Model", VARIANT_GROUPED_COLUMN, "City", CAR_AGE_BUCKET_COLUMN], 15),
    (["Make", "Model", VARIANT_GROUPED_COLUMN, CAR_AGE_BUCKET_COLUMN], 10),
    (["Make", "Model", "City", CAR_AGE_BUCKET_COLUMN], 8),
    (["Make", "Model", CAR_AGE_BUCKET_COLUMN], 5),
    (["Make", "Model"], 3),
]


@dataclass
class FeatureSchema:
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_fill_values: dict[str, float]
    categorical_levels: dict[str, list[str]]


@dataclass
class PressureScoreBundle:
    model: LGBMRegressor | None
    schema: FeatureSchema
    variant_top_values: list[str]
    default_score: float

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty:
            return np.array([], dtype=float)

        work = df.copy()
        work[VARIANT_GROUPED_COLUMN] = _group_variants(work["Variant"], self.variant_top_values)
        X = _apply_feature_schema(work, self.schema)

        if self.model is None:
            return np.full(len(df), self.default_score, dtype=float)

        preds = np.asarray(self.model.predict(X), dtype=float).ravel()
        return np.clip(preds, 0.0, 1.0)


@dataclass
class MarketHistoryBundle:
    tables: list[dict[str, Any]]
    defaults: dict[str, float]
    variant_top_values: list[str]

    def predict_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=MARKET_HISTORY_FEATURE_COLUMNS, index=df.index, dtype=float)

        work = _prepare_market_lookup_frame(df, self.variant_top_values)
        result = pd.DataFrame(index=work.index, columns=MARKET_HISTORY_FEATURE_COLUMNS, dtype=float)

        for spec in self.tables:
            keys = spec["keys"]
            table = spec["table"]
            if table.empty:
                continue

            lookup = (
                work.reset_index()[["index"] + keys]
                .merge(table, on=keys, how="left")
                .set_index("index")
                .reindex(work.index)
            )
            unresolved = result[MARKET_HISTORY_FEATURE_COLUMNS[0]].isna()
            available = lookup[MARKET_HISTORY_FEATURE_COLUMNS].notna().all(axis=1)
            fill_index = lookup.index[unresolved & available]
            if len(fill_index):
                result.loc[fill_index, MARKET_HISTORY_FEATURE_COLUMNS] = lookup.loc[
                    fill_index, MARKET_HISTORY_FEATURE_COLUMNS
                ].to_numpy()

        for column in MARKET_HISTORY_FEATURE_COLUMNS:
            result[column] = (
                pd.to_numeric(result[column], errors="coerce")
                .fillna(float(self.defaults.get(column, 0.0)))
                .astype(float)
            )

        return result


@dataclass
class TrainingBundle:
    model: LGBMRegressor
    schema: FeatureSchema
    categorical_encoder: dict[str, Any]
    pressure_bundle: PressureScoreBundle
    market_history_bundle: MarketHistoryBundle
    target_transform: str = TARGET_TRANSFORM

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = clean_features(df, allow_missing_target=True, prediction_mode=True)
        market_features = self.market_history_bundle.predict_features(cleaned)
        for column in MARKET_HISTORY_FEATURE_COLUMNS:
            cleaned[column] = market_features[column]
        cleaned[PRESSURE_FEATURE_COLUMN] = self.pressure_bundle.predict(cleaned)
        cleaned, _ = encode_categoricals(cleaned, fit=False, encoder=self.categorical_encoder)
        return _apply_feature_schema(cleaned, self.schema)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.transform(df)
        if X.empty:
            return np.array([], dtype=float)
        preds_log = np.asarray(self.model.predict(X), dtype=float).ravel()
        return np.exp(preds_log)


def _resolve_price_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        if str(column).startswith("Price"):
            return str(column)
    return None


def _read_source_file(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        parquet_path = path.rsplit(".", 1)[0] + ".parquet"
        if os.path.exists(parquet_path) and os.path.getmtime(parquet_path) >= os.path.getmtime(path):
            return pd.read_parquet(parquet_path)
        df = pd.read_excel(path)
        df.to_parquet(parquet_path, index=False)
        return df
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _collect_training_files() -> list[str]:
    files = (
        glob.glob(os.path.join(DATA_DIR, "normalized_table.*"))
        + glob.glob(os.path.join(DATA_DIR, "normalized_table_*.*"))
        + glob.glob(os.path.join(DATA_DIR, "Cars24_*.*"))
        + glob.glob(os.path.join(DATA_DIR, "Spinny_*.*"))
    )
    return [
        path
        for path in files
        if not (path.lower().endswith(".parquet") and os.path.exists(path.rsplit(".", 1)[0] + ".xlsx"))
    ]


def _load_training_history() -> pd.DataFrame:
    files = _collect_training_files()
    if not files:
        raise RuntimeError(f"No data files found in {DATA_DIR}")

    print(f"Loading {len(files)} files for leakage-safe training...")
    frames = [_read_source_file(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def _group_variants(series: pd.Series, top_values: list[str]) -> pd.Series:
    top_set = set(top_values)
    work = series.fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
    keep_mask = work.isin(top_set) | work.eq("Unknown")
    return work.where(keep_mask, "other")


def _age_bucket_from_car_age(series: pd.Series) -> pd.Series:
    car_age = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    bucket = np.floor(car_age / 2.0).clip(0, 12).astype(int)
    return bucket.astype(str)


def _prepare_market_lookup_frame(df: pd.DataFrame, variant_top_values: list[str]) -> pd.DataFrame:
    work = df.copy()
    work[VARIANT_GROUPED_COLUMN] = _group_variants(work["Variant"], variant_top_values)
    work[CAR_AGE_BUCKET_COLUMN] = _age_bucket_from_car_age(work["Car Age"])
    return work


def _smoothed_target_mean(
    keys: pd.Series,
    target: pd.Series,
    global_mean: float,
    smoothing: float,
) -> dict[str, float]:
    frame = pd.DataFrame({"key": keys.astype(str), "target": pd.Series(target, index=keys.index, dtype=float)})
    agg = frame.groupby("key", observed=True)["target"].agg(["mean", "count"])
    smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
    return {str(key): float(value) for key, value in smooth.items()}


def _fit_feature_schema(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> FeatureSchema:
    numeric_fill_values: dict[str, float] = {}
    categorical_levels: dict[str, list[str]] = {}

    for column in numeric_columns:
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce")
        else:
            series = pd.Series(dtype=float)
        median = series.median()
        numeric_fill_values[column] = float(median) if pd.notna(median) else 0.0

    for column in categorical_columns:
        if column in df.columns:
            values = df[column].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
        else:
            values = pd.Series(["Unknown"], dtype="object")
        levels = sorted(values.unique().tolist())
        if "Unknown" not in levels:
            levels.append("Unknown")
        if column == VARIANT_GROUPED_COLUMN and "other" not in levels:
            levels.append("other")
        categorical_levels[column] = levels

    return FeatureSchema(
        numeric_columns=list(numeric_columns),
        categorical_columns=list(categorical_columns),
        numeric_fill_values=numeric_fill_values,
        categorical_levels=categorical_levels,
    )


def _apply_feature_schema(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    work = df.copy()

    for column in schema.numeric_columns:
        if column not in work.columns:
            work[column] = np.nan
        work[column] = (
            pd.to_numeric(work[column], errors="coerce")
            .fillna(schema.numeric_fill_values[column])
            .astype(float)
        )

    for column in schema.categorical_columns:
        levels = schema.categorical_levels[column]
        fallback = "other" if column == VARIANT_GROUPED_COLUMN and "other" in levels else "Unknown"
        if column not in work.columns:
            work[column] = fallback
        values = work[column].fillna(fallback).astype(str).str.strip().replace({"": fallback})
        values = values.where(values.isin(levels), fallback)
        work[column] = pd.Categorical(values, categories=levels)

    return work[schema.numeric_columns + schema.categorical_columns]


def _fit_pressure_normalizer(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    params: dict[str, dict[str, float]] = {}
    for column in PRESSURE_COMPONENT_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        low = float(values.quantile(0.05)) if values.notna().any() else 0.0
        high = float(values.quantile(0.95)) if values.notna().any() else 1.0
        if not np.isfinite(low):
            low = 0.0
        if not np.isfinite(high):
            high = 1.0
        if high <= low:
            high = low + 1.0
        params[column] = {"low": low, "high": high}
    return params


def _apply_pressure_normalizer(
    frame: pd.DataFrame,
    normalizer: dict[str, dict[str, float]],
) -> pd.DataFrame:
    work = frame.copy()
    for column in PRESSURE_COMPONENT_COLUMNS:
        low = normalizer[column]["low"]
        high = normalizer[column]["high"]
        scale = high - low if high > low else 1.0
        normalized = (pd.to_numeric(work[column], errors="coerce").fillna(low) - low) / scale
        work[f"{column} Normalized"] = normalized.clip(0.0, 1.0)

    work[PRESSURE_FEATURE_COLUMN] = (
        0.4 * work["Avg Drop 30d Raw Normalized"]
        + 0.3 * work["Time To Sell Inverse Raw Normalized"]
        + 0.3 * work["Volatility Raw Normalized"]
    )
    return work


def _build_pressure_model(n_estimators: int = 600) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        importance_type="gain",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l1=1.0,
        lambda_l2=2.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def _build_main_model(n_estimators: int = 2500) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        importance_type="gain",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=64,
        min_data_in_leaf=150,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l1=5.0,
        lambda_l2=5.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def _fit_main_regressor(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    *,
    categorical_features: list[str],
    X_valid: pd.DataFrame | None = None,
    y_valid_log: pd.Series | None = None,
    n_estimators: int = 2500,
) -> tuple[LGBMRegressor, int]:
    model = _build_main_model(n_estimators=n_estimators)

    fit_kwargs: dict[str, Any] = {
        "categorical_feature": list(categorical_features),
    }
    if X_valid is not None and y_valid_log is not None and not X_valid.empty:
        fit_kwargs["eval_set"] = [(X_valid, y_valid_log)]
        fit_kwargs["callbacks"] = [
            lgb_early_stopping(stopping_rounds=200, first_metric_only=True, verbose=False),
            lgb_log_evaluation(period=-1),
        ]

    model.fit(X_train, y_train_log, **fit_kwargs)
    best_iteration = int(model.best_iteration_ or model.n_estimators_)
    return model, best_iteration


def _metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    mpe = float(np.mean((y_pred - y_true) / np.clip(y_true, 1.0, None)) * 100.0)
    return {
        "mse": float(mse),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100.0),
        "rmse": float(sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mae),
        "median_ae": float(medae),
        "mpe": mpe,
    }


def _build_group_splitter(groups: pd.Series, desired_splits: int) -> GroupKFold | None:
    unique_groups = int(groups.astype(str).nunique())
    n_splits = min(desired_splits, unique_groups)
    if n_splits < 2:
        return None
    return GroupKFold(n_splits=n_splits)


def clean_features(
    df: pd.DataFrame,
    *,
    allow_missing_target: bool = False,
    reference_datetime: datetime | None = None,
    prediction_mode: bool = False,
) -> pd.DataFrame:
    # Keep only information available at listing time and drop known leakage fields.
    work = standardize_columns(df.copy())
    work["_row_order"] = np.arange(len(work))
    ref_dt = pd.Timestamp(reference_datetime or datetime.now())

    if "Name" in work.columns and "Make" not in work.columns:
        makes, models = _split_name_to_make_model(work["Name"])
        work["Make"] = makes
        work["Model"] = models

    if "Transmission" in work.columns:
        work["Transmission"] = (
            work["Transmission"].astype(str).str.strip().map(_TRANSMISSION_MAP).fillna(work["Transmission"])
        )

    for column in ["Make", "Model", "Variant", "Transmission", "Fuel", "BodyType", "City"]:
        if column not in work.columns:
            work[column] = "Unknown"
    if "Registration" not in work.columns:
        work["Registration"] = ""
    if "Ownership" not in work.columns:
        work["Ownership"] = np.nan
    if "Year" not in work.columns:
        work["Year"] = np.nan
    if "KMs Driven" not in work.columns:
        work["KMs Driven"] = np.nan

    price_column = _resolve_price_column(work)
    if price_column is None:
        work[TARGET_PRICE_COLUMN] = np.nan
    else:
        work[TARGET_PRICE_COLUMN] = (
            work[price_column]
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
            .replace({"": np.nan})
            .astype(float)
        )

    work["Year"] = (
        work["Year"]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .replace({"": np.nan})
        .astype(float)
    )
    work["KMs Driven"] = work["KMs Driven"].apply(_parse_kms)
    work["Ownership"] = pd.to_numeric(work["Ownership"].apply(_parse_ownership), errors="coerce").fillna(1.0)
    listing_source = (
        work["Fetched On"]
        if "Fetched On" in work.columns
        else pd.Series([pd.NaT] * len(work), index=work.index, dtype="datetime64[ns]")
    )
    work[LISTING_DATE_COLUMN] = pd.to_datetime(listing_source, errors="coerce")

    listing_id = work.get("ID", pd.Series([pd.NA] * len(work), index=work.index, dtype="object"))
    listing_id = pd.Series(listing_id, index=work.index, dtype="object").astype("string").str.strip()
    missing_listing_id = listing_id.isna() | (listing_id == "")
    listing_id = listing_id.where(~missing_listing_id, pd.Series([f"row_{idx}" for idx in work.index], index=work.index))
    work[LISTING_ID_COLUMN] = listing_id.astype(str)

    first_seen = work.groupby(LISTING_ID_COLUMN)[LISTING_DATE_COLUMN].transform("min")
    days_since_listing = (work[LISTING_DATE_COLUMN].dt.floor("D") - first_seen.dt.floor("D")).dt.days
    work[DAYS_SINCE_LISTING_COLUMN] = days_since_listing.fillna(0.0).clip(lower=0).astype(float)
    work[IS_LISTING_DAY_COLUMN] = work[DAYS_SINCE_LISTING_COLUMN] <= 1.0

    listing_year = work[LISTING_DATE_COLUMN].dt.year.fillna(float(ref_dt.year))
    work["Car Age"] = (listing_year - work["Year"]).clip(lower=0).astype(float)
    work[CAR_AGE_BUCKET_COLUMN] = _age_bucket_from_car_age(work["Car Age"])
    work["KMs/Year"] = np.where(work["Car Age"] > 0, work["KMs Driven"] / work["Car Age"], work["KMs Driven"])
    work["Log KMs"] = np.log1p(work["KMs Driven"].clip(lower=0))
    work["Age x Ownership"] = work["Car Age"] * work["Ownership"]
    work["KMs x Ownership"] = work["KMs Driven"] * work["Ownership"]

    reg_state_from_registration = work["Registration"].apply(_reg_state)
    if "Reg State" in work.columns:
        existing_reg_state = work["Reg State"].fillna("").astype(str).str.strip()
        reg_state_mask = existing_reg_state != ""
        work["Reg State"] = existing_reg_state.where(reg_state_mask, reg_state_from_registration)
    else:
        work["Reg State"] = reg_state_from_registration
    work["Make"] = _normalize_text_series(work["Make"]).str.upper()
    work["Model"] = _normalize_text_series(work["Model"]).str.upper()
    work["Variant"] = _normalize_text_series(work["Variant"]).str.upper()
    work["Transmission"] = work["Transmission"].map(
        lambda value: _TRANSMISSION_MAP.get(str(value).strip(), str(value).strip() or "Unknown")
    )
    work["Fuel"] = work["Fuel"].map(_normalize_fuel_value)
    work["BodyType"] = work["BodyType"].map(_normalize_bodytype_value)
    work["City"] = work["City"].map(_normalize_city_value)
    work["Reg State"] = work["Reg State"].map(_normalize_reg_state_value)

    # Remove leakage and redundancy from the feature space.
    work = work.drop(columns=["Market Days", "Days On Market", "Fetch Month"], errors="ignore")

    if "ID" in work.columns and not prediction_mode:
        for column in ["Year", "Fuel", "Transmission"]:
            nunique = work.groupby(LISTING_ID_COLUMN)[column].transform("nunique")
            work = work[nunique <= 1]

    year_valid = (work["Year"] >= 2005) & (work["Year"] <= float(ref_dt.year))
    kms_valid = (work["KMs Driven"] >= 0) & (work["KMs Driven"] <= 400_000)

    if prediction_mode:
        work.loc[~year_valid, "Year"] = np.nan
        work.loc[~kms_valid, "KMs Driven"] = np.nan
    elif allow_missing_target:
        work = work.dropna(subset=["Year", "KMs Driven"])
    else:
        work = work.dropna(subset=[TARGET_PRICE_COLUMN, "Year", "KMs Driven"])
        work = work[work[TARGET_PRICE_COLUMN] > 0]

    if not prediction_mode:
        work = work[(work["Year"] >= 2005) & (work["Year"] <= float(ref_dt.year))]
        work = work[(work["KMs Driven"] >= 0) & (work["KMs Driven"] <= 400_000)]
        work = work.sort_values(
            [LISTING_ID_COLUMN, LISTING_DATE_COLUMN, DAYS_SINCE_LISTING_COLUMN, "_row_order"],
            kind="mergesort",
            na_position="last",
        )
    else:
        work = work.sort_values(["_row_order"], kind="mergesort")

    return work.reset_index(drop=True)


def create_price_pressure_score(df: pd.DataFrame) -> pd.DataFrame:
    # Build listing-level raw pressure components from the full repeated-history table.
    required = {LISTING_ID_COLUMN, LISTING_DATE_COLUMN, TARGET_PRICE_COLUMN}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=[LISTING_ID_COLUMN] + PRESSURE_COMPONENT_COLUMNS + [PRESSURE_FEATURE_COLUMN])

    work = df.copy()
    work = work.dropna(subset=[LISTING_ID_COLUMN, LISTING_DATE_COLUMN, TARGET_PRICE_COLUMN]).copy()
    if work.empty:
        return pd.DataFrame(columns=[LISTING_ID_COLUMN] + PRESSURE_COMPONENT_COLUMNS + [PRESSURE_FEATURE_COLUMN])

    work["_day"] = work[LISTING_DATE_COLUMN].dt.floor("D")
    work = work.sort_values([LISTING_ID_COLUMN, "_day", LISTING_DATE_COLUMN, "_row_order"], kind="mergesort")
    work = work.drop_duplicates(subset=[LISTING_ID_COLUMN, "_day"], keep="last")

    rows: list[dict[str, float | str]] = []
    for listing_id, group in work.groupby(LISTING_ID_COLUMN, sort=False):
        group = group.sort_values(["_day", LISTING_DATE_COLUMN, "_row_order"], kind="mergesort")
        prices = pd.to_numeric(group[TARGET_PRICE_COLUMN], errors="coerce").to_numpy(dtype=float)
        prices = prices[np.isfinite(prices) & (prices > 0)]
        if len(prices) == 0:
            continue

        log_prices = np.log(prices)
        first_log_price = float(log_prices[0])
        days = (group["_day"] - group["_day"].iloc[0]).dt.days.astype(float).to_numpy()
        observed_days = float(max(days[-1], 0.0))

        min_log_price_30d = float(log_prices[days <= 30].min()) if np.any(days <= 30) else first_log_price
        avg_drop_30d = max(first_log_price - min_log_price_30d, 0.0) / 30.0
        time_to_sell_inverse = 1.0 / (1.0 + observed_days)
        volatility = float(np.std(log_prices - first_log_price, ddof=0))

        rows.append(
            {
                LISTING_ID_COLUMN: str(listing_id),
                "Avg Drop 30d Raw": avg_drop_30d,
                "Time To Sell Inverse Raw": time_to_sell_inverse,
                "Volatility Raw": volatility,
            }
        )

    pressure = pd.DataFrame(rows, columns=[LISTING_ID_COLUMN] + PRESSURE_COMPONENT_COLUMNS)
    if pressure.empty:
        pressure[PRESSURE_FEATURE_COLUMN] = pd.Series(dtype=float)
        return pressure

    normalizer = _fit_pressure_normalizer(pressure)
    return _apply_pressure_normalizer(pressure, normalizer)


def encode_categoricals(
    df: pd.DataFrame,
    *,
    fit: bool = False,
    encoder: dict[str, Any] | None = None,
    target: pd.Series | np.ndarray | None = None,
    top_n_variants: int = TOP_N_VARIANTS,
    smoothing: float = TARGET_ENCODING_SMOOTHING,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = df.copy()

    if fit:
        if target is None:
            raise ValueError("target is required when fit=True")

        target_series = pd.Series(target, index=work.index, dtype=float)
        variant_counts = work["Variant"].fillna("Unknown").astype(str).value_counts()
        top_variants = variant_counts.head(min(top_n_variants, len(variant_counts))).index.tolist()
        grouped_variant = _group_variants(work["Variant"], top_variants)
        global_mean = float(target_series.mean()) if len(target_series) else 0.0
        model_target_means = _smoothed_target_mean(work["Model"], target_series, global_mean, smoothing)
        variant_target_means = _smoothed_target_mean(grouped_variant, target_series, global_mean, smoothing)

        template = work.copy()
        template[VARIANT_GROUPED_COLUMN] = grouped_variant
        categorical_levels: dict[str, list[str]] = {}
        for column in MAIN_CATEGORICAL_COLUMNS:
            values = template[column].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
            levels = sorted(values.unique().tolist())
            if "Unknown" not in levels:
                levels.append("Unknown")
            if column == VARIANT_GROUPED_COLUMN and "other" not in levels:
                levels.append("other")
            categorical_levels[column] = levels

        encoder = {
            "top_variants": top_variants,
            "global_target_mean": global_mean,
            "model_target_means": model_target_means,
            "variant_target_means": variant_target_means,
            "categorical_levels": categorical_levels,
        }

    if encoder is None:
        raise ValueError("encoder is required when fit=False")

    work[VARIANT_GROUPED_COLUMN] = _group_variants(work["Variant"], encoder["top_variants"])
    work[MODEL_TE_COLUMN] = (
        work["Model"].map(encoder["model_target_means"]).fillna(encoder["global_target_mean"]).astype(float)
    )
    work[VARIANT_TE_COLUMN] = (
        work[VARIANT_GROUPED_COLUMN]
        .map(encoder["variant_target_means"])
        .fillna(encoder["global_target_mean"])
        .astype(float)
    )

    for column in MAIN_CATEGORICAL_COLUMNS:
        levels = encoder["categorical_levels"][column]
        fallback = "other" if column == VARIANT_GROUPED_COLUMN and "other" in levels else "Unknown"
        values = work[column].fillna(fallback).astype(str).str.strip().replace({"": fallback})
        values = values.where(values.isin(levels), fallback)
        work[column] = pd.Categorical(values, categories=levels)

    return work, encoder


def _select_anchor_rows(df: pd.DataFrame) -> pd.DataFrame:
    anchor = df[df[IS_LISTING_DAY_COLUMN]].copy()
    if anchor.empty:
        anchor = df.copy()

    anchor = anchor.sort_values(
        [LISTING_ID_COLUMN, LISTING_DATE_COLUMN, DAYS_SINCE_LISTING_COLUMN, "_row_order"],
        kind="mergesort",
        na_position="last",
    )
    anchor = anchor.drop_duplicates(subset=[LISTING_ID_COLUMN], keep="first")
    return anchor.reset_index(drop=True)


def summarize_prediction_row_flow(
    df: pd.DataFrame,
    *,
    reference_datetime: datetime | None = None,
) -> dict[str, int]:
    cleaned = clean_features(
        df,
        allow_missing_target=True,
        reference_datetime=reference_datetime,
        prediction_mode=True,
    )
    anchor_projection = _select_anchor_rows(cleaned)

    metric_rows = 0
    if TARGET_PRICE_COLUMN in cleaned.columns:
        actual = pd.to_numeric(cleaned[TARGET_PRICE_COLUMN], errors="coerce")
        metric_rows = int((actual.notna() & (actual > 0)).sum())

    return {
        "input_rows": int(len(df)),
        "cleaned_rows": int(len(cleaned)),
        "dropped_in_cleaning": int(len(df) - len(cleaned)),
        "prediction_rows": int(len(cleaned)),
        "rows_if_anchor_deduped": int(len(anchor_projection)),
        "would_drop_in_anchor_dedup": int(len(cleaned) - len(anchor_projection)),
        "metric_rows": metric_rows,
    }


def load_bundle(
    model_dir: str = MODELS_DIR,
    artifact_prefix: str = "leakage_safe",
) -> TrainingBundle:
    bundle_path = os.path.join(model_dir, f"{artifact_prefix}_model_bundle.joblib")
    # Older artifacts were saved when this module was executed as a script, which
    # pickled these dataclasses under the "__main__" module path. Expose the current
    # classes there so those bundles can still be loaded without forcing a retrain.
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        for cls in (FeatureSchema, PressureScoreBundle, MarketHistoryBundle, TrainingBundle):
            setattr(main_module, cls.__name__, cls)
    return joblib.load(bundle_path)


def predict_with_metrics(
    df: pd.DataFrame,
    *,
    bundle: TrainingBundle | None = None,
    model_dir: str = MODELS_DIR,
    artifact_prefix: str = "leakage_safe",
) -> tuple[pd.DataFrame, dict[str, float] | None]:
    # Score listing-time rows with the leakage-safe bundle and, when actual price is present,
    # return regression metrics computed on those same inference-safe rows.
    if bundle is None:
        bundle = load_bundle(model_dir=model_dir, artifact_prefix=artifact_prefix)

    cleaned = clean_features(df, allow_missing_target=True, prediction_mode=True)
    if cleaned.empty:
        raise RuntimeError("No valid rows available for leakage-safe prediction.")

    X = bundle.transform(df)
    if X.empty:
        raise RuntimeError("No transformed rows available for leakage-safe prediction.")

    preds = np.exp(np.asarray(bundle.model.predict(X), dtype=float).ravel())
    output = cleaned.copy()
    output["Predicted Price"] = preds

    metrics: dict[str, float] | None = None
    if TARGET_PRICE_COLUMN in output.columns:
        actual = pd.to_numeric(output[TARGET_PRICE_COLUMN], errors="coerce")
        valid = actual.notna() & np.isfinite(preds) & (actual > 0)
        if valid.any():
            y_true = actual.loc[valid].to_numpy(dtype=float)
            y_pred = output.loc[valid, "Predicted Price"].to_numpy(dtype=float)

            output["Error"] = output["Predicted Price"] - actual
            output["Absolute Error"] = output["Error"].abs()
            output["Absolute Percentage Error (%)"] = (
                output["Absolute Error"] / actual.replace(0, np.nan)
            ) * 100.0

            metrics = _metric_summary(y_true, y_pred)
            metrics["rows"] = int(valid.sum())

    return output.reset_index(drop=True), metrics


def _fit_market_history_bundle(
    train_df: pd.DataFrame,
    pressure_targets: pd.DataFrame,
) -> MarketHistoryBundle:
    variant_top_values = (
        train_df["Variant"].fillna("Unknown").astype(str).value_counts().head(TOP_N_VARIANTS).index.tolist()
    )
    work = _prepare_market_lookup_frame(train_df, variant_top_values)
    target_columns = [LISTING_ID_COLUMN] + PRESSURE_COMPONENT_COLUMNS + [PRESSURE_FEATURE_COLUMN]
    merged = work.merge(
        pressure_targets[target_columns] if set(target_columns).issubset(pressure_targets.columns) else pressure_targets,
        on=LISTING_ID_COLUMN,
        how="left",
    )

    defaults = {
        MARKET_COUNT_FEATURE_COLUMN: float(np.log1p(len(work))),
        MARKET_MEDIAN_PRICE_FEATURE_COLUMN: float(pd.to_numeric(work[TARGET_PRICE_COLUMN], errors="coerce").median()),
        MARKET_PRESSURE_FEATURE_COLUMN: float(pd.to_numeric(merged[PRESSURE_FEATURE_COLUMN], errors="coerce").median())
        if PRESSURE_FEATURE_COLUMN in merged.columns
        else 0.0,
        MARKET_DROP_30D_FEATURE_COLUMN: float(pd.to_numeric(merged["Avg Drop 30d Raw"], errors="coerce").median())
        if "Avg Drop 30d Raw" in merged.columns
        else 0.0,
        MARKET_SELL_SPEED_FEATURE_COLUMN: float(
            pd.to_numeric(merged["Time To Sell Inverse Raw"], errors="coerce").median()
        )
        if "Time To Sell Inverse Raw" in merged.columns
        else 0.0,
        MARKET_VOLATILITY_FEATURE_COLUMN: float(pd.to_numeric(merged["Volatility Raw"], errors="coerce").median())
        if "Volatility Raw" in merged.columns
        else 0.0,
    }

    tables: list[dict[str, Any]] = []
    for keys, min_count in MARKET_HISTORY_LEVELS:
        grouped = (
            merged.groupby(keys, observed=True)
            .agg(
                _listing_count=(LISTING_ID_COLUMN, "size"),
                _median_price=(TARGET_PRICE_COLUMN, "median"),
                _pressure_mean=(PRESSURE_FEATURE_COLUMN, "mean"),
                _drop_30d_mean=("Avg Drop 30d Raw", "mean"),
                _sell_speed_mean=("Time To Sell Inverse Raw", "mean"),
                _volatility_mean=("Volatility Raw", "mean"),
            )
            .reset_index()
        )
        grouped = grouped[grouped["_listing_count"] >= min_count].copy()
        if grouped.empty:
            tables.append({"keys": keys, "table": grouped})
            continue

        grouped[MARKET_COUNT_FEATURE_COLUMN] = np.log1p(grouped["_listing_count"].astype(float))
        grouped[MARKET_MEDIAN_PRICE_FEATURE_COLUMN] = grouped["_median_price"].astype(float)
        grouped[MARKET_PRESSURE_FEATURE_COLUMN] = grouped["_pressure_mean"].astype(float)
        grouped[MARKET_DROP_30D_FEATURE_COLUMN] = grouped["_drop_30d_mean"].astype(float)
        grouped[MARKET_SELL_SPEED_FEATURE_COLUMN] = grouped["_sell_speed_mean"].astype(float)
        grouped[MARKET_VOLATILITY_FEATURE_COLUMN] = grouped["_volatility_mean"].astype(float)
        grouped = grouped[keys + MARKET_HISTORY_FEATURE_COLUMNS]
        tables.append({"keys": keys, "table": grouped})

    return MarketHistoryBundle(tables=tables, defaults=defaults, variant_top_values=variant_top_values)


def _fit_oof_market_history_features(
    train_df: pd.DataFrame,
    pressure_targets: pd.DataFrame,
    groups: pd.Series,
) -> tuple[pd.DataFrame, MarketHistoryBundle]:
    final_bundle = _fit_market_history_bundle(train_df, pressure_targets)
    oof = pd.DataFrame(index=train_df.index, columns=MARKET_HISTORY_FEATURE_COLUMNS, dtype=float)

    splitter = _build_group_splitter(groups, INNER_FOLDS)
    if splitter is None:
        return final_bundle.predict_features(train_df), final_bundle

    for inner_train_idx, hold_idx in splitter.split(train_df, groups=groups):
        inner_train = train_df.iloc[inner_train_idx].reset_index(drop=True)
        inner_bundle = _fit_market_history_bundle(inner_train, pressure_targets)
        hold_features = inner_bundle.predict_features(train_df.iloc[hold_idx].reset_index(drop=True))
        hold_features.index = hold_idx
        oof.loc[hold_idx, MARKET_HISTORY_FEATURE_COLUMNS] = hold_features[MARKET_HISTORY_FEATURE_COLUMNS].to_numpy()

    missing = oof[MARKET_HISTORY_FEATURE_COLUMNS[0]].isna()
    if missing.any():
        fill_features = final_bundle.predict_features(train_df.iloc[np.flatnonzero(missing)].reset_index(drop=True))
        oof.loc[missing, MARKET_HISTORY_FEATURE_COLUMNS] = fill_features[MARKET_HISTORY_FEATURE_COLUMNS].to_numpy()

    for column in MARKET_HISTORY_FEATURE_COLUMNS:
        oof[column] = pd.to_numeric(oof[column], errors="coerce").fillna(final_bundle.defaults[column]).astype(float)

    return oof.reset_index(drop=True), final_bundle


def _fit_pressure_bundle(
    train_df: pd.DataFrame,
    pressure_targets: pd.DataFrame,
) -> PressureScoreBundle:
    variant_top_values = (
        train_df["Variant"].fillna("Unknown").astype(str).value_counts().head(TOP_N_VARIANTS).index.tolist()
    )
    pressure_input = train_df.copy()
    pressure_input[VARIANT_GROUPED_COLUMN] = _group_variants(pressure_input["Variant"], variant_top_values)

    train_ids = train_df[LISTING_ID_COLUMN].astype(str)
    score_map: dict[str, float] = {}
    if not pressure_targets.empty:
        train_targets = pressure_targets[pressure_targets[LISTING_ID_COLUMN].isin(train_ids)].copy()
        if not train_targets.empty:
            normalizer = _fit_pressure_normalizer(train_targets)
            normalized_targets = _apply_pressure_normalizer(train_targets, normalizer)
            score_map = (
                normalized_targets.drop_duplicates(subset=[LISTING_ID_COLUMN], keep="last")
                .set_index(LISTING_ID_COLUMN)[PRESSURE_FEATURE_COLUMN]
                .to_dict()
            )

    score_series = train_ids.map(score_map)
    default_score = float(score_series.dropna().median()) if score_series.notna().any() else 0.0

    fit_mask = score_series.notna()
    schema_source = pressure_input.loc[fit_mask].copy() if fit_mask.any() else pressure_input.copy()
    schema = _fit_feature_schema(schema_source, PRESSURE_NUMERIC_COLUMNS, PRESSURE_CATEGORICAL_COLUMNS)

    if fit_mask.sum() < 50 or score_series.loc[fit_mask].nunique() < 2:
        return PressureScoreBundle(
            model=None,
            schema=schema,
            variant_top_values=variant_top_values,
            default_score=default_score,
        )

    X_train = _apply_feature_schema(pressure_input.loc[fit_mask], schema)
    model = _build_pressure_model()
    model.fit(X_train, score_series.loc[fit_mask].to_numpy(), categorical_feature=PRESSURE_CATEGORICAL_COLUMNS)

    return PressureScoreBundle(
        model=model,
        schema=schema,
        variant_top_values=variant_top_values,
        default_score=default_score,
    )


def _fit_oof_pressure_scores(
    train_df: pd.DataFrame,
    pressure_targets: pd.DataFrame,
    groups: pd.Series,
) -> tuple[np.ndarray, PressureScoreBundle]:
    final_bundle = _fit_pressure_bundle(train_df, pressure_targets)
    oof = np.full(len(train_df), final_bundle.default_score, dtype=float)

    splitter = _build_group_splitter(groups, INNER_FOLDS)
    if splitter is None:
        return final_bundle.predict(train_df), final_bundle

    for inner_train_idx, hold_idx in splitter.split(train_df, groups=groups):
        inner_train = train_df.iloc[inner_train_idx].reset_index(drop=True)
        inner_bundle = _fit_pressure_bundle(inner_train, pressure_targets)
        hold_pred = inner_bundle.predict(train_df.iloc[hold_idx].reset_index(drop=True))
        oof[hold_idx] = hold_pred

    missing = ~np.isfinite(oof)
    if missing.any():
        fill_pred = final_bundle.predict(train_df.iloc[np.flatnonzero(missing)].reset_index(drop=True))
        oof[np.flatnonzero(missing)] = fill_pred

    return np.clip(oof, 0.0, 1.0), final_bundle


def _fit_oof_categorical_encoding(
    train_df: pd.DataFrame,
    y_log: pd.Series,
    groups: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    encoded_train, final_encoder = encode_categoricals(train_df, fit=True, target=y_log)
    oof_model = np.full(len(train_df), final_encoder["global_target_mean"], dtype=float)
    oof_variant = np.full(len(train_df), final_encoder["global_target_mean"], dtype=float)

    splitter = _build_group_splitter(groups, INNER_FOLDS)
    if splitter is None:
        return encoded_train, final_encoder

    y_values = np.asarray(y_log, dtype=float)
    for inner_train_idx, hold_idx in splitter.split(train_df, groups=groups):
        inner_train = train_df.iloc[inner_train_idx].reset_index(drop=True)
        _, inner_encoder = encode_categoricals(inner_train, fit=True, target=y_values[inner_train_idx])
        hold_frame, _ = encode_categoricals(
            train_df.iloc[hold_idx].reset_index(drop=True),
            fit=False,
            encoder=inner_encoder,
        )
        oof_model[hold_idx] = hold_frame[MODEL_TE_COLUMN].to_numpy()
        oof_variant[hold_idx] = hold_frame[VARIANT_TE_COLUMN].to_numpy()

    encoded_train[MODEL_TE_COLUMN] = oof_model
    encoded_train[VARIANT_TE_COLUMN] = oof_variant
    return encoded_train, final_encoder


def train_model(
    df: pd.DataFrame,
    *,
    save_artifacts: bool = True,
    artifact_prefix: str = "leakage_safe",
    model_dir: str = MODELS_DIR,
) -> dict[str, Any]:
    cleaned = clean_features(df)
    pressure_targets = create_price_pressure_score(cleaned)
    anchor = _select_anchor_rows(cleaned)
    if anchor.empty:
        raise RuntimeError("No valid listing-time rows available after cleaning.")

    groups = anchor[LISTING_ID_COLUMN].astype(str)
    y = anchor[TARGET_PRICE_COLUMN].astype(float)
    y_log = pd.Series(np.log(np.clip(y.to_numpy(), 1.0, None)), index=anchor.index)

    print(
        f"Leakage-safe training rows: {len(anchor):,} first-listing rows "
        f"from {groups.nunique():,} unique listings"
    )

    fold_metrics: list[dict[str, float | int]] = []
    best_iterations: list[int] = []
    splitter = _build_group_splitter(groups, OUTER_FOLDS)
    if splitter is not None:
        for fold_id, (train_idx, val_idx) in enumerate(splitter.split(anchor, groups=groups), start=1):
            train_fold = anchor.iloc[train_idx].reset_index(drop=True)
            val_fold = anchor.iloc[val_idx].reset_index(drop=True)
            train_groups = groups.iloc[train_idx].reset_index(drop=True)
            y_train_log = pd.Series(np.log(np.clip(train_fold[TARGET_PRICE_COLUMN].to_numpy(), 1.0, None)))
            y_val = val_fold[TARGET_PRICE_COLUMN].to_numpy(dtype=float)
            y_val_log = pd.Series(np.log(np.clip(y_val, 1.0, None)))

            train_market_features, market_bundle = _fit_oof_market_history_features(
                train_fold,
                pressure_targets,
                train_groups,
            )
            val_market_features = market_bundle.predict_features(val_fold)
            train_pressure_oof, pressure_bundle = _fit_oof_pressure_scores(train_fold, pressure_targets, train_groups)
            val_pressure = pressure_bundle.predict(val_fold)

            train_input = train_fold.copy()
            val_input = val_fold.copy()
            for column in MARKET_HISTORY_FEATURE_COLUMNS:
                train_input[column] = train_market_features[column].to_numpy()
                val_input[column] = val_market_features[column].to_numpy()
            train_input[PRESSURE_FEATURE_COLUMN] = train_pressure_oof
            val_input[PRESSURE_FEATURE_COLUMN] = val_pressure

            train_encoded, encoder = _fit_oof_categorical_encoding(train_input, y_train_log, train_groups)
            val_encoded, _ = encode_categoricals(val_input, fit=False, encoder=encoder)
            val_encoded[PRESSURE_FEATURE_COLUMN] = val_pressure

            schema = _fit_feature_schema(train_encoded, MAIN_NUMERIC_COLUMNS, MAIN_CATEGORICAL_COLUMNS)
            X_train = _apply_feature_schema(train_encoded, schema)
            X_val = _apply_feature_schema(val_encoded, schema)

            model, best_iteration = _fit_main_regressor(
                X_train,
                y_train_log,
                categorical_features=MAIN_CATEGORICAL_COLUMNS,
                X_valid=X_val,
                y_valid_log=y_val_log,
            )
            best_iterations.append(best_iteration)

            pred_val = np.exp(np.asarray(model.predict(X_val), dtype=float).ravel())
            eval_mask = val_fold[DAYS_SINCE_LISTING_COLUMN].to_numpy() <= 1.0
            if not eval_mask.any():
                eval_mask = np.ones(len(val_fold), dtype=bool)

            metrics = _metric_summary(y_val[eval_mask], pred_val[eval_mask])
            fold_metrics.append(
                {
                    "fold": fold_id,
                    "rows": int(eval_mask.sum()),
                    "mape": metrics["mape"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "mae": metrics["mae"],
                    "median_ae": metrics["median_ae"],
                    "mpe": metrics["mpe"],
                }
            )
            print(
                f"Fold {fold_id}: rows={int(eval_mask.sum()):,} "
                f"MAPE={metrics['mape']:.2f}% RMSE={metrics['rmse']:,.0f} "
                f"R2={metrics['r2']:.4f} MAE={metrics['mae']:,.0f}"
            )

    full_market_oof, full_market_bundle = _fit_oof_market_history_features(anchor, pressure_targets, groups)
    full_pressure_oof, full_pressure_bundle = _fit_oof_pressure_scores(anchor, pressure_targets, groups)
    final_input = anchor.copy()
    for column in MARKET_HISTORY_FEATURE_COLUMNS:
        final_input[column] = full_market_oof[column].to_numpy()
    final_input[PRESSURE_FEATURE_COLUMN] = full_pressure_oof
    final_encoded, final_encoder = _fit_oof_categorical_encoding(final_input, y_log, groups)
    for column in MARKET_HISTORY_FEATURE_COLUMNS:
        final_encoded[column] = full_market_oof[column].to_numpy()
    final_encoded[PRESSURE_FEATURE_COLUMN] = full_pressure_oof

    final_schema = _fit_feature_schema(final_encoded, MAIN_NUMERIC_COLUMNS, MAIN_CATEGORICAL_COLUMNS)
    X_full = _apply_feature_schema(final_encoded, final_schema)

    final_n_estimators = int(np.median(best_iterations)) if best_iterations else 1500
    final_model, _ = _fit_main_regressor(
        X_full,
        y_log,
        categorical_features=MAIN_CATEGORICAL_COLUMNS,
        n_estimators=final_n_estimators,
    )
    bundle = TrainingBundle(
        model=final_model,
        schema=final_schema,
        categorical_encoder=final_encoder,
        pressure_bundle=full_pressure_bundle,
        market_history_bundle=full_market_bundle,
    )

    feature_importance = (
        pd.DataFrame(
            {
                "feature": list(X_full.columns),
                "importance": np.asarray(final_model.feature_importances_, dtype=float),
            }
        )
        .sort_values("importance", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    cv_metrics = pd.DataFrame(fold_metrics)
    summary = {
        "target_transform": TARGET_TRANSFORM,
        "training_rows": int(len(anchor)),
        "unique_listings": int(groups.nunique()),
        "feature_columns": list(X_full.columns),
        "categorical_columns": list(MAIN_CATEGORICAL_COLUMNS),
        "cv_mape_mean": float(cv_metrics["mape"].mean()) if not cv_metrics.empty else np.nan,
        "cv_rmse_mean": float(cv_metrics["rmse"].mean()) if not cv_metrics.empty else np.nan,
        "cv_r2_mean": float(cv_metrics["r2"].mean()) if not cv_metrics.empty else np.nan,
        "cv_mae_mean": float(cv_metrics["mae"].mean()) if not cv_metrics.empty else np.nan,
        "cv_median_ae_mean": float(cv_metrics["median_ae"].mean()) if not cv_metrics.empty else np.nan,
        "cv_mpe_mean": float(cv_metrics["mpe"].mean()) if not cv_metrics.empty else np.nan,
    }

    if save_artifacts:
        os.makedirs(model_dir, exist_ok=True)
        bundle_path = os.path.join(model_dir, f"{artifact_prefix}_model_bundle.joblib")
        importance_path = os.path.join(model_dir, f"{artifact_prefix}_feature_importance.csv")
        metrics_path = os.path.join(model_dir, f"{artifact_prefix}_cv_metrics.csv")
        summary_path = os.path.join(model_dir, f"{artifact_prefix}_training_summary.joblib")

        joblib.dump(bundle, bundle_path)
        feature_importance.to_csv(importance_path, index=False, encoding="utf-8-sig")
        cv_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        joblib.dump(summary, summary_path)

        print(f"Saved bundle to {bundle_path}")
        print(f"Saved feature importance to {importance_path}")
        print(f"Saved CV metrics to {metrics_path}")
        print(f"Saved summary to {summary_path}")

    print("\nTop features:")
    print(feature_importance.head(15).to_string(index=False))

    return {
        "bundle": bundle,
        "cv_metrics": cv_metrics,
        "feature_importance": feature_importance,
        "summary": summary,
    }


def main() -> None:
    history = _load_training_history()
    train_model(history)


if __name__ == "__main__":
    main()
