from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

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

TARGET_PRICE_COLUMN = "Target Price"
LISTING_ID_COLUMN = "Listing ID"
LISTING_DATE_COLUMN = "Listing Date"
VARIANT_GROUPED_COLUMN = "variant_grouped"
PRICE_ANCHOR_COLUMN = "median_price_model_city"
PRICE_SEGMENT_COLUMN = "price_segment"
MODEL_TE_COLUMN = "model_target_encode"
VARIANT_TE_COLUMN = "variant_target_encode"
TARGET_TRANSFORM = "log_price"

TOP_N_VARIANTS = 250
TARGET_ENCODING_SMOOTHING = 50.0
ANCHOR_MIN_MODEL_CITY = 15
ANCHOR_MIN_MODEL = 8
ANCHOR_MIN_MAKE = 5
OUTER_FOLDS = 5
INNER_FOLDS = 4
PRICE_SEGMENT_LABELS = ["<=5L", "5-10L", "10-20L", ">20L"]

NUMERIC_FEATURE_COLUMNS = [
    "Car Age",
    "KMs Driven",
    "KMs/Year",
    "Log KMs",
    "Ownership",
    "Age x Ownership",
    "KMs x Ownership",
    PRICE_ANCHOR_COLUMN,
    MODEL_TE_COLUMN,
    VARIANT_TE_COLUMN,
]

CATEGORICAL_FEATURE_COLUMNS = [
    "Make",
    VARIANT_GROUPED_COLUMN,
    "Transmission",
    "Fuel",
    "BodyType",
    "City",
    "Reg State",
    PRICE_SEGMENT_COLUMN,
]


@dataclass
class FeatureSchema:
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_fill_values: dict[str, float]
    categorical_levels: dict[str, list[str]]


@dataclass
class PriceAnchorBundle:
    model_city_stats: pd.DataFrame
    model_stats: pd.DataFrame
    make_stats: pd.DataFrame
    global_median: float
    min_model_city_count: int = ANCHOR_MIN_MODEL_CITY
    min_model_count: int = ANCHOR_MIN_MODEL
    min_make_count: int = ANCHOR_MIN_MAKE

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        base_index = work.index

        model_city = (
            work.reset_index()[["index", "Model", "City"]]
            .merge(self.model_city_stats, on=["Model", "City"], how="left")
            .set_index("index")
            .reindex(base_index)
        )
        model_only = (
            work.reset_index()[["index", "Model"]]
            .merge(self.model_stats, on="Model", how="left")
            .set_index("index")
            .reindex(base_index)
        )
        make_only = (
            work.reset_index()[["index", "Make"]]
            .merge(self.make_stats, on="Make", how="left")
            .set_index("index")
            .reindex(base_index)
        )

        anchor = pd.Series(np.nan, index=base_index, dtype=float)
        model_city_ok = pd.to_numeric(model_city["count"], errors="coerce") >= self.min_model_city_count
        model_ok = pd.to_numeric(model_only["count"], errors="coerce") >= self.min_model_count
        make_ok = pd.to_numeric(make_only["count"], errors="coerce") >= self.min_make_count

        anchor.loc[model_city_ok] = pd.to_numeric(model_city.loc[model_city_ok, "median"], errors="coerce")
        anchor.loc[anchor.isna() & model_ok] = pd.to_numeric(
            model_only.loc[anchor.isna() & model_ok, "median"],
            errors="coerce",
        )
        anchor.loc[anchor.isna() & make_ok] = pd.to_numeric(
            make_only.loc[anchor.isna() & make_ok, "median"],
            errors="coerce",
        )
        anchor = anchor.fillna(float(self.global_median)).clip(lower=1.0)

        work[PRICE_ANCHOR_COLUMN] = anchor.astype(float)
        work[PRICE_SEGMENT_COLUMN] = _bucket_price(anchor)
        return work


@dataclass
class TargetEncodingBundle:
    top_variants: list[str]
    global_mean: float
    model_target_means: dict[str, float]
    variant_target_means: dict[str, float]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        work[VARIANT_GROUPED_COLUMN] = _group_variants(work["Variant"], self.top_variants)
        work[MODEL_TE_COLUMN] = (
            work["Model"].map(self.model_target_means).fillna(self.global_mean).astype(float)
        )
        work[VARIANT_TE_COLUMN] = (
            work[VARIANT_GROUPED_COLUMN].map(self.variant_target_means).fillna(self.global_mean).astype(float)
        )
        return work


@dataclass
class CalibrationBundle:
    slope: float
    intercept: float
    global_factor: float
    segment_factors: dict[str, float]

    def apply(self, preds: np.ndarray, segments: pd.Series | None = None) -> np.ndarray:
        raw = np.asarray(preds, dtype=float).ravel().copy()
        if segments is None:
            factors = np.full(len(raw), self.global_factor, dtype=float)
        else:
            seg_series = pd.Series(segments, index=np.arange(len(raw)), dtype="object").fillna("Unknown")
            factors = seg_series.map(self.segment_factors).fillna(self.global_factor).to_numpy(dtype=float)

        adjusted = raw * factors
        calibrated = self.slope * adjusted + self.intercept
        return np.clip(calibrated, 1.0, None)


@dataclass
class AnchoredPriceModelBundle:
    model: LGBMRegressor
    schema: FeatureSchema
    anchor_bundle: PriceAnchorBundle
    target_encoder: TargetEncodingBundle
    calibration_bundle: CalibrationBundle
    target_transform: str = TARGET_TRANSFORM

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        work = _prepare_rows(df, allow_missing_target=True, prediction_mode=True)
        work, _ = create_price_anchor(work, bundle=self.anchor_bundle)
        work, _ = apply_kfold_target_encoding(work, bundle=self.target_encoder)
        return _apply_feature_schema(work, self.schema)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        work = _prepare_rows(df, allow_missing_target=True, prediction_mode=True)
        work, _ = create_price_anchor(work, bundle=self.anchor_bundle)
        work, _ = apply_kfold_target_encoding(work, bundle=self.target_encoder)
        X = _apply_feature_schema(work, self.schema)
        raw_pred = np.exp(np.asarray(self.model.predict(X), dtype=float).ravel())
        return self.calibration_bundle.apply(raw_pred, work[PRICE_SEGMENT_COLUMN])


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
        raise RuntimeError(f"No training files found in {DATA_DIR}")
    return pd.concat([_read_source_file(path) for path in files], ignore_index=True)


def _build_splitter(groups: pd.Series | None, n_splits: int):
    if groups is None:
        if n_splits < 2:
            return None
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)

    groups_series = pd.Series(groups, dtype="string").fillna("missing")
    unique_groups = int(groups_series.nunique())
    n_splits = min(n_splits, unique_groups)
    if n_splits < 2:
        return None
    return GroupKFold(n_splits=n_splits)


def _iter_split(splitter, X: pd.DataFrame, groups: pd.Series | None):
    if splitter is None:
        yield np.arange(len(X)), np.arange(len(X))
        return

    if isinstance(splitter, GroupKFold):
        yield from splitter.split(X, groups=groups.astype(str))
        return

    yield from splitter.split(X)


def _group_variants(series: pd.Series, top_variants: list[str]) -> pd.Series:
    top_set = set(top_variants)
    values = series.fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
    return values.where(values.isin(top_set), "other")


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


def _bucket_price(price_like: pd.Series | np.ndarray) -> pd.Series:
    values = pd.to_numeric(pd.Series(price_like), errors="coerce").fillna(0.0)
    bins = [-np.inf, 500_000, 1_000_000, 2_000_000, np.inf]
    return pd.cut(values, bins=bins, labels=PRICE_SEGMENT_LABELS, include_lowest=True).astype(str)


def _prepare_rows(
    df: pd.DataFrame,
    *,
    allow_missing_target: bool = False,
    prediction_mode: bool = False,
    reference_datetime: datetime | None = None,
) -> pd.DataFrame:
    work = standardize_columns(df.copy())
    work["_row_order"] = np.arange(len(work))
    ref_dt = pd.Timestamp(reference_datetime or datetime.now())

    if "Name" in work.columns and "Make" not in work.columns:
        makes, models = _split_name_to_make_model(work["Name"])
        work["Make"] = makes
        work["Model"] = models

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
    listing_id = listing_id.where(
        ~missing_listing_id,
        pd.Series([f"row_{idx}" for idx in work.index], index=work.index),
    )
    work[LISTING_ID_COLUMN] = listing_id.astype(str)

    if "Transmission" in work.columns:
        work["Transmission"] = (
            work["Transmission"].astype(str).str.strip().map(_TRANSMISSION_MAP).fillna(work["Transmission"])
        )

    snapshot_year = work[LISTING_DATE_COLUMN].dt.year.fillna(float(ref_dt.year))
    work["Car Age"] = (snapshot_year - work["Year"]).clip(lower=0).astype(float)
    work["KMs/Year"] = np.where(work["Car Age"] > 0, work["KMs Driven"] / work["Car Age"], work["KMs Driven"])
    work["Log KMs"] = np.log1p(pd.to_numeric(work["KMs Driven"], errors="coerce").clip(lower=0))
    work["Age x Ownership"] = work["Car Age"] * work["Ownership"]
    work["KMs x Ownership"] = work["KMs Driven"] * work["Ownership"]

    reg_state_from_registration = work["Registration"].apply(_reg_state)
    if "Reg State" in work.columns:
        existing_reg_state = work["Reg State"].fillna("").astype(str).str.strip()
        work["Reg State"] = existing_reg_state.where(existing_reg_state != "", reg_state_from_registration)
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

    year_valid = (work["Year"] >= 2005) & (work["Year"] <= float(ref_dt.year))
    kms_valid = (work["KMs Driven"] >= 0) & (work["KMs Driven"] <= 400_000)

    if prediction_mode:
        work.loc[~year_valid, "Car Age"] = np.nan
        work.loc[~kms_valid, "KMs Driven"] = np.nan
        work.loc[~kms_valid, "KMs/Year"] = np.nan
        work.loc[~kms_valid, "Log KMs"] = np.nan
        work.loc[~year_valid | ~kms_valid, "Age x Ownership"] = np.nan
        work.loc[~kms_valid, "KMs x Ownership"] = np.nan
    elif allow_missing_target:
        work = work.dropna(subset=["Car Age", "KMs Driven"])
        work = work[year_valid & kms_valid]
    else:
        work = work.dropna(subset=[TARGET_PRICE_COLUMN, "Car Age", "KMs Driven"])
        work = work[work[TARGET_PRICE_COLUMN] > 0]
        work = work[year_valid & kms_valid]

    work = work.drop(columns=["Year", "Market Days", "Days On Market", "Fetch Month"], errors="ignore")
    return work.sort_values("_row_order", kind="mergesort").reset_index(drop=True)


def _fit_price_anchor_bundle(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_PRICE_COLUMN,
    min_model_city_count: int = ANCHOR_MIN_MODEL_CITY,
    min_model_count: int = ANCHOR_MIN_MODEL,
    min_make_count: int = ANCHOR_MIN_MAKE,
) -> PriceAnchorBundle:
    target = pd.to_numeric(df[target_col], errors="coerce")
    work = df.assign(_target=target).dropna(subset=["_target"]).copy()
    if work.empty:
        global_median = 1.0
        empty_model_city = pd.DataFrame(columns=["Model", "City", "median", "count"])
        empty_model = pd.DataFrame(columns=["Model", "median", "count"])
        empty_make = pd.DataFrame(columns=["Make", "median", "count"])
        return PriceAnchorBundle(
            model_city_stats=empty_model_city,
            model_stats=empty_model,
            make_stats=empty_make,
            global_median=global_median,
            min_model_city_count=min_model_city_count,
            min_model_count=min_model_count,
            min_make_count=min_make_count,
        )

    model_city = (
        work.groupby(["Model", "City"], observed=True)["_target"]
        .agg(["median", "count"])
        .reset_index()
    )
    model_only = work.groupby("Model", observed=True)["_target"].agg(["median", "count"]).reset_index()
    make_only = work.groupby("Make", observed=True)["_target"].agg(["median", "count"]).reset_index()
    global_median = float(work["_target"].median())

    return PriceAnchorBundle(
        model_city_stats=model_city,
        model_stats=model_only,
        make_stats=make_only,
        global_median=global_median,
        min_model_city_count=min_model_city_count,
        min_model_count=min_model_count,
        min_make_count=min_make_count,
    )


def create_price_anchor(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_PRICE_COLUMN,
    groups: pd.Series | None = None,
    bundle: PriceAnchorBundle | None = None,
    n_splits: int = INNER_FOLDS,
    min_model_city_count: int = ANCHOR_MIN_MODEL_CITY,
    min_model_count: int = ANCHOR_MIN_MODEL,
    min_make_count: int = ANCHOR_MIN_MAKE,
) -> tuple[pd.DataFrame, PriceAnchorBundle]:
    work = df.copy()

    if bundle is not None:
        return bundle.apply(work), bundle

    if target_col not in work.columns:
        raise ValueError(f"{target_col} is required when bundle is not provided")

    final_bundle = _fit_price_anchor_bundle(
        work,
        target_col=target_col,
        min_model_city_count=min_model_city_count,
        min_model_count=min_model_count,
        min_make_count=min_make_count,
    )

    splitter = _build_splitter(groups, n_splits)
    if splitter is None or len(work) < 2:
        return final_bundle.apply(work), final_bundle

    anchored = work.copy()
    anchored[PRICE_ANCHOR_COLUMN] = np.nan
    anchored[PRICE_SEGMENT_COLUMN] = "Unknown"

    for train_idx, hold_idx in _iter_split(splitter, work, groups):
        if len(train_idx) == 0 or len(hold_idx) == 0:
            continue
        fold_bundle = _fit_price_anchor_bundle(
            work.iloc[train_idx].reset_index(drop=True),
            target_col=target_col,
            min_model_city_count=min_model_city_count,
            min_model_count=min_model_count,
            min_make_count=min_make_count,
        )
        hold_frame = fold_bundle.apply(work.iloc[hold_idx].reset_index(drop=True))
        anchored.loc[work.index[hold_idx], PRICE_ANCHOR_COLUMN] = hold_frame[PRICE_ANCHOR_COLUMN].to_numpy()
        anchored.loc[work.index[hold_idx], PRICE_SEGMENT_COLUMN] = hold_frame[PRICE_SEGMENT_COLUMN].to_numpy()

    missing_anchor = anchored[PRICE_ANCHOR_COLUMN].isna()
    if missing_anchor.any():
        fallback = final_bundle.apply(work.loc[missing_anchor].copy())
        anchored.loc[missing_anchor, PRICE_ANCHOR_COLUMN] = fallback[PRICE_ANCHOR_COLUMN].to_numpy()
        anchored.loc[missing_anchor, PRICE_SEGMENT_COLUMN] = fallback[PRICE_SEGMENT_COLUMN].to_numpy()

    return anchored, final_bundle


def _fit_target_encoder(
    df: pd.DataFrame,
    target: pd.Series,
    *,
    top_n_variants: int = TOP_N_VARIANTS,
    smoothing: float = TARGET_ENCODING_SMOOTHING,
) -> TargetEncodingBundle:
    target_series = pd.Series(target, index=df.index, dtype=float)
    variant_counts = df["Variant"].fillna("Unknown").astype(str).value_counts()
    top_variants = variant_counts.head(min(top_n_variants, len(variant_counts))).index.tolist()
    grouped_variant = _group_variants(df["Variant"], top_variants)
    global_mean = float(target_series.mean()) if len(target_series) else 0.0
    model_target_means = _smoothed_target_mean(df["Model"], target_series, global_mean, smoothing)
    variant_target_means = _smoothed_target_mean(grouped_variant, target_series, global_mean, smoothing)
    return TargetEncodingBundle(
        top_variants=top_variants,
        global_mean=global_mean,
        model_target_means=model_target_means,
        variant_target_means=variant_target_means,
    )


def apply_kfold_target_encoding(
    df: pd.DataFrame,
    *,
    target: pd.Series | np.ndarray | None = None,
    groups: pd.Series | None = None,
    bundle: TargetEncodingBundle | None = None,
    n_splits: int = INNER_FOLDS,
    top_n_variants: int = TOP_N_VARIANTS,
    smoothing: float = TARGET_ENCODING_SMOOTHING,
) -> tuple[pd.DataFrame, TargetEncodingBundle]:
    work = df.copy()

    if bundle is not None:
        return bundle.apply(work), bundle

    if target is None:
        raise ValueError("target is required when bundle is not provided")

    target_series = pd.Series(target, index=work.index, dtype=float)
    final_bundle = _fit_target_encoder(work, target_series, top_n_variants=top_n_variants, smoothing=smoothing)

    splitter = _build_splitter(groups, n_splits)
    if splitter is None or len(work) < 2:
        return final_bundle.apply(work), final_bundle

    encoded = work.copy()
    encoded[VARIANT_GROUPED_COLUMN] = "other"
    encoded[MODEL_TE_COLUMN] = np.nan
    encoded[VARIANT_TE_COLUMN] = np.nan

    for train_idx, hold_idx in _iter_split(splitter, work, groups):
        if len(train_idx) == 0 or len(hold_idx) == 0:
            continue
        fold_bundle = _fit_target_encoder(
            work.iloc[train_idx].reset_index(drop=True),
            target_series.iloc[train_idx].reset_index(drop=True),
            top_n_variants=top_n_variants,
            smoothing=smoothing,
        )
        hold_frame = fold_bundle.apply(work.iloc[hold_idx].reset_index(drop=True))
        encoded.loc[work.index[hold_idx], VARIANT_GROUPED_COLUMN] = hold_frame[VARIANT_GROUPED_COLUMN].to_numpy()
        encoded.loc[work.index[hold_idx], MODEL_TE_COLUMN] = hold_frame[MODEL_TE_COLUMN].to_numpy()
        encoded.loc[work.index[hold_idx], VARIANT_TE_COLUMN] = hold_frame[VARIANT_TE_COLUMN].to_numpy()

    missing_te = encoded[MODEL_TE_COLUMN].isna() | encoded[VARIANT_TE_COLUMN].isna()
    if missing_te.any():
        fallback = final_bundle.apply(work.loc[missing_te].copy())
        encoded.loc[missing_te, VARIANT_GROUPED_COLUMN] = fallback[VARIANT_GROUPED_COLUMN].to_numpy()
        encoded.loc[missing_te, MODEL_TE_COLUMN] = fallback[MODEL_TE_COLUMN].to_numpy()
        encoded.loc[missing_te, VARIANT_TE_COLUMN] = fallback[VARIANT_TE_COLUMN].to_numpy()

    return encoded, final_bundle


def _fit_feature_schema(df: pd.DataFrame) -> FeatureSchema:
    numeric_fill_values: dict[str, float] = {}
    for column in NUMERIC_FEATURE_COLUMNS:
        values = pd.to_numeric(df.get(column, pd.Series(dtype=float)), errors="coerce")
        fill_value = float(values.median()) if values.notna().any() else 0.0
        numeric_fill_values[column] = fill_value

    categorical_levels: dict[str, list[str]] = {}
    for column in CATEGORICAL_FEATURE_COLUMNS:
        values = df.get(column, pd.Series(["Unknown"], dtype="object")).fillna("Unknown").astype(str).str.strip()
        values = values.replace({"": "Unknown"})
        levels = sorted(values.unique().tolist())
        if "Unknown" not in levels:
            levels.append("Unknown")
        if column == VARIANT_GROUPED_COLUMN and "other" not in levels:
            levels.append("other")
        categorical_levels[column] = levels

    return FeatureSchema(
        numeric_columns=list(NUMERIC_FEATURE_COLUMNS),
        categorical_columns=list(CATEGORICAL_FEATURE_COLUMNS),
        numeric_fill_values=numeric_fill_values,
        categorical_levels=categorical_levels,
    )


def _apply_feature_schema(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    work = df.copy()
    for column in schema.numeric_columns:
        work[column] = (
            pd.to_numeric(work.get(column, pd.Series([np.nan] * len(work), index=work.index)), errors="coerce")
            .fillna(schema.numeric_fill_values[column])
            .astype(float)
        )

    for column in schema.categorical_columns:
        levels = schema.categorical_levels[column]
        fallback = "other" if column == VARIANT_GROUPED_COLUMN and "other" in levels else "Unknown"
        values = work.get(column, pd.Series([fallback] * len(work), index=work.index))
        values = values.fillna(fallback).astype(str).str.strip().replace({"": fallback})
        values = values.where(values.isin(levels), fallback)
        work[column] = pd.Categorical(values, categories=levels)

    return work[schema.numeric_columns + schema.categorical_columns]


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    if not valid.any():
        return {
            "mape": np.nan,
            "mpe": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "mae": np.nan,
            "median_ae": np.nan,
        }

    a = y_true[valid]
    p = y_pred[valid]
    ape = np.abs(p - a) / np.clip(np.abs(a), 1.0, None)
    pe = (p - a) / np.clip(np.abs(a), 1.0, None)
    mse = mean_squared_error(a, p)
    return {
        "mape": float(np.mean(ape) * 100.0),
        "mpe": float(np.mean(pe) * 100.0),
        "rmse": float(sqrt(mse)),
        "r2": float(r2_score(a, p)),
        "mae": float(mean_absolute_error(a, p)),
        "median_ae": float(median_absolute_error(a, p)),
    }


def _price_space_mape_from_log(y_true_log: np.ndarray, y_pred_log: np.ndarray):
    actual = np.exp(np.asarray(y_true_log, dtype=float))
    pred = np.exp(np.asarray(y_pred_log, dtype=float))
    ape = np.abs(pred - actual) / np.clip(actual, 1.0, None)
    return "price_mape", float(np.mean(ape)), False


def _build_sample_weight(price: np.ndarray) -> np.ndarray:
    raw_weight = 1.0 / (np.asarray(price, dtype=float).ravel() + 1.0)
    raw_weight = np.clip(raw_weight, np.finfo(float).eps, None)
    return raw_weight / raw_weight.mean()


def _calibration_objective(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict[str, float]]:
    metrics = _regression_metrics(y_true, y_pred)
    mpe = float(metrics["mpe"])
    mape = float(metrics["mape"])
    penalty = max(abs(mpe) - 5.0, 0.0) * 25.0
    objective = mape + penalty
    return objective, metrics


def _fit_regressor(
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    sample_weight: np.ndarray,
    *,
    X_valid: pd.DataFrame | None = None,
    y_valid_log: np.ndarray | None = None,
    n_estimators: int = 3000,
) -> tuple[LGBMRegressor, int]:
    model = LGBMRegressor(
        objective="regression",
        metric="mape",
        n_estimators=n_estimators,
        num_leaves=64,
        learning_rate=0.03,
        min_data_in_leaf=150,
        feature_fraction=0.7,
        lambda_l1=5.0,
        lambda_l2=5.0,
        importance_type="gain",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    fit_kwargs: dict[str, Any] = {
        "X": X_train,
        "y": np.asarray(y_train_log, dtype=float),
        "sample_weight": np.asarray(sample_weight, dtype=float),
        "categorical_feature": CATEGORICAL_FEATURE_COLUMNS,
    }
    if X_valid is not None and y_valid_log is not None:
        fit_kwargs["eval_set"] = [(X_valid, np.asarray(y_valid_log, dtype=float))]
        fit_kwargs["eval_metric"] = _price_space_mape_from_log
        fit_kwargs["callbacks"] = [
            lgb_early_stopping(stopping_rounds=200, first_metric_only=True, verbose=False),
            lgb_log_evaluation(period=0),
        ]

    model.fit(**fit_kwargs)
    best_iteration = int(getattr(model, "best_iteration_", 0) or n_estimators)
    return model, best_iteration


def calibrate_predictions(
    preds: np.ndarray,
    y_true: np.ndarray,
    segments: pd.Series | None = None,
) -> CalibrationBundle:
    raw_pred = np.asarray(preds, dtype=float).ravel()
    actual = np.asarray(y_true, dtype=float).ravel()
    valid = np.isfinite(raw_pred) & np.isfinite(actual) & (raw_pred > 0) & (actual > 0)
    if not valid.any():
        return CalibrationBundle(slope=1.0, intercept=0.0, global_factor=1.0, segment_factors={})

    if segments is None:
        segment_values = pd.Series(["all"] * int(valid.sum()))
    else:
        segment_values = pd.Series(segments, index=np.arange(len(raw_pred)), dtype="object").iloc[valid].fillna("Unknown")

    frame = pd.DataFrame(
        {
            "pred": raw_pred[valid],
            "actual": actual[valid],
            "segment": segment_values.to_numpy(),
        }
    )
    frame["ratio"] = frame["actual"] / np.clip(frame["pred"], 1.0, None)

    global_factor = float(frame["ratio"].median())
    segment_factors: dict[str, float] = {}
    for segment, group in frame.groupby("segment", sort=False):
        factor = float(group["ratio"].median()) if len(group) >= 20 else global_factor
        segment_factors[str(segment)] = float(np.clip(factor, 0.4, 1.6))

    actual_values = frame["actual"].to_numpy(dtype=float)
    raw_values = frame["pred"].to_numpy(dtype=float)
    segment_adjusted = frame["pred"] * frame["segment"].map(segment_factors).fillna(global_factor)
    linear = LinearRegression()
    linear.fit(segment_adjusted.to_numpy().reshape(-1, 1), frame["actual"].to_numpy())

    slope = float(linear.coef_[0]) if np.isfinite(linear.coef_[0]) else 1.0
    intercept = float(linear.intercept_) if np.isfinite(linear.intercept_) else 0.0
    linear_adjusted = np.clip(slope * segment_adjusted.to_numpy() + intercept, 1.0, None)

    candidates = [
        {
            "name": "identity",
            "bundle": CalibrationBundle(slope=1.0, intercept=0.0, global_factor=1.0, segment_factors={}),
            "pred": raw_values,
        },
        {
            "name": "global_factor",
            "bundle": CalibrationBundle(
                slope=1.0,
                intercept=0.0,
                global_factor=float(np.clip(global_factor, 0.4, 1.6)),
                segment_factors={},
            ),
            "pred": np.clip(raw_values * float(np.clip(global_factor, 0.4, 1.6)), 1.0, None),
        },
        {
            "name": "segment_factor",
            "bundle": CalibrationBundle(
                slope=1.0,
                intercept=0.0,
                global_factor=float(np.clip(global_factor, 0.4, 1.6)),
                segment_factors=segment_factors,
            ),
            "pred": segment_adjusted.to_numpy(dtype=float),
        },
        {
            "name": "segment_factor_linear",
            "bundle": CalibrationBundle(
                slope=slope,
                intercept=intercept,
                global_factor=float(np.clip(global_factor, 0.4, 1.6)),
                segment_factors=segment_factors,
            ),
            "pred": linear_adjusted,
        },
    ]

    scored: list[tuple[bool, float, float, dict[str, float], CalibrationBundle]] = []
    for candidate in candidates:
        objective, metrics = _calibration_objective(actual_values, candidate["pred"])
        within_bias = bool(np.isfinite(metrics["mpe"]) and abs(metrics["mpe"]) <= 5.0)
        scored.append((within_bias, objective, float(metrics["mape"]), metrics, candidate["bundle"]))

    scored.sort(key=lambda item: (0 if item[0] else 1, item[1], item[2]))
    return scored[0][4]


def train_model(
    df: pd.DataFrame,
    *,
    save_artifacts: bool = True,
    artifact_prefix: str = "anchored_price",
    model_dir: str = MODELS_DIR,
) -> dict[str, Any]:
    print("Preparing leakage-safe training rows...")
    prepared = _prepare_rows(df, allow_missing_target=False)
    if prepared.empty:
        raise RuntimeError("No valid rows available after cleaning.")

    y = prepared[TARGET_PRICE_COLUMN].astype(float).to_numpy()

    # Train on log(price), not raw price.
    # Used-car prices are right-skewed and the dominant effects are multiplicative:
    # brand premiums, mileage depreciation, and age decay behave more like
    # percentage moves than additive rupee moves. Learning in log space reduces
    # variance, stabilizes the objective, and is the most direct fix for the
    # mean-regression bias between cheap and expensive cars.
    y_log = np.log(np.clip(y, 1.0, None))

    weights = _build_sample_weight(y)
    groups = prepared[LISTING_ID_COLUMN].astype(str)
    print(
        f"Training on {len(prepared):,} rows from {groups.nunique():,} unique listings "
        f"with target transform={TARGET_TRANSFORM}"
    )
    print(
        f"Sample weight range: {weights.min():.6f} to {weights.max():.6f} "
        f"(mean {weights.mean():.6f})"
    )

    splitter = _build_splitter(groups, OUTER_FOLDS)
    fold_metrics: list[dict[str, float | int]] = []
    best_iterations: list[int] = []
    oof_raw_pred = np.full(len(prepared), np.nan, dtype=float)
    oof_price_segment = pd.Series(["Unknown"] * len(prepared), index=prepared.index, dtype="object")

    if splitter is not None:
        print(f"Running {OUTER_FOLDS}-fold GroupKFold validation...")
        for fold_id, (train_idx, val_idx) in enumerate(_iter_split(splitter, prepared, groups), start=1):
            print(
                f"\nFold {fold_id}/{OUTER_FOLDS}: "
                f"building anchors and encodings for {len(train_idx):,} train / {len(val_idx):,} valid rows"
            )
            train_fold = prepared.iloc[train_idx].reset_index(drop=True)
            val_fold = prepared.iloc[val_idx].reset_index(drop=True)
            train_groups = groups.iloc[train_idx].reset_index(drop=True)

            train_with_anchor, anchor_bundle = create_price_anchor(
                train_fold,
                groups=train_groups,
                target_col=TARGET_PRICE_COLUMN,
            )
            val_with_anchor, _ = create_price_anchor(val_fold, bundle=anchor_bundle)

            train_enriched, encoder_bundle = apply_kfold_target_encoding(
                train_with_anchor,
                target=np.log(train_fold[TARGET_PRICE_COLUMN].astype(float).to_numpy()),
                groups=train_groups,
            )
            val_enriched, _ = apply_kfold_target_encoding(val_with_anchor, bundle=encoder_bundle)

            schema = _fit_feature_schema(train_enriched)
            X_train = _apply_feature_schema(train_enriched, schema)
            X_val = _apply_feature_schema(val_enriched, schema)
            y_train = train_fold[TARGET_PRICE_COLUMN].astype(float).to_numpy()
            y_val = val_fold[TARGET_PRICE_COLUMN].astype(float).to_numpy()
            y_train_log = np.log(np.clip(y_train, 1.0, None))
            y_val_log = np.log(np.clip(y_val, 1.0, None))
            fold_weights = _build_sample_weight(y_train)
            print(
                f"Fold {fold_id}/{OUTER_FOLDS}: fitting LightGBM on "
                f"{X_train.shape[1]} features"
            )

            model, best_iteration = _fit_regressor(
                X_train,
                y_train_log,
                fold_weights,
                X_valid=X_val,
                y_valid_log=y_val_log,
            )
            best_iterations.append(best_iteration)

            raw_val_pred = np.exp(np.asarray(model.predict(X_val), dtype=float).ravel())
            oof_raw_pred[val_idx] = raw_val_pred
            oof_price_segment.iloc[val_idx] = val_enriched[PRICE_SEGMENT_COLUMN].astype(str).to_numpy()

            metrics = _regression_metrics(y_val, raw_val_pred)
            fold_metrics.append(
                {
                    "fold": fold_id,
                    "rows": int(len(val_idx)),
                    "mape": metrics["mape"],
                    "mpe": metrics["mpe"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "mae": metrics["mae"],
                    "median_ae": metrics["median_ae"],
                }
            )
            print(
                f"Fold {fold_id}: rows={len(val_idx):,} "
                f"MAPE={metrics['mape']:.2f}% MPE={metrics['mpe']:+.2f}% "
                f"RMSE={metrics['rmse']:,.0f} R2={metrics['r2']:.4f}"
            )
    else:
        print("Skipping cross-validation because there are not enough unique groups.")

    print("\nFitting calibration on out-of-fold predictions...")
    calibration_bundle = calibrate_predictions(oof_raw_pred, y, oof_price_segment)
    oof_calibrated = calibration_bundle.apply(oof_raw_pred, oof_price_segment)
    raw_oof_metrics = _regression_metrics(y, oof_raw_pred)
    calibrated_oof_metrics = _regression_metrics(y, oof_calibrated)

    print("Building final anchor and encoding bundles on the full training data...")
    _, final_anchor_bundle = create_price_anchor(prepared, target_col=TARGET_PRICE_COLUMN, groups=groups)
    full_anchor_frame, _ = create_price_anchor(prepared, bundle=final_anchor_bundle)
    _, final_encoder_bundle = apply_kfold_target_encoding(
        full_anchor_frame,
        target=y_log,
        groups=groups,
    )
    full_train_frame, _ = apply_kfold_target_encoding(full_anchor_frame, bundle=final_encoder_bundle)

    final_schema = _fit_feature_schema(full_train_frame)
    X_full = _apply_feature_schema(full_train_frame, final_schema)
    final_n_estimators = int(np.median(best_iterations)) if best_iterations else 1200
    print(
        f"Training final LightGBM on {len(X_full):,} rows, "
        f"{X_full.shape[1]} features, n_estimators={final_n_estimators}"
    )
    final_model, _ = _fit_regressor(
        X_full,
        y_log,
        weights,
        n_estimators=final_n_estimators,
    )

    bundle = AnchoredPriceModelBundle(
        model=final_model,
        schema=final_schema,
        anchor_bundle=final_anchor_bundle,
        target_encoder=final_encoder_bundle,
        calibration_bundle=calibration_bundle,
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
        "training_rows": int(len(prepared)),
        "unique_listings": int(groups.nunique()),
        "raw_oof_mape": raw_oof_metrics["mape"],
        "raw_oof_mpe": raw_oof_metrics["mpe"],
        "raw_oof_rmse": raw_oof_metrics["rmse"],
        "raw_oof_r2": raw_oof_metrics["r2"],
        "calibrated_oof_mape": calibrated_oof_metrics["mape"],
        "calibrated_oof_mpe": calibrated_oof_metrics["mpe"],
        "calibrated_oof_rmse": calibrated_oof_metrics["rmse"],
        "calibrated_oof_r2": calibrated_oof_metrics["r2"],
    }

    if save_artifacts:
        print("Saving anchored model artifacts...")
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

    print("\nOOF metrics before calibration:")
    print(
        f"MAPE={raw_oof_metrics['mape']:.2f}% "
        f"MPE={raw_oof_metrics['mpe']:+.2f}% "
        f"RMSE={raw_oof_metrics['rmse']:,.0f} "
        f"R2={raw_oof_metrics['r2']:.4f}"
    )
    print("OOF metrics after calibration:")
    print(
        f"MAPE={calibrated_oof_metrics['mape']:.2f}% "
        f"MPE={calibrated_oof_metrics['mpe']:+.2f}% "
        f"RMSE={calibrated_oof_metrics['rmse']:,.0f} "
        f"R2={calibrated_oof_metrics['r2']:.4f}"
    )

    print("\nTop features:")
    print(feature_importance.head(15).to_string(index=False))

    return {
        "bundle": bundle,
        "cv_metrics": cv_metrics,
        "feature_importance": feature_importance,
        "summary": summary,
        "oof_raw_pred": oof_raw_pred,
        "oof_calibrated_pred": oof_calibrated,
    }


def main() -> None:
    print("Loading training history...")
    history = _load_training_history()
    print(f"Loaded {len(history):,} raw rows")
    result = train_model(history)

    sample_rows = history.head(5).copy()
    print("\nRunning sample predictions on the first 5 rows...")
    sample_pred = result["bundle"].predict(sample_rows)
    print("\nExample prediction flow:")
    print(pd.DataFrame({"predicted_price": sample_pred}).head().to_string(index=False))


if __name__ == "__main__":
    main()
