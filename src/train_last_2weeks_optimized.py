from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

from anchored_price_pipeline import (
    LISTING_ID_COLUMN,
    PRICE_SEGMENT_COLUMN,
    TARGET_PRICE_COLUMN,
    _apply_feature_schema,
    _build_sample_weight,
    _fit_feature_schema,
    _fit_regressor,
    _prepare_rows,
    _regression_metrics,
    apply_kfold_target_encoding,
    calibrate_predictions,
    create_price_anchor,
)
from market_dynamics import DYNAMICS_FEATURES, build_listing_dynamics_targets, build_market_dynamics_features
from model_preprocessing import DATA_DIR, keep_first_snapshot_per_listing, load_and_clean


BASE_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = BASE_DIR / "last_2weeks_optimized_predictions.csv"
COMPARISON_PATH = BASE_DIR / "last_2weeks_optimized_model_comparison.csv"
DETAILED_PREDICTIONS_PATH = BASE_DIR / "last_2weeks_optimized_predictions_detailed.csv"
SEGMENT_ANALYSIS_PATH = BASE_DIR / "last_2weeks_segment_error_analysis.csv"
RANDOM_STATE = 42


def collect_training_files() -> list[str]:
    files = (
        glob.glob(os.path.join(DATA_DIR, "normalized_table.*"))
        + glob.glob(os.path.join(DATA_DIR, "normalized_table_*.*"))
        + glob.glob(os.path.join(DATA_DIR, "Cars24_*.*"))
        + glob.glob(os.path.join(DATA_DIR, "Spinny_*.*"))
    )
    files = [
        path
        for path in files
        if not (path.lower().endswith(".parquet") and os.path.exists(path.rsplit(".", 1)[0] + ".xlsx"))
    ]
    if not files:
        raise RuntimeError(f"No data files found in {DATA_DIR}")
    return files


def drop_extra_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only one price column before sending rows into anchored preprocessing.

    Some source paths carry both the correctly encoded rupee column and a legacy
    misencoded alias. Keeping both causes duplicate standardized column names.
    """
    price_cols = [column for column in df.columns if str(column).startswith("Price")]
    if len(price_cols) <= 1:
        return df

    keep_col = next((column for column in price_cols if "\\u20b9" in ascii(str(column))), price_cols[0])
    drop_cols = [column for column in price_cols if column != keep_col]
    return df.drop(columns=drop_cols, errors="ignore")


def load_last_2weeks_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    files = collect_training_files()
    print(f"Loading {len(files)} files for optimized last-2-weeks training...")
    snapshot_df = pd.concat([load_and_clean(path) for path in files], ignore_index=True)
    snapshot_df = drop_extra_price_columns(snapshot_df)

    fetched_on = pd.to_datetime(snapshot_df["Fetched On"], errors="coerce")
    latest_snapshot = fetched_on.max()
    if pd.isna(latest_snapshot):
        raise RuntimeError("Fetched On is empty after loading; cannot build a last-2-weeks dataset.")

    cutoff = latest_snapshot - pd.Timedelta(days=14)
    recent_df = snapshot_df.loc[fetched_on >= cutoff].copy()
    first_df = keep_first_snapshot_per_listing(recent_df).reset_index(drop=True)
    first_df = drop_extra_price_columns(first_df)
    # `_prepare_rows()` reparses `Year` from text. Passing float values like
    # `2017.0` makes it read `20170`, so normalize to integer-like strings here.
    if "Year" in first_df.columns:
        year_numeric = pd.to_numeric(first_df["Year"], errors="coerce")
        first_df["Year"] = year_numeric.round().astype("Int64").astype("string")
    first_df["_source_row_id"] = np.arange(len(first_df))

    print(f"Raw cleaned history rows: {len(snapshot_df):,}")
    print(f"Rows in last 2 weeks: {len(recent_df):,}")
    print(f"First-occurrence rows in last 2 weeks: {len(first_df):,}")
    return snapshot_df.reset_index(drop=True), first_df


def build_recent_dynamics_target_frame(
    first_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
) -> pd.DataFrame:
    print("Building history-derived dynamics targets...")
    dynamics_targets = build_listing_dynamics_targets(snapshot_df)
    target_df = pd.DataFrame(index=first_df.index, columns=DYNAMICS_FEATURES, dtype=float)

    if "ID" in first_df.columns and not dynamics_targets.empty:
        target_df = first_df[["ID"]].copy()
        target_df["_row_order"] = np.arange(len(target_df))
        target_df["ID"] = target_df["ID"].astype("string").str.strip()
        target_df = (
            target_df.merge(dynamics_targets, on="ID", how="left")
            .sort_values("_row_order", kind="mergesort")
            .drop(columns=["ID", "_row_order"], errors="ignore")
            .reset_index(drop=True)
        )

    known = int(target_df.notna().all(axis=1).sum()) if not target_df.empty else 0
    print(f"Dynamics targets available for {known:,} of {len(first_df):,} recent listings")
    return target_df


def prepare_recent_rows(first_df: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_rows(first_df, allow_missing_target=False)
    if prepared.empty:
        raise RuntimeError("No valid last-2-weeks rows remain after anchored preprocessing.")
    print(f"Prepared leakage-safe training rows: {len(prepared):,}")
    return prepared


def split_recent_rows(prepared: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    price = pd.to_numeric(prepared[TARGET_PRICE_COLUMN], errors="coerce").clip(lower=1.0)
    try:
        strata = pd.qcut(np.log(price), q=5, duplicates="drop").astype(str)
    except ValueError:
        strata = pd.Series(["all"] * len(prepared), index=prepared.index, dtype="object")

    if strata.value_counts(dropna=False).min() < 2:
        strata = pd.Series(["all"] * len(prepared), index=prepared.index, dtype="object")

    idx = np.arange(len(prepared))
    idx_train, idx_temp = train_test_split(
        idx,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=strata if strata.nunique() > 1 else None,
    )
    idx_valid, idx_test = train_test_split(
        idx_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=strata.iloc[idx_temp] if strata.iloc[idx_temp].nunique() > 1 else None,
    )

    print(
        "Split sizes: "
        f"train={len(idx_train):,}, valid={len(idx_valid):,}, test={len(idx_test):,}"
    )
    return idx_train, idx_valid, idx_test


def enrich_with_anchor_and_encoding(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any, Any]:
    train_groups = train_df[LISTING_ID_COLUMN].astype(str)
    train_anchor, anchor_bundle = create_price_anchor(
        train_df,
        groups=train_groups,
        target_col=TARGET_PRICE_COLUMN,
    )
    valid_anchor, _ = create_price_anchor(valid_df, bundle=anchor_bundle)
    test_anchor, _ = create_price_anchor(test_df, bundle=anchor_bundle)

    train_enriched, encoder_bundle = apply_kfold_target_encoding(
        train_anchor,
        target=np.log(train_df[TARGET_PRICE_COLUMN].astype(float).to_numpy()),
        groups=train_groups,
    )
    valid_enriched, _ = apply_kfold_target_encoding(valid_anchor, bundle=encoder_bundle)
    test_enriched, _ = apply_kfold_target_encoding(test_anchor, bundle=encoder_bundle)
    return train_enriched, valid_enriched, test_enriched, anchor_bundle, encoder_bundle


def build_dynamics_feature_frames(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    prepared: pd.DataFrame,
    idx_train: np.ndarray,
    idx_valid: np.ndarray,
    idx_test: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_source_idx = prepared.iloc[idx_train]["_source_row_id"].astype(int).to_numpy()
    print("Generating leakage-safe predicted market-dynamics features...")
    dynamics_all, _ = build_market_dynamics_features(source_df, target_df, train_source_idx)

    def take_rows(indices: np.ndarray) -> pd.DataFrame:
        row_ids = prepared.iloc[indices]["_source_row_id"].astype(int).to_numpy()
        frame = dynamics_all.loc[row_ids, DYNAMICS_FEATURES].reset_index(drop=True)
        return frame.apply(pd.to_numeric, errors="coerce")

    return take_rows(idx_train), take_rows(idx_valid), take_rows(idx_test)


def compose_feature_matrix(
    frame: pd.DataFrame,
    schema: Any,
    *,
    extra_numeric: pd.DataFrame | None = None,
    extra_fill_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    base = _apply_feature_schema(frame, schema)
    numeric_part = base[schema.numeric_columns].copy()
    categorical_part = base[schema.categorical_columns].copy()

    if extra_numeric is None:
        return pd.concat([numeric_part, categorical_part], axis=1)

    extra = extra_numeric.copy().reset_index(drop=True)
    fill_values = extra_fill_values or {}
    for column in extra.columns:
        extra[column] = pd.to_numeric(extra[column], errors="coerce").fillna(float(fill_values.get(column, 0.0)))
    return pd.concat([numeric_part.reset_index(drop=True), extra, categorical_part.reset_index(drop=True)], axis=1)


def fit_catboost_candidate(
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    sample_weight: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid_log: np.ndarray,
    categorical_columns: list[str],
) -> CatBoostRegressor:
    cat_feature_idx = [X_train.columns.get_loc(column) for column in categorical_columns]
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="MAPE",
        depth=8,
        learning_rate=0.03,
        l2_leaf_reg=5.0,
        min_data_in_leaf=50,
        iterations=4000,
        random_seed=RANDOM_STATE,
        verbose=False,
    )
    model.fit(
        X_train,
        y_train_log,
        sample_weight=sample_weight,
        eval_set=(X_valid, y_valid_log),
        use_best_model=True,
        cat_features=cat_feature_idx,
    )
    return model


def evaluate_candidate(
    *,
    name: str,
    model: Any,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    valid_segments: pd.Series,
    test_segments: pd.Series,
) -> dict[str, Any]:
    valid_pred_raw = np.exp(np.asarray(model.predict(X_valid), dtype=float).ravel())
    test_pred_raw = np.exp(np.asarray(model.predict(X_test), dtype=float).ravel())
    calibration_bundle = calibrate_predictions(valid_pred_raw, y_valid, valid_segments)
    valid_pred_cal = calibration_bundle.apply(valid_pred_raw, valid_segments)
    test_pred_cal = calibration_bundle.apply(test_pred_raw, test_segments)

    valid_metrics = _regression_metrics(y_valid, valid_pred_cal)
    test_metrics = _regression_metrics(y_test, test_pred_cal)
    print(
        f"{name}: valid MAPE={valid_metrics['mape']:.2f}% "
        f"test MAPE={test_metrics['mape']:.2f}% "
        f"test MPE={test_metrics['mpe']:+.2f}%"
    )
    return {
        "name": name,
        "model": model,
        "calibration_bundle": calibration_bundle,
        "valid_pred": valid_pred_cal,
        "test_pred": test_pred_cal,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }


def build_blend_candidate(
    name: str,
    valid_preds: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    y_valid: np.ndarray,
    y_test: np.ndarray,
    valid_segments: pd.Series,
    test_segments: pd.Series,
) -> dict[str, Any]:
    weights = {
        key: 1.0 / max(_regression_metrics(y_valid, pred)["mape"], 1e-6)
        for key, pred in valid_preds.items()
    }
    total = sum(weights.values())
    weights = {key: value / total for key, value in weights.items()}

    blended_valid = sum(weights[key] * valid_preds[key] for key in weights)
    blended_test = sum(weights[key] * test_preds[key] for key in weights)
    calibration_bundle = calibrate_predictions(blended_valid, y_valid, valid_segments)
    blended_valid_cal = calibration_bundle.apply(blended_valid, valid_segments)
    blended_test_cal = calibration_bundle.apply(blended_test, test_segments)

    valid_metrics = _regression_metrics(y_valid, blended_valid_cal)
    test_metrics = _regression_metrics(y_test, blended_test_cal)
    print(
        f"{name}: valid MAPE={valid_metrics['mape']:.2f}% "
        f"test MAPE={test_metrics['mape']:.2f}% "
        f"test MPE={test_metrics['mpe']:+.2f}%"
    )
    return {
        "name": name,
        "blend_weights": weights,
        "calibration_bundle": calibration_bundle,
        "valid_pred": blended_valid_cal,
        "test_pred": blended_test_cal,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }


def run_model_search(
    prepared: pd.DataFrame,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_valid: np.ndarray,
    idx_test: np.ndarray,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    train_df = prepared.iloc[idx_train].reset_index(drop=True)
    valid_df = prepared.iloc[idx_valid].reset_index(drop=True)
    test_df = prepared.iloc[idx_test].reset_index(drop=True)

    print("Building price anchors and target encodings...")
    train_enriched, valid_enriched, test_enriched, _, _ = enrich_with_anchor_and_encoding(
        train_df,
        valid_df,
        test_df,
    )

    y_train = train_df[TARGET_PRICE_COLUMN].astype(float).to_numpy()
    y_valid = valid_df[TARGET_PRICE_COLUMN].astype(float).to_numpy()
    y_test = test_df[TARGET_PRICE_COLUMN].astype(float).to_numpy()
    y_train_log = np.log(np.clip(y_train, 1.0, None))
    y_valid_log = np.log(np.clip(y_valid, 1.0, None))
    sample_weight = _build_sample_weight(y_train)

    print("Fitting base feature schema...")
    base_schema = _fit_feature_schema(train_enriched)
    X_train_base = compose_feature_matrix(train_enriched, base_schema)
    X_valid_base = compose_feature_matrix(valid_enriched, base_schema)
    X_test_base = compose_feature_matrix(test_enriched, base_schema)

    print("Preparing optional dynamics feature frames...")
    dyn_train, dyn_valid, dyn_test = build_dynamics_feature_frames(
        source_df,
        target_df,
        prepared,
        idx_train,
        idx_valid,
        idx_test,
    )
    dyn_fill_values = {
        column: float(pd.to_numeric(dyn_train[column], errors="coerce").median())
        if dyn_train[column].notna().any()
        else 0.0
        for column in dyn_train.columns
    }
    X_train_dyn = compose_feature_matrix(
        train_enriched,
        base_schema,
        extra_numeric=dyn_train,
        extra_fill_values=dyn_fill_values,
    )
    X_valid_dyn = compose_feature_matrix(
        valid_enriched,
        base_schema,
        extra_numeric=dyn_valid,
        extra_fill_values=dyn_fill_values,
    )
    X_test_dyn = compose_feature_matrix(
        test_enriched,
        base_schema,
        extra_numeric=dyn_test,
        extra_fill_values=dyn_fill_values,
    )

    valid_segments = valid_enriched[PRICE_SEGMENT_COLUMN].astype(str)
    test_segments = test_enriched[PRICE_SEGMENT_COLUMN].astype(str)

    print("Training candidate models...")
    candidates: list[dict[str, Any]] = []
    raw_valid_preds: dict[str, np.ndarray] = {}
    raw_test_preds: dict[str, np.ndarray] = {}

    lgb_base, _ = _fit_regressor(
        X_train_base,
        y_train_log,
        sample_weight,
        X_valid=X_valid_base,
        y_valid_log=y_valid_log,
        n_estimators=3000,
    )
    result = evaluate_candidate(
        name="lightgbm_anchor",
        model=lgb_base,
        X_valid=X_valid_base,
        X_test=X_test_base,
        y_valid=y_valid,
        y_test=y_test,
        valid_segments=valid_segments,
        test_segments=test_segments,
    )
    candidates.append(result)
    raw_valid_preds["lightgbm_anchor"] = np.exp(np.asarray(lgb_base.predict(X_valid_base), dtype=float).ravel())
    raw_test_preds["lightgbm_anchor"] = np.exp(np.asarray(lgb_base.predict(X_test_base), dtype=float).ravel())

    cat_base = fit_catboost_candidate(
        X_train_base,
        y_train_log,
        sample_weight,
        X_valid_base,
        y_valid_log,
        list(base_schema.categorical_columns),
    )
    result = evaluate_candidate(
        name="catboost_anchor",
        model=cat_base,
        X_valid=X_valid_base,
        X_test=X_test_base,
        y_valid=y_valid,
        y_test=y_test,
        valid_segments=valid_segments,
        test_segments=test_segments,
    )
    candidates.append(result)
    raw_valid_preds["catboost_anchor"] = np.exp(np.asarray(cat_base.predict(X_valid_base), dtype=float).ravel())
    raw_test_preds["catboost_anchor"] = np.exp(np.asarray(cat_base.predict(X_test_base), dtype=float).ravel())

    lgb_dyn, _ = _fit_regressor(
        X_train_dyn,
        y_train_log,
        sample_weight,
        X_valid=X_valid_dyn,
        y_valid_log=y_valid_log,
        n_estimators=3000,
    )
    result = evaluate_candidate(
        name="lightgbm_anchor_dynamics",
        model=lgb_dyn,
        X_valid=X_valid_dyn,
        X_test=X_test_dyn,
        y_valid=y_valid,
        y_test=y_test,
        valid_segments=valid_segments,
        test_segments=test_segments,
    )
    candidates.append(result)
    raw_valid_preds["lightgbm_anchor_dynamics"] = np.exp(np.asarray(lgb_dyn.predict(X_valid_dyn), dtype=float).ravel())
    raw_test_preds["lightgbm_anchor_dynamics"] = np.exp(np.asarray(lgb_dyn.predict(X_test_dyn), dtype=float).ravel())

    cat_dyn = fit_catboost_candidate(
        X_train_dyn,
        y_train_log,
        sample_weight,
        X_valid_dyn,
        y_valid_log,
        list(base_schema.categorical_columns),
    )
    result = evaluate_candidate(
        name="catboost_anchor_dynamics",
        model=cat_dyn,
        X_valid=X_valid_dyn,
        X_test=X_test_dyn,
        y_valid=y_valid,
        y_test=y_test,
        valid_segments=valid_segments,
        test_segments=test_segments,
    )
    candidates.append(result)
    raw_valid_preds["catboost_anchor_dynamics"] = np.exp(np.asarray(cat_dyn.predict(X_valid_dyn), dtype=float).ravel())
    raw_test_preds["catboost_anchor_dynamics"] = np.exp(np.asarray(cat_dyn.predict(X_test_dyn), dtype=float).ravel())

    blend_base = build_blend_candidate(
        "blend_anchor",
        {
            "lightgbm_anchor": raw_valid_preds["lightgbm_anchor"],
            "catboost_anchor": raw_valid_preds["catboost_anchor"],
        },
        {
            "lightgbm_anchor": raw_test_preds["lightgbm_anchor"],
            "catboost_anchor": raw_test_preds["catboost_anchor"],
        },
        y_valid,
        y_test,
        valid_segments,
        test_segments,
    )
    candidates.append(blend_base)

    blend_dyn = build_blend_candidate(
        "blend_anchor_dynamics",
        {
            "lightgbm_anchor_dynamics": raw_valid_preds["lightgbm_anchor_dynamics"],
            "catboost_anchor_dynamics": raw_valid_preds["catboost_anchor_dynamics"],
        },
        {
            "lightgbm_anchor_dynamics": raw_test_preds["lightgbm_anchor_dynamics"],
            "catboost_anchor_dynamics": raw_test_preds["catboost_anchor_dynamics"],
        },
        y_valid,
        y_test,
        valid_segments,
        test_segments,
    )
    candidates.append(blend_dyn)

    comparison_rows = []
    for candidate in candidates:
        comparison_rows.append(
            {
                "candidate": candidate["name"],
                "valid_mape": candidate["valid_metrics"]["mape"],
                "valid_mpe": candidate["valid_metrics"]["mpe"],
                "valid_rmse": candidate["valid_metrics"]["rmse"],
                "test_mape": candidate["test_metrics"]["mape"],
                "test_mpe": candidate["test_metrics"]["mpe"],
                "test_rmse": candidate["test_metrics"]["rmse"],
                "test_r2": candidate["test_metrics"]["r2"],
                "test_mae": candidate["test_metrics"]["mae"],
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values(["test_mape", "valid_mape"], kind="mergesort")

    test_predictions = pd.DataFrame(
        {
            "listing_id": test_df[LISTING_ID_COLUMN].astype(str).to_numpy(),
            "actual_price": y_test,
        }
    )
    for candidate in candidates:
        test_predictions[candidate["name"]] = np.asarray(candidate["test_pred"], dtype=float)

    return candidates, comparison, test_predictions


def summarize_segment_errors(
    detailed_predictions: pd.DataFrame,
    group_col: str,
    *,
    top_n: int = 20,
    min_rows: int = 20,
) -> pd.DataFrame:
    rows = []
    for key, group in detailed_predictions.groupby(group_col, dropna=False, sort=False):
        if len(group) < min_rows:
            continue
        metrics = _regression_metrics(
            group[TARGET_PRICE_COLUMN].to_numpy(),
            group["predicted_price"].to_numpy(),
        )
        rows.append(
            {
                "segment_type": group_col,
                "segment_value": str(key),
                "rows": len(group),
                "mape": metrics["mape"],
                "mpe": metrics["mpe"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "median_actual": float(group[TARGET_PRICE_COLUMN].median()),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(["mape", "rows"], ascending=[False, False]).head(top_n)


def save_diagnostics(
    prepared: pd.DataFrame,
    idx_test: np.ndarray,
    best_name: str,
    test_predictions: pd.DataFrame,
) -> None:
    test_df = prepared.iloc[idx_test].reset_index(drop=True).copy()
    detailed = pd.DataFrame(
        {
            LISTING_ID_COLUMN: test_df[LISTING_ID_COLUMN].astype(str).to_numpy(),
            "Make": test_df["Make"].astype(str).to_numpy(),
            "Model": test_df["Model"].astype(str).to_numpy(),
            "Variant": test_df["Variant"].astype(str).to_numpy(),
            "City": test_df["City"].astype(str).to_numpy(),
            "Fuel": test_df["Fuel"].astype(str).to_numpy(),
            "BodyType": test_df["BodyType"].astype(str).to_numpy(),
            TARGET_PRICE_COLUMN: pd.to_numeric(test_df[TARGET_PRICE_COLUMN], errors="coerce").to_numpy(),
            "predicted_price": pd.to_numeric(test_predictions[best_name], errors="coerce").to_numpy(),
        }
    )
    detailed["ape_pct"] = (
        (detailed["predicted_price"] - detailed[TARGET_PRICE_COLUMN]).abs()
        / detailed[TARGET_PRICE_COLUMN].clip(lower=1.0)
        * 100.0
    )
    detailed["pe_pct"] = (
        (detailed["predicted_price"] - detailed[TARGET_PRICE_COLUMN])
        / detailed[TARGET_PRICE_COLUMN].clip(lower=1.0)
        * 100.0
    )
    detailed["actual_price_band"] = pd.cut(
        detailed[TARGET_PRICE_COLUMN],
        bins=[-np.inf, 500_000, 1_000_000, 2_000_000, np.inf],
        labels=["<=5L", "5-10L", "10-20L", ">20L"],
        include_lowest=True,
    ).astype(str)

    segment_frames = [
        summarize_segment_errors(detailed, "actual_price_band", top_n=10, min_rows=20),
        summarize_segment_errors(detailed, "Make", top_n=15, min_rows=30),
        summarize_segment_errors(detailed, "Model", top_n=20, min_rows=30),
        summarize_segment_errors(detailed, "City", top_n=15, min_rows=30),
        summarize_segment_errors(detailed, "Fuel", top_n=10, min_rows=20),
        summarize_segment_errors(detailed, "BodyType", top_n=10, min_rows=20),
    ]
    segment_analysis = pd.concat(segment_frames, ignore_index=True)

    detailed.to_csv(DETAILED_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    segment_analysis.to_csv(SEGMENT_ANALYSIS_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved detailed predictions to: {DETAILED_PREDICTIONS_PATH}")
    print(f"Saved segment error analysis to: {SEGMENT_ANALYSIS_PATH}")


def main() -> None:
    snapshot_df, first_df = load_last_2weeks_history()
    target_df = build_recent_dynamics_target_frame(first_df, snapshot_df)
    prepared = prepare_recent_rows(first_df)
    idx_train, idx_valid, idx_test = split_recent_rows(prepared)

    candidates, comparison, test_predictions = run_model_search(
        prepared,
        first_df,
        target_df,
        idx_train,
        idx_valid,
        idx_test,
    )

    best_name = comparison.iloc[0]["candidate"]
    best_row = comparison.iloc[0]
    best_predictions = test_predictions[["listing_id", "actual_price", best_name]].copy()
    best_predictions = best_predictions.rename(columns={best_name: "predicted_price"})

    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")
    best_predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    save_diagnostics(prepared, idx_test, best_name, test_predictions)

    print("\nBest candidate:")
    print(
        f"{best_name} -> test MAPE={best_row['test_mape']:.2f}% "
        f"MPE={best_row['test_mpe']:+.2f}% "
        f"RMSE={best_row['test_rmse']:,.0f}"
    )
    print(f"Saved model comparison to: {COMPARISON_PATH}")
    print(f"Saved best predictions to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
