from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model_preprocessing import DATA_DIR, _parse_kms, _parse_ownership, _split_name_to_make_model, standardize_columns
from train_last_2weeks_baseline import BASE_DIR, CANONICAL_COLUMNS, clean_features, evaluate_model, get_first_occurrence
from train_last_2weeks_with_market_dynamics import (
    DYNAMICS_FEATURES,
    apply_feature_schema,
    build_dynamic_feature_predictions,
    build_listing_dynamics_targets,
    fit_feature_schema,
    make_main_model,
)


PREDICTIONS_PATH = Path(BASE_DIR) / "normalized_table_part4_baseline_vs_dynamics_predictions.csv"
PART4_FILE_STEMS = [
    "normalized_table_part4.xlsx",
    "normalized_table_part4.parquet",
    "normalized_table_part4.csv",
]
RANDOM_STATE = 42


def load_part4_data() -> pd.DataFrame:
    """Load only normalized_table_part4 and map it to the same canonical schema as the baseline script."""
    selected_path = None
    for filename in PART4_FILE_STEMS:
        candidate = os.path.join(DATA_DIR, filename)
        if os.path.exists(candidate):
            selected_path = candidate
            break
    if selected_path is None:
        raise RuntimeError("normalized_table_part4 source file not found in data directory.")

    if selected_path.lower().endswith((".xlsx", ".xls")):
        parquet_path = selected_path.rsplit(".", 1)[0] + ".parquet"
        if os.path.exists(parquet_path) and os.path.getmtime(parquet_path) >= os.path.getmtime(selected_path):
            df = pd.read_parquet(parquet_path)
        else:
            df = pd.read_excel(selected_path)
            df.to_parquet(parquet_path, index=False)
    elif selected_path.lower().endswith(".parquet"):
        df = pd.read_parquet(selected_path)
    else:
        df = pd.read_csv(selected_path)

    df = standardize_columns(df)
    rename_map = {column: CANONICAL_COLUMNS[column] for column in df.columns if column in CANONICAL_COLUMNS}
    df = df.rename(columns=rename_map)

    if "name" in df.columns and ("make" not in df.columns or "model" not in df.columns):
        makes, models = _split_name_to_make_model(df["name"])
        if "make" not in df.columns:
            df["make"] = makes
        if "model" not in df.columns:
            df["model"] = models

    required_defaults = {
        "car_id": np.nan,
        "date_posted": pd.NaT,
        "price": np.nan,
        "make": "UNKNOWN",
        "model": "UNKNOWN",
        "variant": "UNKNOWN",
        "kms_driven": np.nan,
        "ownership": np.nan,
        "city": "UNKNOWN",
        "car_age": np.nan,
        "year": np.nan,
    }
    for column, default in required_defaults.items():
        if column not in df.columns:
            df[column] = default

    df["car_id"] = df["car_id"].astype("string").str.strip()
    df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")
    df["price"] = (
        df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True).replace({"": np.nan}).astype(float)
    )
    df["kms_driven"] = df["kms_driven"].apply(_parse_kms)
    df["ownership"] = pd.to_numeric(df["ownership"].apply(_parse_ownership), errors="coerce")
    df["year"] = (
        df["year"].astype(str).str.replace(r"[^\d]", "", regex=True).replace({"": np.nan}).astype(float)
    )

    for column in ["make", "model", "variant", "city"]:
        df[column] = df[column].fillna("UNKNOWN").astype(str).str.strip().str.upper()

    if df["car_age"].isna().all() and df["year"].notna().any():
        snapshot_year = df["date_posted"].dt.year.fillna(float(pd.Timestamp.now().year))
        df["car_age"] = (snapshot_year - df["year"]).clip(lower=0)
    df["car_age"] = pd.to_numeric(df["car_age"], errors="coerce")
    return df


def train_part4_models(df: pd.DataFrame, dynamics_targets: pd.DataFrame) -> dict[str, object]:
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

    y_train = np.log(train_df["price"].astype(float).to_numpy())
    y_valid = np.log(valid_df["price"].astype(float).to_numpy())
    actual_price = np.exp(y_valid)

    print("Training baseline model on normalized_table_part4...")
    baseline_model = make_main_model()
    baseline_model.fit(
        X_train_base,
        y_train,
        categorical_feature=schema.categorical_columns,
    )
    baseline_pred = np.exp(np.asarray(baseline_model.predict(X_valid_base), dtype=float).ravel())
    print("Baseline metrics:")
    baseline_metrics = evaluate_model(actual_price, baseline_pred)

    print("Training dynamics-augmented model on normalized_table_part4...")
    dynamics_train, dynamics_valid = build_dynamic_feature_predictions(
        X_train_base,
        X_valid_base,
        train_df[DYNAMICS_FEATURES],
    )

    X_train_dyn = X_train_base.copy()
    X_valid_dyn = X_valid_base.copy()
    for column in DYNAMICS_FEATURES:
        X_train_dyn[column] = dynamics_train[column].astype(float)
        X_valid_dyn[column] = dynamics_valid[column].astype(float)

    dynamics_model = make_main_model()
    dynamics_model.fit(
        X_train_dyn,
        y_train,
        categorical_feature=schema.categorical_columns,
    )
    dynamics_pred = np.exp(np.asarray(dynamics_model.predict(X_valid_dyn), dtype=float).ravel())
    print("Dynamics metrics:")
    dynamics_metrics = evaluate_model(actual_price, dynamics_pred)

    predictions = pd.DataFrame(
        {
            "car_id": valid_df["car_id"].astype(str).to_numpy(),
            "actual_price": actual_price,
            "baseline_predicted_price": baseline_pred,
            "dynamics_predicted_price": dynamics_pred,
        }
    )
    for column in DYNAMICS_FEATURES:
        predictions[f"Pred {column}"] = dynamics_valid[column].to_numpy()
    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved predictions to: {PREDICTIONS_PATH}")

    return {
        "baseline_model": baseline_model,
        "dynamics_model": dynamics_model,
        "baseline_metrics": baseline_metrics,
        "dynamics_metrics": dynamics_metrics,
        "predictions": predictions,
    }


def main() -> None:
    print("Loading normalized_table_part4...")
    raw_df = load_part4_data()
    print(f"Loaded {len(raw_df):,} raw rows from normalized_table_part4")

    print("Building listing-level dynamics targets from normalized_table_part4 history...")
    dynamics_targets = build_listing_dynamics_targets(raw_df)
    print(f"Dynamics targets built for {len(dynamics_targets):,} unique cars")

    print("Keeping first occurrence per car...")
    first_df = get_first_occurrence(raw_df)
    print(f"Rows after first-occurrence dedupe: {len(first_df):,}")

    print("Cleaning static features...")
    clean_df = clean_features(first_df)
    print(f"Rows after cleaning: {len(clean_df):,}")

    train_part4_models(clean_df, dynamics_targets)


if __name__ == "__main__":
    main()
