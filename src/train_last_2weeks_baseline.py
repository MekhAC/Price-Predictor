from __future__ import annotations

import glob
import os
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from model_preprocessing import DATA_DIR, _parse_kms, _parse_ownership, _split_name_to_make_model, standardize_columns


BASE_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = BASE_DIR / "last_2weeks_predictions.csv"
RANDOM_STATE = 42
TOP_N_VARIANTS = 20

CANONICAL_COLUMNS = {
    "ID": "car_id",
    "id": "car_id",
    "car_id": "car_id",
    "Name": "name",
    "name": "name",
    "Fetched On": "date_posted",
    "date": "date_posted",
    "date_posted": "date_posted",
    "Price (₹)": "price",
    "Price (â‚¹)": "price",
    "price": "price",
    "Make": "make",
    "make": "make",
    "Model": "model",
    "model": "model",
    "Variant": "variant",
    "variant": "variant",
    "KMs Driven": "kms_driven",
    "kms_driven": "kms_driven",
    "Ownership": "ownership",
    "ownership": "ownership",
    "City": "city",
    "city": "city",
    "Car Age": "car_age",
    "car_age": "car_age",
    "Year": "year",
    "year": "year",
}

DYNAMIC_COLUMNS = [
    "days_since_listing",
    "price_change_count",
    "expected_avg_drop_7d",
    "expected_avg_drop_30d",
    "expected_time_to_first_drop",
    "expected_price_volatility",
    "expected_time_to_sell",
    "expected_price_drop_rate",
    "expected_market_liquidity_score",
    "market_days",
    "days_on_market",
    "fetch_month",
]

TIME_LEAKAGE_PATTERNS = (
    "days_since",
    "time_to_",
    "avg_drop",
    "price_change",
    "volatility",
    "market_day",
    "days_on_market",
    "fetch_month",
)


def load_data() -> pd.DataFrame:
    """Load all raw tables from the local data directory and map them to a clean baseline schema."""
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

    frames = []
    for path in files:
        if path.lower().endswith((".xlsx", ".xls")):
            parquet_path = path.rsplit(".", 1)[0] + ".parquet"
            if os.path.exists(parquet_path) and os.path.getmtime(parquet_path) >= os.path.getmtime(path):
                frame = pd.read_parquet(parquet_path)
            else:
                frame = pd.read_excel(path)
                frame.to_parquet(parquet_path, index=False)
        elif path.lower().endswith(".parquet"):
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
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
        df["price"]
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .replace({"": np.nan})
        .astype(float)
    )
    df["kms_driven"] = df["kms_driven"].apply(_parse_kms)
    df["ownership"] = pd.to_numeric(df["ownership"].apply(_parse_ownership), errors="coerce")
    df["year"] = (
        df["year"]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .replace({"": np.nan})
        .astype(float)
    )
    df["make"] = df["make"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    df["model"] = df["model"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    df["variant"] = df["variant"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    df["city"] = df["city"].fillna("UNKNOWN").astype(str).str.strip().str.upper()

    if df["car_age"].isna().all() and df["year"].notna().any():
        snapshot_year = df["date_posted"].dt.year.fillna(float(pd.Timestamp.now().year))
        df["car_age"] = (snapshot_year - df["year"]).clip(lower=0)

    df["car_age"] = pd.to_numeric(df["car_age"], errors="coerce")
    return df


def filter_last_2weeks(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict training to the most recent 14 days only."""
    if df["date_posted"].notna().sum() == 0:
        raise RuntimeError("date_posted is empty after parsing; cannot filter the last 2 weeks.")
    cutoff_date = df["date_posted"].max() - pd.Timedelta(days=14)
    return df.loc[df["date_posted"] >= cutoff_date].copy()


def get_first_occurrence(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the earliest row per car_id inside the last-2-weeks window."""
    ordered = df.sort_values(["car_id", "date_posted"], kind="mergesort")
    return ordered.groupby("car_id", as_index=False, sort=False).head(1).reset_index(drop=True)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop dynamic features, keep static columns, group rare variants, and prepare a simple baseline frame."""
    work = df.copy()

    dynamic_cols = [column for column in work.columns if column in DYNAMIC_COLUMNS]
    dynamic_cols += [
        column
        for column in work.columns
        if any(pattern in str(column).strip().lower() for pattern in TIME_LEAKAGE_PATTERNS)
    ]
    work = work.drop(columns=sorted(set(dynamic_cols)), errors="ignore")

    if "car_age" in work.columns and "year" in work.columns:
        work = work.drop(columns=["year"], errors="ignore")

    work = work[work["price"].notna() & (work["price"] > 0)].copy()
    work = work[work["car_id"].notna() & work["date_posted"].notna()].copy()

    keep_columns = ["car_id", "date_posted", "price", "make", "model", "variant", "kms_driven", "ownership", "city", "car_age"]
    keep_columns = [column for column in keep_columns if column in work.columns]
    work = work[keep_columns].copy()

    top_variants = work["variant"].value_counts(dropna=True).head(TOP_N_VARIANTS).index
    work["variant"] = work["variant"].where(work["variant"].isin(top_variants), "OTHER")

    numeric_columns = [column for column in ["kms_driven", "ownership", "car_age"] if column in work.columns]
    categorical_columns = [column for column in ["make", "model", "variant", "city"] if column in work.columns]

    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
        work[column] = work[column].fillna(work[column].median())

    for column in categorical_columns:
        work[column] = work[column].fillna("UNKNOWN").astype(str).str.strip().replace({"": "UNKNOWN"})

    return work.reset_index(drop=True)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics in original price space."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    if not valid.any():
        raise RuntimeError("No valid rows available for evaluation.")

    actual = y_true[valid]
    pred = y_pred[valid]
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    ape = np.abs(pred - actual) / np.clip(actual, 1.0, None)
    pe = (pred - actual) / np.clip(actual, 1.0, None)

    metrics = {
        "mape": float(np.mean(ape) * 100.0),
        "mae": float(mae),
        "rmse": float(sqrt(mse)),
        "mpe": float(np.mean(pe) * 100.0),
    }
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"MAE : {metrics['mae']:,.0f}")
    print(f"RMSE: {metrics['rmse']:,.0f}")
    print(f"MPE : {metrics['mpe']:+.2f}%")
    return metrics


def train_model(df: pd.DataFrame) -> dict[str, object]:
    """Train a simple LightGBM baseline on last-2-weeks first-occurrence rows only."""
    feature_columns = ["make", "model", "variant", "kms_driven", "ownership", "city", "car_age"]
    feature_columns = [column for column in feature_columns if column in df.columns]

    X = df[feature_columns].copy()
    y = np.log(df["price"].astype(float).to_numpy())
    car_ids = df["car_id"].astype(str).to_numpy()

    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X,
        y,
        car_ids,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    numeric_columns = [column for column in ["kms_driven", "ownership", "car_age"] if column in X_train.columns]
    categorical_columns = [column for column in ["make", "model", "variant", "city"] if column in X_train.columns]

    numeric_fill_values = {column: float(pd.to_numeric(X_train[column], errors="coerce").median()) for column in numeric_columns}
    category_levels: dict[str, list[str]] = {}
    for column in categorical_columns:
        values = X_train[column].fillna("UNKNOWN").astype(str).str.strip().replace({"": "UNKNOWN"})
        levels = sorted(values.unique().tolist())
        if "UNKNOWN" not in levels:
            levels.append("UNKNOWN")
        if column == "variant" and "OTHER" not in levels:
            levels.append("OTHER")
        category_levels[column] = levels

    for frame in (X_train, X_val):
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(numeric_fill_values[column]).astype(float)
        for column in categorical_columns:
            values = frame[column].fillna("UNKNOWN").astype(str).str.strip().replace({"": "UNKNOWN"})
            values = values.where(values.isin(category_levels[column]), "OTHER" if column == "variant" else "UNKNOWN")
            frame[column] = pd.Categorical(values, categories=category_levels[column])

    model = LGBMRegressor(
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
    model.fit(X_train, y_train, categorical_feature=categorical_columns)

    pred_log = model.predict(X_val)
    pred_price = np.exp(np.asarray(pred_log, dtype=float).ravel())
    actual_price = np.exp(np.asarray(y_val, dtype=float).ravel())

    metrics = evaluate_model(actual_price, pred_price)

    predictions = pd.DataFrame(
        {
            "car_id": id_val,
            "actual_price": actual_price,
            "predicted_price": pred_price,
        }
    )
    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved predictions to: {PREDICTIONS_PATH}")

    return {
        "model": model,
        "metrics": metrics,
        "predictions": predictions,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
    }


def main() -> None:
    print("Loading data...")
    raw_df = load_data()
    print(f"Loaded {len(raw_df):,} raw rows")

    print("Filtering last 2 weeks...")
    recent_df = filter_last_2weeks(raw_df)
    print(f"Rows after last-2-weeks filter: {len(recent_df):,}")

    print("Keeping first occurrence per car...")
    first_df = get_first_occurrence(recent_df)
    print(f"Rows after first-occurrence dedupe: {len(first_df):,}")

    print("Cleaning static baseline features...")
    baseline_df = clean_features(first_df)
    print(f"Rows after cleaning: {len(baseline_df):,}")

    print("Training baseline model...")
    train_model(baseline_df)


if __name__ == "__main__":
    main()
