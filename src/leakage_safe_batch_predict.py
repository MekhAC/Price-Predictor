from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from leakage_safe_price_pipeline import load_bundle, predict_with_metrics, summarize_prediction_row_flow
from model_preprocessing import standardize_columns


def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _write_any(df: pd.DataFrame, path: str) -> None:
    if path.lower().endswith(('.xlsx', '.xls')):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False, encoding='utf-8-sig')


def _default_output_path(input_path: str) -> str:
    root, ext = os.path.splitext(input_path)
    if ext.lower() in ('.xlsx', '.xls', '.csv'):
        return f"{root}_predictions{ext}"
    return f"{input_path}_predictions.csv"


def _ensure_kms_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'KMs Driven' in df.columns:
        return df
    aliases = ['kms', 'kmsdriven', 'km driven', 'kms driven', 'odometer']
    normalized = {str(c).strip().lower().replace('_', ' ').replace('-', ' '): c for c in df.columns}
    for alias in aliases:
        if alias in normalized:
            df['KMs Driven'] = df[normalized[alias]]
            return df
    raise KeyError("Could not find KMs Driven column (or known alias).")


def _resolve_actual_price_column(df: pd.DataFrame) -> str | None:
    for column in ['Target Price', 'Price (₹)', 'Price (â‚¹)', 'Price (Ã¢â€šÂ¹)']:
        if column in df.columns:
            return column
    return None


def _print_metrics(label: str, actual: pd.Series, pred: pd.Series) -> None:
    valid = actual.notna() & pred.notna() & (actual != 0)
    if not valid.any():
        print(f"{label}: no valid rows for metric computation")
        return

    a = actual[valid]
    p = pred[valid]
    mse = mean_squared_error(a, p)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(a, p)
    median_ae = median_absolute_error(a, p)
    r2 = r2_score(a, p)
    mape = (((p - a).abs() / a).mean()) * 100
    mpe = (((p - a) / a).mean()) * 100

    print(f"\n--- {label} ---")
    print(f"Rows : {int(valid.sum())}")
    print(f"MSE  : {mse:,.0f}")
    print(f"RMSE : {rmse:,.0f}")
    print(f"MAE  : {mae:,.0f}")
    print(f"Median AE : {median_ae:,.0f}")
    print(f"RÂ²   : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"MPE  : {mpe:+.2f}%")


def _print_segment_error_summary(df: pd.DataFrame, pred_col: str, label: str) -> None:
    actual_col = _resolve_actual_price_column(df)
    if actual_col is None:
        print(f"\n--- Segment Error Summary ({label}) ---")
        print("No actual price column available")
        return

    actual = pd.to_numeric(df[actual_col], errors='coerce')
    pred = pd.to_numeric(df[pred_col], errors='coerce')
    valid = actual.notna() & pred.notna() & (actual != 0)
    if not valid.any():
        print(f"\n--- Segment Error Summary ({label}) ---")
        print("No valid rows")
        return

    required_cols = ['Make', 'Year', 'City', 'KMs Driven', 'Ownership', 'Transmission', 'Fuel', 'BodyType']
    cols_present = [c for c in required_cols if c in df.columns]
    work = df.loc[valid, cols_present].copy()
    for column in required_cols:
        if column not in work.columns:
            work[column] = np.nan

    work['actual'] = actual[valid]
    work['pred'] = pred[valid]
    work['ape'] = ((work['pred'] - work['actual']).abs() / work['actual']) * 100
    work['pe'] = ((work['pred'] - work['actual']) / work['actual']) * 100

    work['Make'] = work['Make'].fillna('Unknown').astype(str).str.strip()
    work['City'] = work['City'].fillna('Unknown').astype(str).str.strip()
    work['Transmission'] = work['Transmission'].fillna('Unknown').astype(str).str.strip()
    work['Fuel'] = work['Fuel'].fillna('Unknown').astype(str).str.strip()
    work['BodyType'] = work['BodyType'].fillna('Unknown').astype(str).str.strip()
    work['Year'] = pd.to_numeric(work['Year'], errors='coerce')
    work['KMs Driven'] = pd.to_numeric(work['KMs Driven'], errors='coerce')
    work['Ownership'] = pd.to_numeric(work['Ownership'], errors='coerce')

    def _print_group_table(title: str, grouped: pd.DataFrame, index_name: str = 'Segment') -> None:
        print(f"\n--- {title} ---")
        print(f"{index_name:>15} | {'MAPE':>8} | {'MPE':>8} | Bias")
        print("-" * 56)
        for seg, row in grouped.iterrows():
            mpe = row['mpe']
            direction = 'POSITIVE(over)' if mpe > 0 else ('NEGATIVE(under)' if mpe < 0 else 'NEUTRAL')
            print(f"{str(seg):>15} | {row['mape']:>7.2f}% | {mpe:>+7.2f}% | {direction}")

    bins = [-np.inf, 300000, 700000, 1200000, 2000000, np.inf]
    names = ['<=3L', '3-7L', '7-12L', '12-20L', '>20L']
    work['price_band'] = pd.cut(work['actual'], bins=bins, labels=names)

    band_stats = work.groupby('price_band', observed=True).agg(mape=('ape', 'mean'), mpe=('pe', 'mean')).sort_index()
    _print_group_table(f"Segment Error Summary ({label}) by Price Band", band_stats)

    top_makes = work['Make'].value_counts().head(10).index
    make_stats = (
        work[work['Make'].isin(top_makes)]
        .groupby('Make')
        .agg(mape=('ape', 'mean'), mpe=('pe', 'mean'))
        .sort_values(by='mape')
    )
    _print_group_table(f"Segment Error Summary ({label}) by Top Makes", make_stats, index_name='Make')


def _print_row_flow(summary: dict[str, int]) -> None:
    print("\n--- Leakage-Safe Row Flow ---")
    print(f"Input Rows : {summary['input_rows']}")
    print(f"Rows After Cleaning : {summary['cleaned_rows']}")
    print(f"Dropped In Cleaning : {summary['dropped_in_cleaning']}")
    print(f"Prediction Rows : {summary['prediction_rows']}")
    print(f"Rows If Anchor/Dedup Applied : {summary['rows_if_anchor_deduped']}")
    print(f"Would Drop In Anchor/Dedup : {summary['would_drop_in_anchor_dedup']}")
    print(f"Metric Rows : {summary['metric_rows']}")


def predict_batch(input_path: str, output_path: Optional[str] = None) -> None:
    bundle = load_bundle()
    df = standardize_columns(_read_any(input_path).copy())
    df = _ensure_kms_column(df)
    _print_row_flow(summarize_prediction_row_flow(df))

    output_df, metrics = predict_with_metrics(df, bundle=bundle)

    actual_col = _resolve_actual_price_column(output_df)
    if actual_col is not None:
        actual = pd.to_numeric(output_df[actual_col], errors='coerce')
        pred = pd.to_numeric(output_df['Predicted Price'], errors='coerce')
        _print_metrics('Leakage-Safe Prediction', actual, pred)
        _print_segment_error_summary(output_df, 'Predicted Price', 'Leakage-Safe')

    if metrics is not None:
        print("\n--- Leakage-Safe Summary ---")
        print(f"Rows : {metrics['rows']}")
        print(f"MSE  : {metrics['mse']:,.0f}")
        print(f"RMSE : {metrics['rmse']:,.0f}")
        print(f"MAE  : {metrics['mae']:,.0f}")
        print(f"Median AE : {metrics['median_ae']:,.0f}")
        print(f"RÂ²   : {metrics['r2']:.4f}")
        print(f"MAPE : {metrics['mape']:.2f}%")
        print(f"MPE  : {metrics['mpe']:+.2f}%")

    out = output_path or _default_output_path(input_path)
    _write_any(output_df, out)
    print(f"Saved predictions to: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Batch price prediction using leakage-safe model artifacts.')
    parser.add_argument('--input', required=True, help='Path to input .xlsx/.xls/.csv file')
    parser.add_argument('--output', default='', help='Optional output path (.xlsx/.xls/.csv)')
    args = parser.parse_args()
    predict_batch(args.input, args.output or None)


if __name__ == '__main__':
    main()
