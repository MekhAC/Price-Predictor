# primary runtime batch entrypoint
import os
import argparse
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.metrics import mean_squared_error, r2_score

from primary_model_runtime import load_primary_artifacts, predict_primary_rows

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


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
    for a in aliases:
        if a in normalized:
            df['KMs Driven'] = df[normalized[a]]
            return df
    raise KeyError("Could not find KMs Driven column (or known alias).")


def _reg_state_from_registration(x) -> str:
    if pd.isna(x):
        return 'UNK'
    s = str(x).strip().upper()
    return s[:2] if len(s) >= 2 and s[:2].isalpha() else 'UNK'


def _parse_ownership(x) -> float:
    if pd.isna(x):
        return 1.0
    s = str(x).strip().lower()
    mapping = {
        'first': 1.0, '1st': 1.0, 'one': 1.0,
        'second': 2.0, '2nd': 2.0, 'two': 2.0,
        'third': 3.0, '3rd': 3.0, 'three': 3.0,
        'fourth': 4.0, '4th': 4.0, 'four': 4.0,
    }
    for k, v in mapping.items():
        if k in s:
            return v
    m = ''.join(ch for ch in s if ch.isdigit())
    return float(m) if m else 1.0


def _parse_number_no_regex(x) -> float:
    """Parse numeric-looking values without regex (e.g. 89,489 km, 2,017, 78.4k)."""
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x).strip().lower()
    if not s:
        return np.nan

    # Keep alnum/dot/minus/space, strip common separators/symbols.
    filtered = ''.join(ch if (ch.isalnum() or ch in '.- ') else ' ' for ch in s)
    filtered = filtered.replace(',', '')

    for token in filtered.split():
        if token in {'km', 'kms', 'owner', 'owners', 'yr', 'yrs', 'year', 'years'}:
            continue

        mult = 1.0
        t = token

        # Handle shorthand scales in tokens (e.g. 78.4k, 1.2l).
        if t.endswith('lakh'):
            mult = 100_000.0
            t = t[:-4]
        elif t.endswith('lac'):
            mult = 100_000.0
            t = t[:-3]
        elif t.endswith('km'):
            t = t[:-2]

        if t.endswith('k') and len(t) > 1:
            mult = 1_000.0
            t = t[:-1]
        elif t.endswith('l') and len(t) > 1:
            mult = 100_000.0
            t = t[:-1]

        if not t or t in {'-', '.'}:
            continue

        try:
            return float(t) * mult
        except ValueError:
            continue

    return np.nan


def _print_metrics(label: str, actual: pd.Series, pred: pd.Series) -> None:
    valid = actual.notna() & pred.notna() & (actual != 0)
    if not valid.any():
        print(f"{label}: no valid rows for metric computation")
        return

    a = actual[valid]
    p = pred[valid]
    mse = mean_squared_error(a, p)
    rmse = np.sqrt(mse)
    mae = np.abs(p - a).mean()
    median_ae = np.median(np.abs(p - a))
    r2 = float('nan') if len(a) < 2 else r2_score(a, p)
    mape = (((p - a).abs() / a).mean()) * 100
    mpe = (((p - a) / a).mean()) * 100

    print(f"\n--- {label} ---")
    print(f"Rows : {int(valid.sum())}")
    print(f"MSE  : {mse:,.0f}")
    print(f"RMSE : {rmse:,.0f}")
    print(f"MAE  : {mae:,.0f}")
    print(f"Median AE : {median_ae:,.0f}")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"MPE  : {mpe:+.2f}%")


def _print_segment_error_summary(df: pd.DataFrame, actual_col: str, pred_col: str, label: str) -> None:
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
    for c in required_cols:
        if c not in work.columns:
            work[c] = np.nan

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

    band_stats = work.groupby('price_band', observed=True).agg(
        mape=('ape', 'mean'),
        mpe=('pe', 'mean')
    ).sort_index()
    _print_group_table(f"Segment Error Summary ({label}) by Price Band", band_stats)

    top_makes = work['Make'].fillna('Unknown').astype(str).value_counts().head(10).index
    make_stats = work[work['Make'].isin(top_makes)].groupby('Make').agg(
        mape=('ape', 'mean'),
        mpe=('pe', 'mean')
    ).sort_values(by='mape')
    _print_group_table(f"Segment Error Summary ({label}) by Top Makes", make_stats, index_name='Make')

    year_stats = (
        work.dropna(subset=['Year'])
        .assign(Year=lambda d: d['Year'].astype(int))
        .groupby('Year')
        .agg(mape=('ape', 'mean'), mpe=('pe', 'mean'))
        .sort_index()
    )
    _print_group_table(f"Segment Error Summary ({label}) by Manufacturing Year", year_stats, index_name='Year')

    top_cities = work['City'].value_counts().head(10).index
    city_stats = work[work['City'].isin(top_cities)].groupby('City').agg(
        mape=('ape', 'mean'),
        mpe=('pe', 'mean')
    ).sort_values(by='mape')
    _print_group_table(f"Segment Error Summary ({label}) by Top Cities", city_stats, index_name='City')

    kms_bins = [-np.inf, 20_000, 40_000, 60_000, 100_000, 150_000, np.inf]
    kms_names = ['<=20k', '20k-40k', '40k-60k', '60k-100k', '100k-150k', '>150k']
    work['kms_band'] = pd.cut(work['KMs Driven'], bins=kms_bins, labels=kms_names)
    kms_stats = work.groupby('kms_band', observed=True).agg(
        mape=('ape', 'mean'),
        mpe=('pe', 'mean')
    ).sort_index()
    _print_group_table(f"Segment Error Summary ({label}) by KMs Driven Range", kms_stats)

    own_bins = [0, 1, 2, 3, 4, np.inf]
    own_names = ['1', '2', '3', '4', '5+']
    work['ownership_band'] = pd.cut(work['Ownership'], bins=own_bins, labels=own_names, include_lowest=True)
    ownership_stats = work.groupby('ownership_band', observed=True).agg(
        mape=('ape', 'mean'),
        mpe=('pe', 'mean')
    ).sort_index()
    _print_group_table(f"Segment Error Summary ({label}) by Ownership", ownership_stats)

    transmission_stats = work.groupby('Transmission').agg(
        mape=('ape', 'mean'),
        mpe=('pe', 'mean')
    ).sort_values(by='mape')
    _print_group_table(f"Segment Error Summary ({label}) by Transmission", transmission_stats, index_name='Transmission')

    fuel_stats = work.groupby('Fuel').agg(
        mape=('ape', 'mean'),
        mpe=('pe', 'mean')
    ).sort_values(by='mape')
    _print_group_table(f"Segment Error Summary ({label}) by Fuel", fuel_stats, index_name='Fuel')

    bodytype_stats = work.groupby('BodyType').agg(
        mape=('ape', 'mean'),
        mpe=('pe', 'mean')
    ).sort_values(by='mape')
    _print_group_table(f"Segment Error Summary ({label}) by BodyType", bodytype_stats, index_name='BodyType')


def predict_batch(input_path: str, output_path: Optional[str] = None) -> None:
    artifacts = load_primary_artifacts()
    df, pred = predict_primary_rows(_read_any(input_path), artifacts)
    df['Predicted Price'] = pred

    actual_col = None
    for column in ['Price (₹)', 'Price (â‚¹)', 'Price (Ã¢â€šÂ¹)', 'Price (?)', 'Price', 'Target Price', 'price']:
        if column in df.columns:
            actual_col = column
            break

    if actual_col is not None:
        actual = pd.to_numeric(df[actual_col], errors='coerce')
        predicted = pd.to_numeric(df['Predicted Price'], errors='coerce')

        df['Error'] = predicted - actual
        df['Absolute Error'] = (predicted - actual).abs()
        df['Absolute Percentage Error (%)'] = (df['Absolute Error'] / actual.replace(0, np.nan)) * 100

        _print_metrics('Primary Model Prediction', actual, predicted)
        _print_segment_error_summary(df, actual_col, 'Predicted Price', 'Primary')

    out = output_path or _default_output_path(input_path)
    _write_any(df, out)
    print(f"Saved predictions to: {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch price prediction from Excel/CSV using the primary model artifacts.')
    parser.add_argument('--input', required=True, help='Path to input .xlsx/.xls/.csv file')
    parser.add_argument('--output', default='', help='Optional output path (.xlsx/.xls/.csv)')
    args = parser.parse_args()

    predict_batch(args.input, args.output or None)
