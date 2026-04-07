# data_preprocessing.py
r'''
import os, glob, re
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

CURRENT_YEAR = datetime.now().year
MARKET_EPOCH = datetime(2020, 1, 1)
PREPROCESSOR_VERSION = 9
NUMERIC_FEATURES = ['Year','KMs Driven','Car Age','KMs/Year','Ownership',
                    'Log KMs','Age x Ownership','KMs x Ownership',
                    'Fetch Month','Market Days']
CATEGORICAL_FEATURES = ['Make','Model','Variant','Transmission','Fuel','BodyType','City','Reg State']
OTHER_CATEGORY = '__OTHER__'

# ── COLUMN NORMALIZATION ───────────────────────────────────────────────────────
# map many possible header spellings to a single canonical name
CANON = {
    'id': 'ID',
    'city': 'City',
    'name': 'Name',
    'make': 'Make',
    'model': 'Model',
    'variant': 'Variant',
    'year': 'Year',
    'km driven': 'KMs Driven',
    'kms driven': 'KMs Driven',
    'k m driven': 'KMs Driven',
    'km_driven': 'KMs Driven',
    'kms_driven': 'KMs Driven',
    'ownership': 'Ownership',
    'transmission': 'Transmission',
    'fuel': 'Fuel',
    'bodytype': 'BodyType',
    'body type': 'BodyType',
    'body_type': 'BodyType',
    'price (₹)': 'Price (₹)',
    'price(₹)': 'Price (₹)',
    'price': 'Price (₹)',
    'registration': 'Registration',
    'image': 'Image',
    'fetched on': 'Fetched On',
    'date': 'Fetched On',
}

# Known multi-word makes that must not be split naively.
_MULTI_WORD_MAKES = [
    'MARUTI SUZUKI', 'LAND ROVER', 'ASTON MARTIN', 'ROLLS ROYCE',
    'FORCE MOTORS', 'MINI COOPER',
]

def _split_name_to_make_model(name_series: pd.Series) -> tuple:
    """Split a 'Name' column like 'MARUTI SUZUKI SWIFT' into (Make, Model)."""
    makes, models = [], []
    for raw in name_series.fillna('Unknown'):
        val = str(raw).strip().upper()
        matched = False
        for prefix in _MULTI_WORD_MAKES:
            if val.startswith(prefix):
                makes.append(prefix.title())
                remainder = val[len(prefix):].strip()
                models.append(remainder.title() if remainder else 'Unknown')
                matched = True
                break
        if not matched:
            parts = val.split(None, 1)
            makes.append(parts[0].title() if parts else 'Unknown')
            models.append(parts[1].title() if len(parts) > 1 else 'Unknown')
    return makes, models

_TRANSMISSION_MAP = {
    'M': 'Manual', 'm': 'Manual', 'MANUAL': 'Manual', 'manual': 'Manual',
    'A': 'Automatic', 'a': 'Automatic', 'AUTOMATIC': 'Automatic', 'automatic': 'Automatic',
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        key = re.sub(r'\s+', ' ', str(c).strip().lower())
        if key in CANON:
            rename_map[c] = CANON[key]
    return df.rename(columns=rename_map)

# ── HELPERS ────────────────────────────────────────────────────────────────────
def _parse_kms(x):
    """Handle strings like '78.42k km', '1.2L km', '45,000', '28720'."""
    if pd.isna(x): return np.nan
    s = str(x).strip().lower().replace(',', '')
    m = re.search(r'(\d+(?:\.\d+)?)', s)
    if not m: return np.nan
    val = float(m.group(1))
    if 'l' in s:      # lakh
        val *= 100_000
    elif 'k' in s:
        val *= 1_000
    return float(round(val, 0))

_WORD_2_OWN = {
    'first': 1, '1st': 1, 'one': 1,
    'second': 2, '2nd': 2, 'two': 2,
    'third': 3, '3rd': 3, 'three': 3,
    'fourth': 4, '4th': 4, 'four': 4,
}
def _parse_ownership(x):
    """'1st owner', '3st owner', 'second owner' → int"""
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    m = re.search(r'(\d+)', s)
    if m: return int(m.group(1))
    for k, v in _WORD_2_OWN.items():
        if k in s: return v
    return np.nan

def _reg_state(x):
    """KA51**9597 or KA01 → KA"""
    if pd.isna(x): return 'UNK'
    s = str(x).strip().upper()
    m = re.match(r'([A-Z]{2})', s)
    return m.group(1) if m else 'UNK'


def _normalize_text_value(x, default='Unknown'):
    if pd.isna(x):
        return default
    s = re.sub(r'\s+', ' ', str(x).strip())
    return s if s else default


def _normalize_category_key(x) -> str:
    return re.sub(r'\s+', ' ', str(x).strip()).casefold()


def _normalize_fuel_value(x):
    s = _normalize_text_value(x).upper()
    fuel_aliases = {
        'PETROL + CNG': 'PETROL+CNG',
        'PETROL/CNG': 'PETROL+CNG',
        'CNG + PETROL': 'PETROL+CNG',
        'HYBRID ELECTRIC': 'HYBRID',
        'HYBRID PETROL': 'HYBRID',
    }
    return fuel_aliases.get(s, s)


def _normalize_bodytype_value(x):
    s = _normalize_text_value(x).upper()
    bodytype_map = {
        'HATCHBACK': 'Hatchback',
        'SEDAN': 'Sedan',
        'SUV': 'SUV',
        'MUV': 'MUV',
        'CROSSOVER': 'Crossover',
        'COUPE': 'Coupe',
        'CONVERTIBLE': 'Convertible',
        'WAGON': 'Wagon',
        'VAN': 'Van',
        'PICKUP': 'Pickup',
        'PICKUP TRUCK': 'Pickup',
    }
    return bodytype_map.get(s, _normalize_text_value(x))


def _normalize_city_value(x):
    s = _normalize_text_value(x)
    return s.upper() if s != 'Unknown' else s


def _normalize_reg_state_value(x):
    s = _normalize_text_value(x, default='UNK').upper()
    if s == 'UNKNOWN':
        return 'UNK'
    return s[:2] if len(s) >= 2 else 'UNK'


def add_model_features(df: pd.DataFrame, reference_datetime: datetime | None = None) -> pd.DataFrame:
    """
    Add model features that can be derived from car attributes plus the current
    or snapshot date. Listing-history-only fields such as days on market are
    intentionally excluded.
    """
    work = df.copy()
    ref_dt = pd.Timestamp(reference_datetime or datetime.now())

    year = pd.to_numeric(work.get('Year'), errors='coerce')
    kms = pd.to_numeric(work.get('KMs Driven'), errors='coerce')
    ownership = pd.to_numeric(work.get('Ownership'), errors='coerce').fillna(1.0)

    if 'Fetched On' in work.columns:
        fetched_on = pd.to_datetime(work['Fetched On'], errors='coerce')
    else:
        fetched_on = pd.Series(pd.NaT, index=work.index, dtype='datetime64[ns]')

    snapshot_year = fetched_on.dt.year.fillna(float(ref_dt.year))
    car_age = (snapshot_year - year).clip(lower=0)
    kms_safe = kms.clip(lower=0)

    work['Car Age'] = car_age.astype(float)
    work['KMs/Year'] = np.where(car_age > 0, kms / car_age, kms)
    work['Log KMs'] = np.log1p(kms_safe)
    work['Age x Ownership'] = work['Car Age'] * ownership
    work['KMs x Ownership'] = kms * ownership
    work['Fetch Month'] = fetched_on.dt.month.fillna(float(ref_dt.month)).astype(float)
    market_days = (fetched_on - MARKET_EPOCH).dt.days
    work['Market Days'] = market_days.fillna(float((ref_dt - MARKET_EPOCH).days)).astype(float)

    return work

# ── CLEANING FUNCTION ──────────────────────────────────────────────────────────
def load_and_clean(path):
    # Auto-cache xlsx → parquet for fast reloads
    if path.lower().endswith(('.xlsx', '.xls')):
        parquet_path = path.rsplit('.', 1)[0] + '.parquet'
        if os.path.exists(parquet_path) and os.path.getmtime(parquet_path) >= os.path.getmtime(path):
            print(f"  ⚡ Loading cached {os.path.basename(parquet_path)}")
            df = pd.read_parquet(parquet_path)
        else:
            print(f"  📖 Reading {os.path.basename(path)} (slow, will cache as parquet)…")
            df = pd.read_excel(path)
            df.to_parquet(parquet_path, index=False)
            print(f"  💾 Cached → {os.path.basename(parquet_path)}")
    elif path.lower().endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df = standardize_columns(df)

    # Split combined Name column into Make / Model if separate columns are absent.
    if 'Name' in df.columns and 'Make' not in df.columns:
        makes, models = _split_name_to_make_model(df['Name'])
        df['Make'] = makes
        df['Model'] = models

    # Map short transmission codes to full words.
    if 'Transmission' in df.columns:
        df['Transmission'] = df['Transmission'].astype(str).str.strip().map(_TRANSMISSION_MAP).fillna(df['Transmission'])

    # Normalize BodyType casing (HATCHBACK → Hatchback, SUV stays SUV via explicit map).
    if 'BodyType' in df.columns:
        df['BodyType'] = df['BodyType'].map(_normalize_bodytype_value)

    # Ensure key columns exist (create empty if missing)
    for must in ['KMs Driven','Year','Ownership','Price (₹)']:
        if must not in df.columns:
            df[must] = np.nan

    # Parse/clean fields
    df['KMs Driven'] = df['KMs Driven'].apply(_parse_kms)

    df['Ownership'] = df['Ownership'].apply(_parse_ownership)

    # Year sometimes appears as '2,017'
    df['Year'] = (
        df['Year'].astype(str)
                  .str.replace(r'[^\d]', '', regex=True)
                  .replace({'': np.nan})
                  .astype(float)
    )

    # Price may include commas/symbols
    df['Price (₹)'] = (
        df['Price (₹)'].astype(str)
                       .str.replace(r'[^\d.]', '', regex=True)
                       .replace({'': np.nan})
                       .astype(float)
    )

    # Timestamps if present; use row-level snapshot year for age derivation.
    if 'Fetched On' in df.columns:
        df['Fetched On'] = pd.to_datetime(df['Fetched On'], errors='coerce')
        df['_snapshot_year'] = df['Fetched On'].dt.year.fillna(CURRENT_YEAR).astype(float)
    else:
        df['_snapshot_year'] = float(CURRENT_YEAR)

    # Registration → state
    df['Reg State'] = df['Registration'].apply(_reg_state) if 'Registration' in df else 'UNK'

    # Derivations
    df['Car Age'] = (df['_snapshot_year'] - df['Year']).clip(lower=0)
    df['KMs/Year'] = np.where(df['Car Age'] > 0, df['KMs Driven'] / df['Car Age'], df['KMs Driven'])
    df['Log KMs'] = np.log1p(df['KMs Driven'])
    df['Age x Ownership'] = df['Car Age'] * df['Ownership'].fillna(1)
    df['KMs x Ownership'] = df['KMs Driven'] * df['Ownership'].fillna(1)

    # ── Time-based features (from snapshot date) ──
    if 'Fetched On' in df.columns and df['Fetched On'].notna().any():
        df['Fetch Month'] = df['Fetched On'].dt.month.fillna(6).astype(float)
        df['Market Days'] = (df['Fetched On'] - MARKET_EPOCH).dt.days.astype(float)
        df['Market Days'] = df['Market Days'].fillna(0.0)
        # Days On Market: how long since this listing first appeared
        if 'ID' in df.columns:
            first_seen = df.groupby('ID')['Fetched On'].transform('min')
            df['Days On Market'] = (df['Fetched On'] - first_seen).dt.days.astype(float)
            df['Days On Market'] = df['Days On Market'].fillna(0.0)
        else:
            df['Days On Market'] = 0.0
    else:
        now = datetime.now()
        df['Fetch Month'] = float(now.month)
        df['Market Days'] = float((now - MARKET_EPOCH).days)
        df['Days On Market'] = 0.0

    # Minimal normalizations
    for c in ['Make','Model','Variant','Transmission','Fuel','BodyType','City']:
        if c not in df: df[c] = 'Unknown'
        df[c] = df[c].fillna('Unknown').astype(str).str.strip()

    # Normalize body type like 'Suv' → 'SUV'
    df['BodyType'] = df['BodyType'].replace({'Suv': 'SUV', 'Muv': 'MUV'})

    # Remove scraper-bug rows: IDs where identity columns flip across snapshots
    if 'ID' in df.columns:
        for col in ['Year', 'Fuel', 'Transmission']:
            if col in df.columns:
                nunique = df.groupby('ID')[col].transform('nunique')
                df = df[nunique <= 1]

    # Filter sanity
    df = df.dropna(subset=['Price (₹)', 'Year', 'KMs Driven'])
    df = df[(df['Year'] >= 2005) & (df['Year'] <= df['_snapshot_year'])]
    df = df[(df['KMs Driven'] >= 0) & (df['KMs Driven'] <= 400_000)]

    lo, hi = df['Price (₹)'].quantile([0.01, 0.99])
    df = df[(df['Price (₹)'] >= lo) & (df['Price (₹)'] <= hi)]

    df = df.drop(columns=['_snapshot_year'], errors='ignore')

    # Backward-compat: Brand = Make
    df['Brand'] = df['Make']

    return df.reset_index(drop=True)


def prepare_model_input(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    numeric_feats = meta['numeric_feats']
    categorical_feats = meta['categorical_feats']
    numeric_fill_values = meta['numeric_fill_values']
    categorical_levels = meta['categorical_levels']

    X = df.copy()

    for col in numeric_feats:
        if col not in X.columns:
            X[col] = np.nan
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(numeric_fill_values[col]).astype(float)

    for col in categorical_feats:
        levels = categorical_levels[col]
        allowed = set(levels)
        if col not in X.columns:
            X[col] = OTHER_CATEGORY
        X[col] = X[col].fillna(OTHER_CATEGORY).astype(str).str.strip()
        X[col] = X[col].where(X[col].isin(allowed), OTHER_CATEGORY)
        X[col] = pd.Categorical(X[col], categories=levels)

    return X[numeric_feats + categorical_feats]


def is_native_lightgbm_preprocessor(meta: dict) -> bool:
    return (
        meta.get('preprocessor_type') == 'lightgbm_native'
        and int(meta.get('preprocessor_version', 0)) >= PREPROCESSOR_VERSION
    )

# ── PREPROCESSOR ───────────────────────────────────────────────────────────────
def build_preprocessor(df):
    numeric_feats = NUMERIC_FEATURES
    categorical_feats = CATEGORICAL_FEATURES

    numeric_fill_values = {
        col: float(pd.to_numeric(df[col], errors='coerce').median())
        for col in numeric_feats
    }
    categorical_levels = {}
    for col in categorical_feats:
        values = (
            df[col].fillna(OTHER_CATEGORY).astype(str).str.strip().replace({'': OTHER_CATEGORY})
        )
        levels = sorted(v for v in values.unique().tolist() if v != OTHER_CATEGORY)
        levels.append(OTHER_CATEGORY)
        categorical_levels[col] = levels

    meta = {
        'preprocessor_type': 'lightgbm_native',
        'preprocessor_version': PREPROCESSOR_VERSION,
        'numeric_feats': numeric_feats,
        'categorical_feats': categorical_feats,
        'numeric_fill_values': numeric_fill_values,
        'categorical_levels': categorical_levels,
    }

    out_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    joblib.dump(meta, out_path)
    print(f"✅ Preprocessor saved to {out_path}")
    return meta

'''

import glob
import os

import pandas as pd

from model_preprocessing import *  # noqa: F401,F403,E402
from model_preprocessing import DATA_DIR, build_preprocessor, load_and_clean, keep_first_snapshot_per_listing


def main() -> None:
    files = (
        glob.glob(os.path.join(DATA_DIR, 'normalized_table.*'))
        + glob.glob(os.path.join(DATA_DIR, 'normalized_table_*.*'))
        + glob.glob(os.path.join(DATA_DIR, 'Cars24_*.*'))
        + glob.glob(os.path.join(DATA_DIR, 'Spinny_*.*'))
    )
    files = [
        path
        for path in files
        if not (path.lower().endswith('.parquet') and os.path.exists(path.rsplit('.', 1)[0] + '.xlsx'))
    ]
    if not files:
        raise RuntimeError(f"No data files found in {DATA_DIR}")

    print(f"Loading {len(files)} files...")
    df = pd.concat([load_and_clean(path) for path in files], ignore_index=True)
    original_rows = len(df)
    df = keep_first_snapshot_per_listing(df)
    removed_rows = original_rows - len(df)
    if removed_rows > 0:
        print(f"Using earliest snapshot per listing: kept {len(df):,} rows, removed {removed_rows:,} repeats")
    build_preprocessor(df)


if __name__ == "__main__":
    main()
