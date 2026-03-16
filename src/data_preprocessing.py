# data_preprocessing.py
import os, glob, re
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

CURRENT_YEAR = 2025

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
    'price (₹)': 'Price (₹)',
    'price(₹)': 'Price (₹)',
    'price': 'Price (₹)',
    'registration': 'Registration',
    'image': 'Image',
    'fetched on': 'Fetched On',
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

# ── CLEANING FUNCTION ──────────────────────────────────────────────────────────
def load_and_clean(path):
    # Support both .xlsx and .csv
    if path.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df = standardize_columns(df)

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

    # Timestamps if present
    if 'Fetched On' in df.columns:
        df['Fetched On'] = pd.to_datetime(df['Fetched On'], errors='coerce')

    # Registration → state
    df['Reg State'] = df['Registration'].apply(_reg_state) if 'Registration' in df else 'UNK'

    # Derivations
    df['Car Age'] = (CURRENT_YEAR - df['Year']).clip(lower=0)
    df['KMs/Year'] = np.where(df['Car Age'] > 0, df['KMs Driven'] / df['Car Age'], df['KMs Driven'])

    # Minimal normalizations
    for c in ['City','Make','Model','Variant','Transmission','Fuel','BodyType']:
        if c not in df: df[c] = 'Unknown'
        df[c] = df[c].fillna('Unknown').astype(str).str.strip()

    # Normalize body type like 'Suv' → 'SUV'
    df['BodyType'] = df['BodyType'].replace({'Suv': 'SUV', 'Muv': 'MUV'})

    # Deduplicate by ID with latest snapshot
    if 'ID' in df.columns and 'Fetched On' in df.columns:
        df.sort_values('Fetched On', inplace=True)
        df = df.drop_duplicates(subset=['ID'], keep='last')

    # Filter sanity
    df = df.dropna(subset=['Price (₹)', 'Year', 'KMs Driven'])
    df = df[(df['Year'] >= 2005) & (df['Year'] <= CURRENT_YEAR)]
    df = df[(df['KMs Driven'] >= 0) & (df['KMs Driven'] <= 400_000)]

    lo, hi = df['Price (₹)'].quantile([0.01, 0.99])
    df = df[(df['Price (₹)'] >= lo) & (df['Price (₹)'] <= hi)]

    # Backward-compat: Brand = Make
    df['Brand'] = df['Make']

    return df.reset_index(drop=True)

# ── PREPROCESSOR ───────────────────────────────────────────────────────────────
def build_preprocessor(df):
    numeric_feats = ['Year','KMs Driven','Car Age','KMs/Year','Ownership']
    categorical_feats = ['City','Make','Model','Variant','Transmission','Fuel','BodyType','Reg State']

    num_pipe = Pipeline([('impute', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore')),
    ])

    pre = ColumnTransformer([
        ('num', num_pipe, numeric_feats),
        ('cat', cat_pipe, categorical_feats),
    ])

    pre.fit(df[numeric_feats + categorical_feats])

    out_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    joblib.dump({'preprocessor': pre,
                 'numeric_feats': numeric_feats,
                 'categorical_feats': categorical_feats}, out_path)
    print(f"✅ Preprocessor saved to {out_path}")
    return pre

# ── SCRIPT ENTRYPOINT ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    files = (
        glob.glob(os.path.join(DATA_DIR, 'Cars24_*.*')) +
        glob.glob(os.path.join(DATA_DIR, 'Spinny_*.*'))
    )
    if not files:
        raise RuntimeError(f"No data files found in {DATA_DIR}")

    print(f"🔍 Loading {len(files)} files...")
    df = pd.concat([load_and_clean(f) for f in files], ignore_index=True)
    build_preprocessor(df)
