import glob
import os
import re
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

CURRENT_YEAR = datetime.now().year
MARKET_EPOCH = datetime(2020, 1, 1)
PREPROCESSOR_VERSION = 10
BASE_NUMERIC_FEATURES = [
    'Year',
    'KMs Driven',
    'Car Age',
    'KMs/Year',
    'Ownership',
    'Log KMs',
    'Age x Ownership',
    'KMs x Ownership',
    'Fetch Month',
    'Market Days',
    'Days On Market',
]
DYNAMICS_FEATURES = [
    'Expected Avg Drop 7d',
    'Expected Avg Drop 30d',
    'Expected Price Drop Rate',
    'Expected Time To First Drop',
    'Expected Time To Sell',
    'Expected Price Volatility',
    'Expected Market Liquidity Score',
]
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + DYNAMICS_FEATURES
CATEGORICAL_FEATURES = ['Make', 'Model', 'Variant', 'Transmission', 'Fuel', 'BodyType', 'City', 'Reg State']
OTHER_CATEGORY = '__OTHER__'

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
    'price (â‚¹)': 'Price (₹)',
    'price(â‚¹)': 'Price (₹)',
    'price': 'Price (₹)',
    'registration': 'Registration',
    'image': 'Image',
    'fetched on': 'Fetched On',
    'date': 'Fetched On',
}

_MULTI_WORD_MAKES = [
    'MARUTI SUZUKI',
    'LAND ROVER',
    'ASTON MARTIN',
    'ROLLS ROYCE',
    'FORCE MOTORS',
    'MINI COOPER',
]

_TRANSMISSION_MAP = {
    'M': 'Manual',
    'm': 'Manual',
    'MANUAL': 'Manual',
    'manual': 'Manual',
    'A': 'Automatic',
    'a': 'Automatic',
    'AUTOMATIC': 'Automatic',
    'automatic': 'Automatic',
}

_WORD_2_OWN = {
    'first': 1,
    '1st': 1,
    'one': 1,
    'second': 2,
    '2nd': 2,
    'two': 2,
    'third': 3,
    '3rd': 3,
    'three': 3,
    'fourth': 4,
    '4th': 4,
    'four': 4,
}

_BODYTYPE_MAP = {
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

_FUEL_ALIASES = {
    'PETROL + CNG': 'PETROL+CNG',
    'PETROL/CNG': 'PETROL+CNG',
    'CNG + PETROL': 'PETROL+CNG',
    'HYBRID ELECTRIC': 'HYBRID',
    'HYBRID PETROL': 'HYBRID',
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for column in df.columns:
        key = re.sub(r'\s+', ' ', str(column).strip().lower())
        if key in CANON:
            rename_map[column] = CANON[key]
    return df.rename(columns=rename_map)


def _split_name_to_make_model(name_series: pd.Series) -> tuple[list[str], list[str]]:
    makes, models = [], []
    for raw in name_series.fillna('Unknown'):
        value = str(raw).strip().upper()
        matched = False
        for prefix in _MULTI_WORD_MAKES:
            if value.startswith(prefix):
                makes.append(prefix.title())
                remainder = value[len(prefix):].strip()
                models.append(remainder.title() if remainder else 'Unknown')
                matched = True
                break
        if not matched:
            parts = value.split(None, 1)
            makes.append(parts[0].title() if parts else 'Unknown')
            models.append(parts[1].title() if len(parts) > 1 else 'Unknown')
    return makes, models


def _parse_kms(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower().replace(',', '')
    match = re.search(r'(\d+(?:\.\d+)?)', s)
    if not match:
        return np.nan
    value = float(match.group(1))
    if 'l' in s:
        value *= 100_000
    elif 'k' in s:
        value *= 1_000
    return float(round(value, 0))


def _parse_ownership(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    match = re.search(r'(\d+)', s)
    if match:
        return int(match.group(1))
    for key, value in _WORD_2_OWN.items():
        if key in s:
            return value
    return np.nan


def _reg_state(x):
    if pd.isna(x):
        return 'UNK'
    s = str(x).strip().upper()
    match = re.match(r'([A-Z]{2})', s)
    return match.group(1) if match else 'UNK'


def _normalize_text_value(x, default='Unknown'):
    if pd.isna(x):
        return default
    s = re.sub(r'\s+', ' ', str(x).strip())
    return s if s else default


def _normalize_category_key(x) -> str:
    return re.sub(r'\s+', ' ', str(x).strip()).casefold()


def _normalize_fuel_value(x):
    value = _normalize_text_value(x).upper()
    return _FUEL_ALIASES.get(value, value)


def _normalize_bodytype_value(x):
    value = _normalize_text_value(x).upper()
    return _BODYTYPE_MAP.get(value, _normalize_text_value(x))


def _normalize_city_value(x):
    value = _normalize_text_value(x)
    return value.upper() if value != 'Unknown' else value


def _normalize_reg_state_value(x):
    value = _normalize_text_value(x, default='UNK').upper()
    if value == 'UNKNOWN':
        return 'UNK'
    return value[:2] if len(value) >= 2 else 'UNK'


def _normalize_text_series(series: pd.Series, default='Unknown') -> pd.Series:
    return series.map(lambda value: _normalize_text_value(value, default=default))


def add_model_features(df: pd.DataFrame, reference_datetime: datetime | None = None) -> pd.DataFrame:
    work = df.copy()
    ref_dt = pd.Timestamp(reference_datetime or datetime.now())

    year = pd.to_numeric(work['Year'], errors='coerce')
    kms = pd.to_numeric(work.get('KMs Driven', pd.Series([np.nan] * len(work))), errors='coerce')
    ownership = pd.to_numeric(work['Ownership'], errors='coerce').fillna(1.0)

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
    if 'Fetched On' in work.columns and fetched_on.notna().any():
        if 'ID' in work.columns:
            first_seen = fetched_on.groupby(work['ID']).transform('min')
            days_on_market = (fetched_on - first_seen).dt.days
            work['Days On Market'] = days_on_market.fillna(0.0).astype(float)
        else:
            work['Days On Market'] = 0.0
    else:
        work['Days On Market'] = 0.0

    return work


def keep_first_snapshot_per_listing(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the earliest snapshot per listing ID for model training."""
    if 'ID' not in df.columns:
        return df.reset_index(drop=True)

    listing_id = df['ID'].astype('string').str.strip()
    has_listing_id = listing_id.notna() & (listing_id != '')
    if not has_listing_id.any():
        return df.reset_index(drop=True)

    work = df.copy()
    work['_row_order'] = np.arange(len(work))

    with_id = work.loc[has_listing_id].copy()
    with_id['_listing_id'] = listing_id.loc[has_listing_id]

    sort_cols = ['_listing_id', '_row_order']
    if 'Fetched On' in with_id.columns:
        with_id['_fetched_on'] = pd.to_datetime(with_id['Fetched On'], errors='coerce')
        sort_cols = ['_listing_id', '_fetched_on', '_row_order']

    with_id = with_id.sort_values(sort_cols, kind='mergesort', na_position='last')
    with_id = with_id.drop_duplicates(subset=['_listing_id'], keep='first')

    without_id = work.loc[~has_listing_id].copy()
    deduped = pd.concat([with_id, without_id], axis=0)
    deduped = deduped.sort_values('_row_order', kind='mergesort')
    deduped = deduped.drop(columns=['_row_order', '_listing_id', '_fetched_on'], errors='ignore')
    return deduped.reset_index(drop=True)


def load_and_clean(path):
    if path.lower().endswith(('.xlsx', '.xls')):
        parquet_path = path.rsplit('.', 1)[0] + '.parquet'
        if os.path.exists(parquet_path) and os.path.getmtime(parquet_path) >= os.path.getmtime(path):
            print(f"  Loading cached {os.path.basename(parquet_path)}")
            df = pd.read_parquet(parquet_path)
        else:
            print(f"  Reading {os.path.basename(path)} and caching parquet")
            df = pd.read_excel(path)
            df.to_parquet(parquet_path, index=False)
            print(f"  Cached {os.path.basename(parquet_path)}")
    elif path.lower().endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df = standardize_columns(df)

    if 'Name' in df.columns and 'Make' not in df.columns:
        makes, models = _split_name_to_make_model(df['Name'])
        df['Make'] = makes
        df['Model'] = models

    if 'Transmission' in df.columns:
        df['Transmission'] = df['Transmission'].astype(str).str.strip().map(_TRANSMISSION_MAP).fillna(df['Transmission'])

    if 'BodyType' in df.columns:
        df['BodyType'] = df['BodyType'].map(_normalize_bodytype_value)

    for column in ['KMs Driven', 'Year', 'Ownership', 'Price (₹)']:
        if column not in df.columns:
            df[column] = np.nan

    df['KMs Driven'] = df['KMs Driven'].apply(_parse_kms)
    df['Ownership'] = df['Ownership'].apply(_parse_ownership)
    df['Year'] = (
        df['Year']
        .astype(str)
        .str.replace(r'[^\d]', '', regex=True)
        .replace({'': np.nan})
        .astype(float)
    )
    df['Price (₹)'] = (
        df['Price (₹)']
        .astype(str)
        .str.replace(r'[^\d.]', '', regex=True)
        .replace({'': np.nan})
        .astype(float)
    )
    df['Price (â‚¹)'] = df['Price (₹)']

    if 'Fetched On' in df.columns:
        df['Fetched On'] = pd.to_datetime(df['Fetched On'], errors='coerce')
        df['_snapshot_year'] = df['Fetched On'].dt.year.fillna(CURRENT_YEAR).astype(float)
    else:
        df['_snapshot_year'] = float(CURRENT_YEAR)

    df['Reg State'] = df['Registration'].apply(_reg_state) if 'Registration' in df.columns else 'UNK'
    df = add_model_features(df)

    for column in ['Make', 'Model', 'Variant', 'Transmission', 'Fuel', 'BodyType', 'City']:
        if column not in df.columns:
            df[column] = 'Unknown'

    df['Make'] = _normalize_text_series(df['Make'])
    df['Model'] = _normalize_text_series(df['Model'])
    df['Variant'] = _normalize_text_series(df['Variant'])
    df['Transmission'] = df['Transmission'].map(
        lambda value: _TRANSMISSION_MAP.get(str(value).strip(), _normalize_text_value(value))
    )
    df['Fuel'] = df['Fuel'].map(_normalize_fuel_value)
    df['BodyType'] = df['BodyType'].map(_normalize_bodytype_value)
    df['City'] = df['City'].map(_normalize_city_value)
    df['Reg State'] = df['Reg State'].map(_normalize_reg_state_value)

    if 'ID' in df.columns:
        for column in ['Year', 'Fuel', 'Transmission']:
            if column in df.columns:
                nunique = df.groupby('ID')[column].transform('nunique')
                df = df[nunique <= 1]

    df = df.dropna(subset=['Price (₹)', 'Year', 'KMs Driven'])
    df = df[(df['Year'] >= 2005) & (df['Year'] <= df['_snapshot_year'])]
    df = df[(df['KMs Driven'] >= 0) & (df['KMs Driven'] <= 400_000)]

    lo, hi = df['Price (₹)'].quantile([0.01, 0.99])
    df = df[(df['Price (₹)'] >= lo) & (df['Price (₹)'] <= hi)]

    df = df.drop(columns=['_snapshot_year'], errors='ignore')
    df['Brand'] = df['Make']

    return df.reset_index(drop=True)


def prepare_model_input(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    numeric_feats = meta['numeric_feats']
    categorical_feats = meta['categorical_feats']
    numeric_fill_values = meta['numeric_fill_values']
    categorical_levels = meta['categorical_levels']

    X = df.copy()

    for column in numeric_feats:
        if column not in X.columns:
            X[column] = np.nan
        X[column] = pd.to_numeric(X[column], errors='coerce').fillna(numeric_fill_values[column]).astype(float)

    for column in categorical_feats:
        levels = categorical_levels[column]
        lookup = {_normalize_category_key(level): level for level in levels if level != OTHER_CATEGORY}
        if column not in X.columns:
            X[column] = OTHER_CATEGORY
        normalized = (
            X[column]
            .fillna(OTHER_CATEGORY)
            .astype(str)
            .map(lambda value: lookup.get(_normalize_category_key(value), _normalize_text_value(value, default=OTHER_CATEGORY)))
        )
        normalized = normalized.where(normalized.isin(levels), OTHER_CATEGORY)
        X[column] = pd.Categorical(normalized, categories=levels)

    return X[numeric_feats + categorical_feats]


def is_native_lightgbm_preprocessor(meta: dict) -> bool:
    return (
        meta.get('preprocessor_type') == 'lightgbm_native'
        and int(meta.get('preprocessor_version', 0)) >= PREPROCESSOR_VERSION
    )


def preprocessor_uses_market_dynamics(meta: dict) -> bool:
    numeric_feats = meta.get('numeric_feats', [])
    return all(feature in numeric_feats for feature in DYNAMICS_FEATURES)


def build_feature_meta(
    df: pd.DataFrame,
    numeric_feats: list[str] | None = None,
    categorical_feats: list[str] | None = None,
    *,
    preprocessor_type: str = 'lightgbm_native',
    preprocessor_version: int = PREPROCESSOR_VERSION,
) -> dict:
    numeric_feats = numeric_feats or NUMERIC_FEATURES
    categorical_feats = categorical_feats or CATEGORICAL_FEATURES

    numeric_fill_values = {}
    for column in numeric_feats:
        series = pd.to_numeric(df[column], errors='coerce') if column in df.columns else pd.Series(dtype=float)
        median = series.median()
        numeric_fill_values[column] = float(median) if pd.notna(median) else 0.0

    categorical_levels = {}
    for column in categorical_feats:
        if column in df.columns:
            values = df[column].fillna(OTHER_CATEGORY).astype(str).str.strip().replace({'': OTHER_CATEGORY})
        else:
            values = pd.Series([OTHER_CATEGORY], dtype='object')
        levels = sorted(value for value in values.unique().tolist() if value != OTHER_CATEGORY)
        levels.append(OTHER_CATEGORY)
        categorical_levels[column] = levels

    return {
        'preprocessor_type': preprocessor_type,
        'preprocessor_version': preprocessor_version,
        'numeric_feats': numeric_feats,
        'categorical_feats': categorical_feats,
        'numeric_fill_values': numeric_fill_values,
        'categorical_levels': categorical_levels,
    }


def build_preprocessor(df):
    meta = build_feature_meta(df)

    out_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    joblib.dump(meta, out_path)
    print(f"Preprocessor saved to {out_path}")
    return meta


__all__ = [
    'BASE_DIR',
    'DATA_DIR',
    'MODELS_DIR',
    'CURRENT_YEAR',
    'MARKET_EPOCH',
    'PREPROCESSOR_VERSION',
    'BASE_NUMERIC_FEATURES',
    'DYNAMICS_FEATURES',
    'NUMERIC_FEATURES',
    'CATEGORICAL_FEATURES',
    'OTHER_CATEGORY',
    'CANON',
    '_TRANSMISSION_MAP',
    '_split_name_to_make_model',
    '_parse_kms',
    '_parse_ownership',
    '_reg_state',
    'standardize_columns',
    'add_model_features',
    'keep_first_snapshot_per_listing',
    'load_and_clean',
    'prepare_model_input',
    'is_native_lightgbm_preprocessor',
    'preprocessor_uses_market_dynamics',
    'build_feature_meta',
    'build_preprocessor',
]
