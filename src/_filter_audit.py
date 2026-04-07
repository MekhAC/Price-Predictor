"""Audit which filters remove the most rows. Run: python src/_filter_audit.py"""
import os, sys, glob, re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import (
    standardize_columns, _split_name_to_make_model, _TRANSMISSION_MAP,
    _parse_kms, _parse_ownership, _reg_state, CURRENT_YEAR,
)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
files = (
    glob.glob(os.path.join(DATA_DIR, 'normalized_table.*')) +
    glob.glob(os.path.join(DATA_DIR, 'normalized_table_*.*')) +
    glob.glob(os.path.join(DATA_DIR, 'Cars24_*.*')) +
    glob.glob(os.path.join(DATA_DIR, 'Spinny_*.*'))
)

print(f"Loading {len(files)} files (raw)...")
raw_dfs = []
for f in files:
    print(f"  Reading {os.path.basename(f)}...", end=" ", flush=True)
    if f.lower().endswith(('.xlsx', '.xls')):
        d = pd.read_excel(f)
    else:
        d = pd.read_csv(f)
    print(f"{len(d):,} rows")
    raw_dfs.append(d)

df = pd.concat(raw_dfs, ignore_index=True)
total_raw = len(df)
print(f"\nTotal raw rows: {total_raw:,}")
print("=" * 70)


def report(label, before, after):
    lost = before - after
    pct = lost / total_raw * 100
    print(f"  {label:45s} | removed {lost:>10,} rows ({pct:5.1f}%) | remaining {after:>10,}")


# Step 1: standardize columns
df = standardize_columns(df)
if 'Name' in df.columns and 'Make' not in df.columns:
    makes, models = _split_name_to_make_model(df['Name'])
    df['Make'] = makes
    df['Model'] = models
if 'Transmission' in df.columns:
    df['Transmission'] = df['Transmission'].astype(str).str.strip().map(_TRANSMISSION_MAP).fillna(df['Transmission'])

_BT_MAP = {'HATCHBACK': 'Hatchback', 'SEDAN': 'Sedan', 'SUV': 'SUV', 'MUV': 'MUV',
           'CROSSOVER': 'Crossover', 'COUPE': 'Coupe', 'CONVERTIBLE': 'Convertible',
           'WAGON': 'Wagon', 'VAN': 'Van', 'PICKUP': 'Pickup'}
if 'BodyType' in df.columns:
    df['BodyType'] = df['BodyType'].astype(str).str.strip().str.upper().map(_BT_MAP).fillna(df['BodyType'])

for must in ['KMs Driven', 'Year', 'Ownership', 'Price (\u20b9)']:
    if must not in df.columns:
        df[must] = np.nan

# Parse fields (can introduce NaN)
n0 = len(df)
df['KMs Driven'] = df['KMs Driven'].apply(_parse_kms)
df['Ownership'] = df['Ownership'].apply(_parse_ownership)
df['Year'] = (
    df['Year'].astype(str).str.replace(r'[^\d]', '', regex=True)
    .replace({'': np.nan}).astype(float)
)
df['Price (\u20b9)'] = (
    df['Price (\u20b9)'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    .replace({'': np.nan}).astype(float)
)

# Check how many NaN each parse step introduced
print("\n--- After parsing (NaN counts that will trigger dropna) ---")
for col in ['Price (\u20b9)', 'Year', 'KMs Driven']:
    nas = df[col].isna().sum()
    print(f"  {col:20s}: {nas:>10,} NaN ({nas/total_raw*100:.1f}%)")

# Timestamps
if 'Fetched On' in df.columns:
    df['Fetched On'] = pd.to_datetime(df['Fetched On'], errors='coerce')
    df['_snapshot_year'] = df['Fetched On'].dt.year.fillna(CURRENT_YEAR).astype(float)
else:
    df['_snapshot_year'] = float(CURRENT_YEAR)

print(f"\n--- Filter-by-filter audit (starting from {n0:,} rows) ---")

# Filter 1: Dedup by ID
before = len(df)
if 'ID' in df.columns and 'Fetched On' in df.columns:
    df.sort_values('Fetched On', inplace=True)
    df_deduped = df.drop_duplicates(subset=['ID'], keep='last')
    report("Dedup by ID (keep last)", before, len(df_deduped))
    df = df_deduped
elif 'ID' in df.columns:
    df_deduped = df.drop_duplicates(subset=['ID'], keep='last')
    report("Dedup by ID (no date)", before, len(df_deduped))
    df = df_deduped
else:
    report("Dedup by ID (no ID col)", before, before)

# Filter 2: dropna on Price, Year, KMs
before = len(df)
mask_na = df[['Price (\u20b9)', 'Year', 'KMs Driven']].isna().any(axis=1)
df = df[~mask_na]
report("dropna(Price, Year, KMs)", before, len(df))

# Show what specifically was NaN
for col in ['Price (\u20b9)', 'Year', 'KMs Driven']:
    n_na = mask_na.sum()  # already counted above

# Filter 3: Year range
before = len(df)
bad_year_low = (df['Year'] < 2005).sum()
bad_year_high = (df['Year'] > df['_snapshot_year']).sum()
df = df[(df['Year'] >= 2005) & (df['Year'] <= df['_snapshot_year'])]
report(f"Year >= 2005 & <= snapshot ({CURRENT_YEAR})", before, len(df))
print(f"    -> Year < 2005: {bad_year_low:,}  |  Year > snapshot: {bad_year_high:,}")

# Filter 4: KMs range
before = len(df)
bad_km_neg = (df['KMs Driven'] < 0).sum()
bad_km_high = (df['KMs Driven'] > 400_000).sum()
df = df[(df['KMs Driven'] >= 0) & (df['KMs Driven'] <= 400_000)]
report("KMs 0-400,000", before, len(df))
print(f"    -> KMs < 0: {bad_km_neg:,}  |  KMs > 400k: {bad_km_high:,}")

# Filter 5: Price percentile
before = len(df)
lo, hi = df['Price (\u20b9)'].quantile([0.01, 0.99])
below = (df['Price (\u20b9)'] < lo).sum()
above = (df['Price (\u20b9)'] > hi).sum()
df = df[(df['Price (\u20b9)'] >= lo) & (df['Price (\u20b9)'] <= hi)]
report(f"Price 1st-99th pctl ({lo:,.0f} - {hi:,.0f})", before, len(df))
print(f"    -> Below {lo:,.0f}: {below:,}  |  Above {hi:,.0f}: {above:,}")

# Summary
print()
print("=" * 70)
print(f"FINAL: {len(df):,} rows from {total_raw:,} raw ({len(df)/total_raw*100:.1f}% retained)")
print(f"LOST:  {total_raw - len(df):,} rows ({(total_raw-len(df))/total_raw*100:.1f}%)")

# Extra: show distribution of years dropped
print("\n--- Year distribution of rows that WOULD be removed by Year<2005 ---")
df_tmp = pd.concat(raw_dfs, ignore_index=True)
df_tmp = standardize_columns(df_tmp)
df_tmp['Year'] = df_tmp['Year'].astype(str).str.replace(r'[^\d]', '', regex=True).replace({'': np.nan}).astype(float)
old_cars = df_tmp[df_tmp['Year'] < 2005]['Year'].value_counts().sort_index()
if len(old_cars) > 0:
    print(old_cars.to_string())
else:
    print("  (none)")
