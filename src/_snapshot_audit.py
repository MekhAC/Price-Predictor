"""Quick snapshot analysis - checks if duplicate IDs have different field values."""
import os, sys, glob
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import standardize_columns

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
files = glob.glob(os.path.join(DATA_DIR, 'normalized_table_*.*'))

# Read just the first file to keep it fast
print(f"Reading {os.path.basename(files[0])} (1 of {len(files)})...")
if files[0].lower().endswith(('.xlsx', '.xls')):
    df = pd.read_excel(files[0])
else:
    df = pd.read_csv(files[0])

df = standardize_columns(df)
print(f"Rows: {len(df):,}  |  Unique IDs: {df['ID'].nunique():,}")
print(f"Avg snapshots per ID: {len(df) / df['ID'].nunique():.1f}")
print()

# For each column, count how many IDs have >1 unique value across snapshots
cols_to_check = [c for c in df.columns if c not in ('ID', 'Fetched On', 'Image')]
print(f"{'Column':25s} | {'IDs w/ >1 value':>15s} | {'% of multi-snap IDs':>20s} | Sample changes")
print("-" * 100)

multi_ids = df.groupby('ID').filter(lambda g: len(g) > 1)['ID'].unique()
n_multi = len(multi_ids)

for col in cols_to_check:
    nuniq = df.groupby('ID')[col].nunique()
    changed = (nuniq > 1).sum()
    pct = changed / n_multi * 100 if n_multi > 0 else 0

    # Show a sample of what changed
    sample_str = ""
    if changed > 0:
        changed_ids = nuniq[nuniq > 1].index[:3]
        samples = []
        for cid in changed_ids:
            vals = df.loc[df['ID'] == cid, col].unique()
            if len(vals) > 5:
                vals = vals[:5]
            samples.append(f"{list(vals)}")
        sample_str = " | ".join(samples[:2])
        if len(sample_str) > 50:
            sample_str = sample_str[:50] + "..."

    print(f"{col:25s} | {changed:>15,} | {pct:>19.1f}% | {sample_str}")

# Price-specific deep dive
print()
print("=" * 70)
print("PRICE DEEP DIVE (same file)")
df['Price_clean'] = df['Price (\u20b9)'].astype(str).str.replace(r'[^\d.]', '', regex=True).replace({'': np.nan}).astype(float)
grp = df.groupby('ID')['Price_clean'].agg(['count', 'nunique', 'min', 'max', 'mean'])
multi = grp[grp['count'] > 1].copy()
multi['spread_pct'] = ((multi['max'] - multi['min']) / multi['mean'] * 100).fillna(0)

print(f"IDs with >1 snapshot: {len(multi):,}")
print(f"  Price NEVER changed: {(multi['nunique'] == 1).sum():,} ({(multi['nunique'] == 1).sum()/len(multi)*100:.1f}%)")
print(f"  Price changed:       {(multi['nunique'] > 1).sum():,} ({(multi['nunique'] > 1).sum()/len(multi)*100:.1f}%)")

changed = multi[multi['nunique'] > 1]
if len(changed) > 0:
    print(f"  Median spread:       {changed['spread_pct'].median():.1f}%")
    print(f"  Mean spread:         {changed['spread_pct'].mean():.1f}%")
