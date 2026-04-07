"""One-off data diagnosis script — safe to delete after use."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from data_preprocessing import load_and_clean
import glob

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
files = (
    glob.glob(os.path.join(DATA_DIR, 'normalized_table.*')) +
    glob.glob(os.path.join(DATA_DIR, 'normalized_table_*.*')) +
    glob.glob(os.path.join(DATA_DIR, 'Cars24_*.*')) +
    glob.glob(os.path.join(DATA_DIR, 'Spinny_*.*'))
)
df = pd.concat([load_and_clean(f) for f in files], ignore_index=True)
PRICE = 'Price (\u20b9)'

print('=' * 60)
print(f'DATASET SIZE: {len(df):,} rows')
print()

# 1. Target distribution
print('=== PRICE DISTRIBUTION ===')
print(df[PRICE].describe().to_string())
print(f'Skewness: {df[PRICE].skew():.2f}')
print(f'Kurtosis: {df[PRICE].kurt():.2f}')
print()

# 2. Categorical cardinality
print('=== CATEGORICAL CARDINALITY ===')
for c in ['Make','Model','Variant','Transmission','Fuel','BodyType','Reg State']:
    if c not in df.columns:
        continue
    n = df[c].nunique()
    top = df[c].value_counts().head(1)
    tail_pct = (df[c].value_counts() < 10).sum() / n * 100
    print(f'{c:15s}: {n:5d} unique  |  <10 samples: {tail_pct:.0f}%  |  top: {top.index[0]} ({top.iloc[0]})')
print()

# 3. Variant sparsity
vc = df['Variant'].value_counts()
print('=== VARIANT FREQUENCY DISTRIBUTION ===')
for thresh in [1, 2, 5, 10, 20, 50]:
    n = (vc <= thresh).sum()
    print(f'  Variants with <= {thresh:2d} samples: {n:5d} ({n/len(vc)*100:.1f}%)')
print()

# 4. Intra-group price variance (Make+Model+Year+Fuel+Transmission)
print('=== INTRA-GROUP PRICE VARIANCE (Make+Model+Year+F;uel+Trans) ===')
grp = df.groupby(['Make','Model','Year','Fuel','Transmission'])[PRICE]
stats = grp.agg(['count','mean','std','min','max']).dropna()
stats['cv'] = stats['std'] / stats['mean'] * 100
stats = stats[stats['count'] >= 5]
print(f'Groups with 5+ cars: {len(stats)}')
print(f'Median CV: {stats["cv"].median():.1f}%')
print(f'Mean CV:   {stats["cv"].mean():.1f}%')
print(f'75th pctl: {stats["cv"].quantile(0.75):.1f}%')
print(f'90th pctl: {stats["cv"].quantile(0.90):.1f}%')
print()
print('Worst 10 groups (highest price spread):')
for idx, row in stats.nlargest(10, 'cv').iterrows():
    make, model, yr, fuel, trans = idx
    print(f'  {make} {model} {yr:.0f} {fuel} {trans}: n={row["count"]:.0f}  mean={row["mean"]:,.0f}  CV={row["cv"]:.0f}%  range={row["min"]:,.0f}-{row["max"]:,.0f}')
print()

# 5. Including Variant
print('=== INTRA-GROUP PRICE VARIANCE (incl. Variant) ===')
grp2 = df.groupby(['Make','Model','Variant','Year','Fuel','Transmission'])[PRICE]
stats2 = grp2.agg(['count','mean','std','min','max']).dropna()
stats2['cv'] = stats2['std'] / stats2['mean'] * 100
stats2 = stats2[stats2['count'] >= 5]
print(f'Groups with 5+ cars: {len(stats2)}')
if len(stats2) > 0:
    print(f'Median CV: {stats2["cv"].median():.1f}%')
    print(f'Mean CV:   {stats2["cv"].mean():.1f}%')
print()

# 6. Feature correlations with price
print('=== FEATURE-PRICE CORRELATIONS ===')
for feat in ['Year','KMs Driven','Car Age','KMs/Year','Ownership','Log KMs']:
    if feat in df.columns:
        corr = df[feat].corr(df[PRICE])
        print(f'  {feat:18s} vs Price: r = {corr:+.3f}')
print()

# 7. Ownership distribution
print('=== OWNERSHIP DISTRIBUTION ===')
print(df['Ownership'].value_counts().sort_index().to_string())
print()

# 8. Year distribution
print('=== YEAR DISTRIBUTION ===')
print(df['Year'].value_counts().sort_index().to_string())
print()

# 9. Missing / Unknown rates
print('=== MISSING / UNKNOWN RATES ===')
for c in ['Make','Model','Variant','Transmission','Fuel','BodyType','Reg State']:
    if c in df.columns:
        unk = ((df[c] == 'Unknown') | (df[c] == 'UNK') | (df[c].isna())).sum()
        print(f'  {c:15s}: {unk:6d} ({unk/len(df)*100:.1f}%)')
print()

# 10. Near-duplicate detection
print('=== POTENTIAL NOISE: SAME CAR, DIFFERENT PRICES ===')
dup_cols = ['Make','Model','Variant','Year','KMs Driven','Fuel','Transmission','Ownership']
dups = df[df.duplicated(subset=dup_cols, keep=False)].copy()
if len(dups) > 0:
    grp3 = dups.groupby(dup_cols)[PRICE].agg(['count','min','max','mean','std'])
    grp3['spread_pct'] = (grp3['max'] - grp3['min']) / grp3['mean'] * 100
    grp3 = grp3[grp3['count'] >= 2].sort_values('spread_pct', ascending=False)
    print(f'Duplicate rows: {len(dups)} | Duplicate groups: {len(grp3)}')
    print(f'Median price spread within duplicates: {grp3["spread_pct"].median():.1f}%')
    print(f'Mean price spread: {grp3["spread_pct"].mean():.1f}%')
    big = grp3[grp3['spread_pct'] > 30]
    print(f'Groups with >30% price spread: {len(big)} ({len(big)/len(grp3)*100:.1f}%)')
else:
    print('No exact duplicates found')
print()

# 11. City / source price variation
if 'City' in df.columns:
    print('=== CITY EFFECT ===')
    city_stats = df.groupby('City')[PRICE].agg(['count','mean'])
    city_stats = city_stats[city_stats['count'] >= 50].sort_values('mean', ascending=False)
    print(f'Cities with 50+ listings: {len(city_stats)}')
    if len(city_stats) > 0:
        print(f'Price range across cities: {city_stats["mean"].min():,.0f} - {city_stats["mean"].max():,.0f}')
        print(city_stats.head(10).to_string())
    print()

# 12. Data source overlap check
print('=== FILE CONTRIBUTIONS ===')
for f in files:
    tmp = load_and_clean(f)
    print(f'  {os.path.basename(f):40s}: {len(tmp):6,} rows  |  price range: {tmp[PRICE].min():,.0f} - {tmp[PRICE].max():,.0f}')
