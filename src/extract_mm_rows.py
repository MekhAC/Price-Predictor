import os, glob, re, argparse
import pandas as pd
from data_preprocessing import standardize_columns, _split_name_to_make_model

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR  = os.path.join(BASE_DIR, 'models')  # keep with other artifacts
os.makedirs(OUT_DIR, exist_ok=True)

def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    return pd.read_csv(path)

def _clean_text(x: str) -> str:
    if pd.isna(x): return 'Unknown'
    s = str(x).strip()
    s = re.sub(r'\s+', ' ', s)
    return s if s else 'Unknown'

def load_minimal(path: str) -> pd.DataFrame:
    df = standardize_columns(_read_any(path))
    # Split combined Name column into Make / Model if separate columns are absent.
    if 'Name' in df.columns and 'Make' not in df.columns:
        makes, models = _split_name_to_make_model(df['Name'])
        df['Make'] = makes
        df['Model'] = models
    for c in ['Make','Model','Variant']:
        if c not in df.columns:
            df[c] = 'Unknown'
        df[c] = df[c].map(_clean_text)
    return df[['Make','Model','Variant']]

def write_csv(path: str, rows: pd.DataFrame):
    # ensure correct column order
    cols = ['type','scope','name','parent','min','max','multiplier','notes']
    rows.reindex(columns=cols).to_csv(path, index=False, encoding='utf-8-sig')
    print(f"✅ Wrote {len(rows)} rows → {path}")

def main(include_unknown: bool, only: str):
    files = (
        glob.glob(os.path.join(DATA_DIR, 'normalized_table.*')) +
        glob.glob(os.path.join(DATA_DIR, 'normalized_table_*.*')) +
        glob.glob(os.path.join(DATA_DIR, 'Cars24_*.*')) +
        glob.glob(os.path.join(DATA_DIR, 'Spinny_*.*'))
    )
    if not files:
        raise RuntimeError(f"No data files found in {DATA_DIR}.")

    parts = [load_minimal(p) for p in files]
    df = pd.concat(parts, ignore_index=True)

    if not include_unknown:
        df = df[(df['Make']   != 'Unknown') &
                (df['Model']  != 'Unknown') &
                (df['Variant']!= 'Unknown')]

    # Unique brands
    brands = (
        df[['Make']].drop_duplicates().rename(columns={'Make':'name'})
        .assign(type='brand', scope='', parent='', min='', max='', multiplier='', notes='')
        [['type','scope','name','parent','min','max','multiplier','notes']]
        .sort_values('name')
        .reset_index(drop=True)
    )

    # Unique (brand, model) → demand CSV “model” rows (name=model, parent=brand)
    models = (
        df[['Make','Model']].drop_duplicates()
        .rename(columns={'Make':'parent','Model':'name'})
        .assign(type='model', scope='', min='', max='', multiplier='', notes='')
        [['type','scope','name','parent','min','max','multiplier','notes']]
        .sort_values(['parent','name'])
        .reset_index(drop=True)
    )

    # Unique (brand, model, variant) → “variant” rows (not used by your adjuster yet,
    # but this gives you a clean list to curate/merge later if you decide to weight variants)
    variants = (
        df[['Make','Model','Variant']].drop_duplicates()
        .rename(columns={'Make':'brand','Model':'model','Variant':'name'})
        .assign(type='variant', scope='', parent=lambda x: x['brand'] + ' | ' + x['model'],
                min='', max='', multiplier='', notes='')
        [['type','scope','name','parent','min','max','multiplier','notes']]
        .sort_values(['parent','name'])
        .reset_index(drop=True)
    )

    # Write files (respect --only flag)
    if only in ('', 'all', 'brands'):
        write_csv(os.path.join(OUT_DIR, 'dm_brands.csv'), brands)
    if only in ('', 'all', 'models'):
        write_csv(os.path.join(OUT_DIR, 'dm_models.csv'), models)
    if only in ('', 'all', 'variants'):
        write_csv(os.path.join(OUT_DIR, 'dm_variants.csv'), variants)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract unique Make–Model–Variant combos formatted for demand_multipliers.csv.")
    ap.add_argument('--include_unknown', action='store_true', help='Include Unknown values')
    ap.add_argument('--only', choices=['brands','models','variants','all'], default='all',
                    help='Which list(s) to produce')
    args = ap.parse_args()
    main(include_unknown=args.include_unknown, only=args.only)
