import os, argparse
import joblib
import pandas as pd
from scipy import sparse
from demand_adjuster import DemandAdjuster

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def predict(args):
    meta = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    pre  = meta['preprocessor']
    num  = meta['numeric_feats']
    cat  = meta['categorical_feats']

    df = pd.DataFrame([{
        'City': args.city,
        'Make': args.make,
        'Model': args.model,
        'Variant': args.variant,
        'Transmission': args.transmission,
        'Fuel': args.fuel,
        'BodyType': args.bodytype,
        'Year': args.year,
        'KMs Driven': args.kms,
        'Ownership': args.ownership,
        'Reg State': args.regstate or (args.registration[:2].upper() if args.registration else 'UNK'),
    }])

    # Derivations
    df['Car Age'] = max(2025 - df.at[0, 'Year'], 0)
    df['KMs/Year'] = (df.at[0, 'KMs Driven'] / df.at[0, 'Car Age']) if df.at[0, 'Car Age'] > 0 else df.at[0, 'KMs Driven']

    X = df[num + cat]
    Xp = pre.transform(X)
    if sparse.issparse(Xp):
        Xp = Xp.tocsr()

    mdl = joblib.load(os.path.join(MODELS_DIR, 'price_model.joblib'))
    base = float(mdl.predict(Xp)[0])

    # Demand adjustment (now includes variant)
    adj = DemandAdjuster()
    composite, breakdown = adj.compute(
        make=df.at[0, 'Make'],
        model=df.at[0, 'Model'],
        variant=df.at[0, 'Variant'],
        city=df.at[0, 'City'],
        bodytype=df.at[0, 'BodyType'],
        fuel=df.at[0, 'Fuel'],
        transmission=df.at[0, 'Transmission'],
        car_age=float(df.at[0, 'Car Age']),
        kms_per_year=float(df.at[0, 'KMs/Year']),
        ownership=float(df.at[0, 'Ownership'] or 1),
        reg_state=df.at[0, 'Reg State'] if df.at[0, 'Reg State'] not in (None, '', 'UNK') else None
    )

    adj_price = base * composite
    lo, hi = adj_price * 0.70, adj_price

    print("\n--- Demand adjustment ---")
    for name, mult, w in breakdown:
        print(f"{name:>14}: ×{mult:.2f} (weight {w:.2f})")
    print(f"{'composite':>14}: ×{composite:.3f}")

    print(f"\nPredicted price: ₹{adj_price:,.0f}")
    print(f"Confidence band: ₹{lo:,.0f} – ₹{hi:,.0f}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--city',        required=True)
    p.add_argument('--make',        required=True)
    p.add_argument('--model',       required=True)
    p.add_argument('--variant',     required=True)
    p.add_argument('--transmission',choices=['Manual','Automatic'], required=True)
    p.add_argument('--fuel',        required=True)
    p.add_argument('--bodytype',    required=True)
    p.add_argument('--year',        type=int,   required=True)
    p.add_argument('--ownership',   type=int,   required=True)
    p.add_argument('--kms',         type=float, required=True)
    p.add_argument('--registration',default='', help='e.g., KA51**9597 (optional)')
    p.add_argument('--regstate',    default='', help='Override state code like KA (optional)')
    args = p.parse_args()
    predict(args)
