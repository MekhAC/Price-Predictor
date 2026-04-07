from __future__ import annotations

import argparse

import pandas as pd

from primary_model_runtime import load_primary_artifacts, predict_primary_rows


def predict(args: argparse.Namespace) -> None:
    artifacts = load_primary_artifacts()
    df = pd.DataFrame(
        [
            {
                'Make': args.make,
                'Model': args.model,
                'Variant': args.variant,
                'Transmission': args.transmission,
                'Fuel': args.fuel,
                'BodyType': args.bodytype,
                'City': getattr(args, 'city', 'Unknown') or 'Unknown',
                'Year': args.year,
                'KMs Driven': args.kms,
                'Ownership': args.ownership,
                'Registration': args.registration,
                'Reg State': args.regstate or '',
            }
        ]
    )

    enriched, pred = predict_primary_rows(df, artifacts)
    if enriched.empty or len(pred) == 0:
        raise RuntimeError('No valid rows available for primary-model prediction.')

    predicted_price = float(pred[0])
    car_age = float(enriched.iloc[0].get('Car Age', 0.0))
    kms_per_year = float(enriched.iloc[0].get('KMs/Year', 0.0))
    expected_drop_30d = float(enriched.iloc[0].get('Expected Avg Drop 30d', 0.0))
    expected_sell_time = float(enriched.iloc[0].get('Expected Time To Sell', 0.0))
    liquidity = float(enriched.iloc[0].get('Expected Market Liquidity Score', 0.0))

    print(f"\nPredicted price: Rs.{predicted_price:,.0f}")
    print(
        "Derived features: "
        f"Car Age={car_age:.1f}y, "
        f"KMs/Year={kms_per_year:,.0f}, "
        f"Expected Avg Drop 30d={expected_drop_30d:.4f}, "
        f"Expected Time To Sell={expected_sell_time:.1f}d, "
        f"Liquidity Score={liquidity:.1f}\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', default='Unknown', help='City (optional)')
    parser.add_argument('--make', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--transmission', choices=['Manual', 'Automatic'], required=True)
    parser.add_argument('--fuel', required=True)
    parser.add_argument('--bodytype', required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--ownership', type=int, required=True)
    parser.add_argument('--kms', type=float, required=True)
    parser.add_argument('--registration', default='', help='e.g., KA51**9597 (optional)')
    parser.add_argument('--regstate', default='', help='Override state code like KA (optional)')
    return parser


if __name__ == "__main__":
    predict(build_parser().parse_args())
