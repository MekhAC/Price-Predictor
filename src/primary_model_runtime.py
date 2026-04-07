from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd

from data_preprocessing import (
    _TRANSMISSION_MAP,
    _parse_kms,
    _parse_ownership,
    _reg_state,
    _split_name_to_make_model,
    add_model_features,
    prepare_model_input,
    preprocessor_uses_market_dynamics,
    standardize_columns,
)
from market_dynamics import add_market_dynamics_features


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def load_primary_artifacts() -> dict[str, Any]:
    meta_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    model_path = os.path.join(MODELS_DIR, 'price_model.joblib')
    dynamics_path = os.path.join(MODELS_DIR, 'market_dynamics_bundle.joblib')

    meta = joblib.load(meta_path)
    required_keys = {'numeric_feats', 'categorical_feats', 'categorical_levels'}
    if meta.get('preprocessor_type') != 'lightgbm_native' or not required_keys.issubset(meta.keys()):
        raise RuntimeError(
            'preprocessor.joblib is not a compatible primary LightGBM preprocessor. '
            'Re-run src/train_model.py to rebuild the primary artifacts.'
        )

    model = joblib.load(model_path)
    dynamics_bundle = None
    if preprocessor_uses_market_dynamics(meta) and os.path.exists(dynamics_path):
        dynamics_bundle = joblib.load(dynamics_path)

    return {
        'meta': meta,
        'model': model,
        'dynamics_bundle': dynamics_bundle,
    }


def build_option_map(meta: dict) -> dict[str, list[str]]:
    categorical_levels = meta.get('categorical_levels', {})
    option_map = {}
    for column in ['Make', 'Model', 'Variant', 'Transmission', 'Fuel', 'BodyType', 'City', 'Reg State']:
        values = list(categorical_levels.get(column, []))
        option_map[column] = [value for value in values if value != '__OTHER__']
    return option_map


def _ensure_kms_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'KMs Driven' in df.columns:
        return df

    aliases = ['kms', 'kmsdriven', 'km driven', 'kms driven', 'odometer']
    normalized = {str(c).strip().lower().replace('_', ' ').replace('-', ' '): c for c in df.columns}
    for alias in aliases:
        if alias in normalized:
            df['KMs Driven'] = df[normalized[alias]]
            return df

    df['KMs Driven'] = np.nan
    return df


def _normalize_runtime_input(df: pd.DataFrame) -> pd.DataFrame:
    work = standardize_columns(df.copy())
    work = _ensure_kms_column(work)

    if 'Name' in work.columns and 'Make' not in work.columns:
        makes, models = _split_name_to_make_model(work['Name'])
        work['Make'] = makes
        work['Model'] = models

    if 'Transmission' in work.columns:
        work['Transmission'] = (
            work['Transmission']
            .astype(str)
            .str.strip()
            .map(_TRANSMISSION_MAP)
            .fillna(work['Transmission'])
        )

    for column in ['Make', 'Model', 'Variant', 'Transmission', 'Fuel', 'BodyType', 'City']:
        if column not in work.columns:
            work[column] = 'Unknown'

    if 'Year' not in work.columns:
        work['Year'] = np.nan
    if 'Ownership' not in work.columns:
        work['Ownership'] = np.nan
    if 'Registration' not in work.columns:
        work['Registration'] = ''

    work['Year'] = (
        work['Year']
        .astype(str)
        .str.replace(r'[^\d]', '', regex=True)
        .replace({'': np.nan})
        .astype(float)
    )
    work['KMs Driven'] = work['KMs Driven'].apply(_parse_kms)
    work['Ownership'] = pd.to_numeric(work['Ownership'].apply(_parse_ownership), errors='coerce')

    if 'Reg State' not in work.columns:
        work['Reg State'] = work['Registration'].apply(_reg_state) if 'Registration' in work.columns else 'UNK'
    else:
        fill_mask = work['Reg State'].isna() | (work['Reg State'].astype(str).str.strip() == '')
        if 'Registration' in work.columns:
            work.loc[fill_mask, 'Reg State'] = work.loc[fill_mask, 'Registration'].apply(_reg_state)
        work['Reg State'] = work['Reg State'].fillna('UNK').astype(str).str.strip().replace({'': 'UNK'})

    return work


def transform_primary_rows(df: pd.DataFrame, artifacts: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = _normalize_runtime_input(df)
    work = add_model_features(work)

    dynamics_bundle = artifacts.get('dynamics_bundle')
    if dynamics_bundle is not None:
        work = add_market_dynamics_features(work, dynamics_bundle)

    X_proc = prepare_model_input(work, artifacts['meta'])
    return work, X_proc


def predict_primary_rows(df: pd.DataFrame, artifacts: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    enriched, X_proc = transform_primary_rows(df, artifacts)
    pred = np.asarray(artifacts['model'].predict(X_proc), dtype=float).ravel()
    return enriched, pred
