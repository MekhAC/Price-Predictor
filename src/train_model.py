import os, glob
import numpy as np
import joblib
import pandas as pd
from math import sqrt
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor

from data_preprocessing import load_and_clean, build_preprocessor
from demand_adjuster import DemandAdjuster

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

files = (
    glob.glob(os.path.join(DATA_DIR, 'Cars24_*.*')) +
    glob.glob(os.path.join(DATA_DIR, 'Spinny_*.*'))
)
if not files:
    raise RuntimeError(f"No data files found in {DATA_DIR}")

print(f"🔍 Training on {len(files)} files…")
df = pd.concat([load_and_clean(f) for f in files], ignore_index=True)

pre_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
if os.path.exists(pre_path):
    meta = joblib.load(pre_path)
    pre = meta['preprocessor']
    numeric_feats = meta['numeric_feats']
    categorical_feats = meta['categorical_feats']
else:
    pre = build_preprocessor(df)  # also saves
    meta = joblib.load(pre_path)
    numeric_feats = meta['numeric_feats']
    categorical_feats = meta['categorical_feats']

FEATURES = numeric_feats + categorical_feats
X = df[FEATURES]
y = df['Price (₹)'].astype(float)

X_proc = pre.transform(X)
if sparse.issparse(X_proc):
    X_proc = X_proc.tocsr()

# ---- Build sample weights per-row (same order as df) ----
adj = DemandAdjuster()
# sw = np.zeros(len(df), dtype=float)
for i, r in df.iterrows():
    car_age = max(2025 - float(r['Year']), 0.0)
    kms_year = (float(r['KMs Driven']) / car_age) if car_age > 0 else float(r['KMs Driven'])
    # comp, _ = adj.compute(
    #     make=r['Make'], model=r['Model'], city=r['City'], bodytype=r['BodyType'],
    #     fuel=r['Fuel'], transmission=r['Transmission'], car_age=car_age,
    #     kms_per_year=kms_year, ownership=r.get('Ownership', 1) or 1,
    #     reg_state=r.get('Reg State', None)
    # )
    # sw[i] = comp

# ---- Split indices first to keep alignment for X, y, sw ----
idx = np.arange(X_proc.shape[0])
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

X_train, X_test = X_proc[idx_train], X_proc[idx_test]
y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
# sw_train = sw[idx_train]

# ---- Model ----
base_rf = RandomForestRegressor(
    n_estimators=350,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
model = TransformedTargetRegressor(
    regressor=base_rf, func=np.log1p, inverse_func=np.expm1
)

# model.fit(X_train, y_train, regressor__sample_weight=sw_train)
model.fit(X_train, y_train)

# ---- Eval ----
y_pred = model.predict(X_test)
mse  = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2   = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print(f"🔎 MSE: {mse:,.0f} ₹")
print(f"🔎 RMSE: {rmse:,.0f} ₹")
print(f"🔎 R²  : {r2:.4f}")
print(f"🔎 MAPE: {mape:.2f}%")

out_model = os.path.join(MODELS_DIR, 'price_model.joblib')
joblib.dump(model, out_model)
print(f"✅ Model saved to {out_model}")
