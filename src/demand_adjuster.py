# demand_adjuster.py
import os
import math
import yaml
import csv

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def _load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def _load_multipliers_csv(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            # Normalize blanks -> None
            clean = {}
            for k, v in r.items():
                if isinstance(v, str):
                    v = v.strip()
                    if v == "": v = None
                clean[k] = v
            # Cast numerics
            for k in ('min', 'max', 'multiplier'):
                if clean.get(k) is not None:
                    try:
                        clean[k] = float(clean[k])
                    except Exception:
                        # tolerate stray text (e.g., quotes) in min/max cells
                        clean[k] = None
            rows.append(clean)
    return rows

def _norm(s):
    return str(s).strip().lower() if s is not None else None

def _split_variant_parent(parent):
    """
    Parent for variant rows can be 'Make | Model' or just 'Make'.
    Return (make_norm, model_norm_or_None).
    """
    if parent is None:
        return None, None
    parts = [p.strip() for p in str(parent).split('|')]
    if len(parts) == 1:
        return _norm(parts[0]), None
    return _norm(parts[0]), _norm(parts[1])

def _find_row(rows, **conds):
    """Case/space-insensitive equality on provided fields."""
    for r in rows:
        ok = True
        for k, v in conds.items():
            rv = r.get(k)
            if _norm(rv) != _norm(v):
                ok = False
                break
        if ok:
            return r
    return None

def _find_variant_row(rows, name, make, model):
    """
    Match a variant row with:
      type='variant', name=<variant>, parent either 'Make | Model' or 'Make'
    Prefer exact Make+Model parent match; else accept Make-only parent.
    """
    name_n = _norm(name)
    make_n = _norm(make)
    model_n = _norm(model)

    candidates = [r for r in rows if _norm(r.get('type')) == 'variant' and _norm(r.get('name')) == name_n]
    best = None
    for r in candidates:
        pmake, pmodel = _split_variant_parent(r.get('parent'))
        if pmake == make_n and pmodel == model_n:
            return r  # perfect match
        if pmake == make_n and pmodel is None:
            best = r  # fallback (make-only)
    return best

def _find_bucket(rows, t, value):
    # inclusive min, exclusive max
    t_norm = _norm(t)
    buckets = [r for r in rows if _norm(r.get('type')) == t_norm]
    for r in buckets:
        mn = r.get('min'); mx = r.get('max')
        if mn is None or mx is None:
            continue
        try:
            if float(value) >= float(mn) and float(value) < float(mx):
                return r
        except Exception:
            continue
    return None

class DemandAdjuster:
    def __init__(self,
                 feature_priority_path=os.path.join(MODELS_DIR, 'feature_priority.yaml'),
                 multipliers_csv_path=os.path.join(MODELS_DIR, 'demand_multipliers.csv')):
        # defaults so it still runs if files are missing
        self.cfg = {'weights': {}, 'blending': 'weighted_geometric_mean', 'clamp': {'min': 0.7, 'max': 1.2}}
        self.rows = []
        if os.path.exists(feature_priority_path):
            self.cfg = _load_yaml(feature_priority_path) or self.cfg
        if os.path.exists(multipliers_csv_path):
            self.rows = _load_multipliers_csv(multipliers_csv_path)
        self.weights = self.cfg.get('weights', {})
        self.blending = self.cfg.get('blending', 'weighted_geometric_mean')
        clamp = self.cfg.get('clamp', {})
        self.clamp_min = clamp.get('min', 0.7)
        self.clamp_max = clamp.get('max', 1.2)

    def compute(self, *, make, model, variant, city, bodytype, fuel, transmission,
                car_age, kms_per_year, ownership, reg_state=None):
        """
        Returns (composite_multiplier, breakdown)
        Breakdown = [(name, multiplier, normalized_weight), ...]
        Weight keys expected in feature_priority.yaml:
            variant, model, make, city, bodytype, fuel, transmission,
            car_age, kms_per_year, ownership, reg_state
        """
        parts = []  # (feature_key, multiplier_value, weight_raw)

        # Variant preferred → then Model → then Brand
        var_row = None
        if variant not in (None, '', 'Unknown'):
            var_row = _find_variant_row(self.rows, variant, make, model)
        if var_row:
            parts.append(('variant', var_row['multiplier'], self.weights.get('variant', 0.0)))
        else:
            mod_row = _find_row(self.rows, type='model', name=model, parent=make)
            if mod_row:
                parts.append(('model', mod_row['multiplier'], self.weights.get('model', 0.0)))
            else:
                brand_row = _find_row(self.rows, type='brand', name=make)
                if brand_row:
                    parts.append(('make', brand_row['multiplier'], self.weights.get('make', 0.0)))

        # City
        city_row = _find_row(self.rows, type='city', name=city)
        if city_row:
            parts.append(('city', city_row['multiplier'], self.weights.get('city', 0.0)))

        # BodyType / Fuel / Transmission
        for typ, val, key in (
            ('bodytype', bodytype, 'bodytype'),
            ('fuel',     fuel,     'fuel'),
            ('transmission', transmission, 'transmission'),
        ):
            row = _find_row(self.rows, type=typ, name=val)
            if row:
                parts.append((key, row['multiplier'], self.weights.get(key, 0.0)))

        # Buckets: age, kms/year, ownership
        age_row = _find_bucket(self.rows, 'age_bucket', float(car_age))
        if age_row:
            parts.append(('car_age', age_row['multiplier'], self.weights.get('car_age', 0.0)))

        kms_row = _find_bucket(self.rows, 'kms_year_bucket', float(kms_per_year))
        if kms_row:
            parts.append(('kms_per_year', kms_row['multiplier'], self.weights.get('kms_per_year', 0.0)))

        own_row = _find_bucket(self.rows, 'ownership_bucket', float(ownership))
        if own_row:
            parts.append(('ownership', own_row['multiplier'], self.weights.get('ownership', 0.0)))

        # Optional reg_state
        if reg_state:
            rs_row = _find_row(self.rows, type='reg_state', name=reg_state)
            if rs_row:
                parts.append(('reg_state', rs_row['multiplier'], self.weights.get('reg_state', 0.0)))

        # Keep only positive-weighted, positive multipliers
        parts = [(n, m, w) for (n, m, w) in parts if (w or 0) > 0 and (m or 0) > 0]
        if not parts:
            return 1.0, []  # neutral

        # Normalize weights among used parts
        w_sum = sum(w for (_, _, w) in parts)
        parts_norm = [(n, m, (w / w_sum) if w_sum > 0 else 0) for (n, m, w) in parts]

        log_sum = 0.0
        breakdown = []
        for name, mult, w in parts_norm:
            eps = 1e-9
            log_sum += w * math.log(max(mult, eps))
            breakdown.append((name, mult, w))

        comp = math.exp(log_sum)
        comp = max(self.clamp_min, min(self.clamp_max, comp))
        return comp, breakdown
