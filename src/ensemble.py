"""
Ensemble model wrapper.

Stores multiple base models (each trained on log1p(price)) and averages their
predictions in original price space.  Exposes a single `.predict(X)` method so
it's a drop-in replacement for the previous single-model joblib artifact.
"""
import numpy as np


class EnsembleModel:
    def __init__(self, models, cat_features=None, weights=None):
        """
        Parameters
        ----------
        models : list of (name, fitted_model) tuples
            Each model must have been trained on log1p(y).
            *name* is one of 'lightgbm', 'xgboost', 'catboost'.
        cat_features : list[str]
            Categorical column names (needed for CatBoost string conversion).
        weights : list[float] or None
            Per-model weights (sum to 1). If None, equal weights are used.
        """
        self.models = models
        self.cat_features = cat_features or []
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights

    def predict(self, X):
        preds = []
        for name, model in self.models:
            if name == 'catboost':
                Xc = X.copy()
                for col in self.cat_features:
                    if col in Xc.columns:
                        Xc[col] = Xc[col].astype(str)
                pred_log = model.predict(Xc)
            else:
                pred_log = model.predict(X)
            preds.append(np.expm1(np.asarray(pred_log, dtype=float).ravel()))
        return np.average(preds, axis=0, weights=self.weights)

    def __repr__(self):
        names = [n for n, _ in self.models]
        return f"EnsembleModel({names}, weights={[round(w, 3) for w in self.weights]})"
