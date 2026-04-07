"""
Ensemble model wrapper.

Stores multiple base models (each trained on log1p(price)) and averages their
predictions in original price space.  Exposes a single `.predict(X)` method so
it's a drop-in replacement for the previous single-model joblib artifact.
"""
import numpy as np
import pandas as pd


class EnsembleModel:
    def __init__(self, models, cat_features=None, weights=None, segment_specialists=None):
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
        segment_specialists : list[dict] or None
            Optional routing specialists. Each item must define:
            `column`, `value`, `model_name`, `model`, and `blend_weight`.
            The specialist prediction is blended with the global prediction only
            on rows where `X[column] == value`.
        """
        self.models = models
        self.cat_features = cat_features or []
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
        self.segment_specialists = segment_specialists or []

    def _predict_submodel(self, name, model, X):
        if name == 'catboost':
            Xc = X.copy()
            for col in self.cat_features:
                if col in Xc.columns:
                    Xc[col] = Xc[col].astype(str)
            pred_log = model.predict(Xc)
        else:
            pred_log = model.predict(X)
        return np.expm1(np.asarray(pred_log, dtype=float).ravel())

    @staticmethod
    def _segment_mask(X, column, value):
        if column not in X.columns:
            return np.zeros(len(X), dtype=bool)
        values = pd.Series(X[column], index=X.index).fillna('').astype(str).str.casefold()
        return (values == str(value).casefold()).to_numpy(dtype=bool)

    def predict(self, X):
        preds = []
        for name, model in self.models:
            preds.append(self._predict_submodel(name, model, X))
        global_pred = np.average(preds, axis=0, weights=self.weights)

        segment_specialists = getattr(self, 'segment_specialists', [])
        if not segment_specialists:
            return global_pred

        weighted_sum = global_pred.astype(float).copy()
        total_weight = np.ones(len(global_pred), dtype=float)

        for specialist in segment_specialists:
            blend_weight = float(specialist.get('blend_weight', 0.0) or 0.0)
            if blend_weight <= 0:
                continue

            mask = self._segment_mask(X, specialist.get('column', ''), specialist.get('value', ''))
            if not mask.any():
                continue

            specialist_pred = self._predict_submodel(
                specialist['model_name'],
                specialist['model'],
                X.loc[mask],
            )
            weighted_sum[mask] += blend_weight * specialist_pred
            total_weight[mask] += blend_weight

        return weighted_sum / np.clip(total_weight, 1e-9, None)

    def __repr__(self):
        names = [n for n, _ in self.models]
        specialist_count = len(getattr(self, 'segment_specialists', []))
        return (
            f"EnsembleModel({names}, weights={[round(w, 3) for w in self.weights]}, "
            f"segment_specialists={specialist_count})"
        )
