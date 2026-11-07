"""Oracle importance from supplied models, without fitting or sampling."""

import numpy as np

from ..sampler import Sampler
from ...LUCID_calculation.loss import prediction_loss


class OptimalLUCIDSampler(Sampler):
    """Exact L2 LUCID using a model with conditional prediction moments.

    ``model.conditional_moments(X, feature_idx)`` returns the mean and
    variance of model predictions after replacing that feature conditionally.
    Use ``calculate_batch`` directly: the ordinary LUCID sampling loop requires
    replacement draws and therefore cannot compute this analytic estimator.
    """

    def __init__(self, model, random_state=None):
        super().__init__(random_state)
        self.model = model
        self.is_fitted = True

    def fit(self, X_train=None, feature_idx=None, feature_type=None):
        return self

    def sample(self, *args, **kwargs):
        raise NotImplementedError("Optimal LUCID uses calculate_batch, without sampling.")

    def calculate_batch(self, evaluation_sample, evaluation_target, feature_idx,
                        loss_type="l2"):
        if loss_type not in {"l2", "mse", "squared_error"}:
            raise ValueError("Analytic optimal LUCID supports only L2 loss.")
        X = np.asarray(evaluation_sample, dtype=float)
        y = np.asarray(evaluation_target, dtype=float)
        if X.ndim != 2 or y.shape != (len(X),):
            raise ValueError("Expected a 2D evaluation sample and one target per row.")
        mean, variance = self.model.conditional_moments(X, feature_idx)
        baseline = prediction_loss(self.model.predict(X), y, loss_type)
        return np.asarray(variance) + (np.asarray(mean) - y) ** 2 - baseline


class OptimalLOCOSampler(Sampler):
    """Supply already known reduced models, indexed by omitted feature.

    Each reduced model's ``predict`` receives X with that feature removed,
    matching LOCOSampler and the existing end-to-end LOCO calculation.
    """

    def __init__(self, models, random_state=None):
        super().__init__(random_state)
        self.models = dict(models)
        self.is_fitted = True

    def fit(self, X_train=None, feature_idx=None, feature_type=None, model=None):
        if feature_idx is not None and feature_idx not in self.models:
            raise ValueError("No known reduced model for this feature.")
        return self

    def predict_batch(self, evaluation_sample, feature_idx):
        X = np.asarray(evaluation_sample, dtype=float)
        if X.ndim != 2:
            raise ValueError("evaluation_sample must be a 2D array.")
        if not isinstance(feature_idx, (int, np.integer)) or not 0 <= feature_idx < X.shape[1]:
            raise ValueError("feature_idx is out of range.")
        if feature_idx not in self.models:
            raise ValueError("No known reduced model for this feature.")
        return np.asarray(self.models[feature_idx].predict(np.delete(X, feature_idx, axis=1)))

    def sample(self, *args, **kwargs):
        raise NotImplementedError("Optimal LOCO predicts with a known reduced model.")
