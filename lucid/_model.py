import numpy as np


class ModelAdapter:
    """Normalize a fitted estimator or callable to scalar predictions."""

    def __init__(self, model):
        # Prefer an estimator's predict method; otherwise use the model itself as a
        # prediction callable.
        if callable(getattr(model, "predict", None)):
            self._predict = model.predict
        elif callable(model):
            self._predict = model
        else:
            raise TypeError(
                "model must have a predict method or be callable."
            )

    def __call__(self, X):
        # Convert prediction output for the explainer's scalar-per-observation handling.
        return np.asarray(self._predict(X), dtype=float)