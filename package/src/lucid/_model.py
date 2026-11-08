import numpy as np


class ModelAdapter:
    """Normalize a fitted estimator or callable to scalar predictions."""

    def __init__(self, model):
        if callable(getattr(model, "predict", None)):
            self._predict = model.predict
        elif callable(model):
            self._predict = model
        else:
            raise TypeError(
                "model must have a predict method or be callable."
            )

    def __call__(self, X):
        return np.asarray(self._predict(X), dtype=float)