import numpy as np
from ..sampler import Sampler


class PermutationSampler(Sampler):
    """Draw replacements from empirical background marginals."""

    def _fit_feature(self, X_train, feature_idx, feature_type):
        # No model estimation is needed; the base class registers the feature.
        return None

    def _sample_feature(
        self,
        fitted_model,
        X_eval,
        feature_idx,
        n_resamples,
        X_background,
    ):
        X_background = np.asarray(X_background)
        if X_background.ndim != 2:
            raise ValueError("X_background must be a 2D array.")
        if X_background.shape[1] != self.feature_size:
            raise ValueError("X_background has the wrong number of features.")

        return self.rng.choice(
            X_background[:, feature_idx],
            size=(len(X_eval), n_resamples),
            replace=True,
        )