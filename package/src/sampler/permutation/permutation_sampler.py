import numpy as np
from ..sampler import Sampler

class PermutationSampler(Sampler):
    def fit(self,X_train=None,feature_idx=None,feature_type=None,):
        # No fitting is required for permutation.
        self.is_fitted = True
        return self
    
    def sample_batch(
        self,
        X_eval,
        feature_indices,
        n_resamples,
        X_background=None,
    ):
        """Draw replacements from empirical marginal distributions.

        Evaluation feature values do not affect the distribution.
        Each observation receives separate draws from the selected
        background columns.
        """
        X_eval = np.asarray(X_eval)
        X_background = np.asarray(X_background)
        if X_background.ndim != 2:
            raise ValueError("X_background must be a 2D array.")

        X_eval, feature_indices, replacement_values = self._prepare_sampling(
            X_eval,
            feature_indices,
            n_resamples,
            n_features=X_background.shape[1],
        )

        for feature_idx in feature_indices:
            if not 0 <= feature_idx < X_background.shape[1]:
                raise ValueError("feature_idx is out of range.")

        n_eval = len(X_eval)

        # Preserve the previous observation-then-feature draw order.
        for row in range(n_eval):
            for local_j, feature_idx in enumerate(feature_indices):
                replacement_values[row, local_j] = self.rng.choice(
                    X_background[:, feature_idx],
                    size=n_resamples,
                    replace=True,
                )

        return replacement_values