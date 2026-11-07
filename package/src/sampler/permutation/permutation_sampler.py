import numpy as np
from ..sampler import Sampler

class PermutationSampler(Sampler):
    def fit(self,X_train=None,feature_idx=None,feature_type=None,):
        # No fitting is required for permutation.
        self.is_fitted = True
        return self
    
    def sample(self, x, feature_idx, replacement_sample_size, X_reference:np.ndarray):
        X_reference = np.asarray(X_reference)
        if X_reference.ndim != 2:
            raise ValueError("X_reference must be a 2D array.")

        if not 0 <= feature_idx < X_reference.shape[1]:
            raise ValueError("feature_idx is out of range.")

        replacement_pool = X_reference[:, feature_idx]
        return self.rng.choice(replacement_pool, size=replacement_sample_size, replace=True)
