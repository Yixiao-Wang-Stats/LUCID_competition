# This file is a base class for all samplers.
# For all the samplers, here is the uniform sampling procedure.
# - 1. We train model on the training dataset, including the mean estimator, the conditional density estimator, a tree based model, or just a fake one for permutation sampling.
# - 2. We generate replacement samples on the evaluation dataset, including residual permutation, permutation.


from abc import ABC, abstractmethod
import numpy as np

class Sampler(ABC):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.is_fitted = False

    @abstractmethod
    def fit(self, X_train, feature_idx, feature_type=None,):
        """
        Fit the sampler using training data.

        Some samplers may not require fitting.
        """
        # drbart train CDE
        # arf train CDE
        # cs_pfi train transformation tree
        # cmr and gcmr train mean
        # clique train a grid
        # permutation fake.
        pass

    @abstractmethod
    def sample(
        self,
        x,
        feature_idx,
        n_resamples,
        X_reference=None,
    ):
        """
        Generate replacement values for one observation
        and one feature.

        Returns
        -------
        replacements : ndarray of shape (n_resamples,)
        """
        pass
    
    def sample_batch(
        self,
        X_eval,
        feature_indices,
        n_resamples,
        X_reference=None,
    ):
        X_eval = np.asarray(X_eval)
        feature_indices = np.asarray(feature_indices)

        n_eval = X_eval.shape[0]
        feature_size = len(feature_indices)

        samples = np.empty(
            (n_eval, feature_size, n_resamples)
        )

        for i, x in enumerate(X_eval):
            for local_j, feature_idx in enumerate(feature_indices):
                samples[i, local_j] = self.sample(
                    x=x,
                    feature_idx=feature_idx,
                    n_resamples=n_resamples,
                    X_reference=X_reference,
                )

        return samples  # (n_eval, n_selected_features, n_resamples)
    

    def _next_r_seed(self):
        return int(self.rng.integers(0,2**31 - 1))
