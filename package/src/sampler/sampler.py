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
        replacement_sample_size,
        X_reference=None,
    ):
        """
        Generate replacement values for one observation
        and one feature.

        Returns
        -------
        replacements : ndarray of shape (replacement_sample_size,)
        """
        pass
    
    def sample_batch(
        self,
        evaluation_sample,
        feature_indices,
        replacement_sample_size,
        X_reference=None,
    ):
        evaluation_sample = np.asarray(evaluation_sample)
        feature_indices = np.asarray(feature_indices)

        evaluation_sample_size = evaluation_sample.shape[0]
        feature_size = len(feature_indices)

        samples = np.empty(
            (evaluation_sample_size, feature_size, replacement_sample_size)
        )

        for i, x in enumerate(evaluation_sample):
            for local_j, feature_idx in enumerate(feature_indices):
                samples[i, local_j] = self.sample(
                    x=x,
                    feature_idx=feature_idx,
                    replacement_sample_size=replacement_sample_size,
                    X_reference=X_reference,
                )

        return samples
    

    def _next_r_seed(self):
        return int(self.rng.integers(0,2**31 - 1))
