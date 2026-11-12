import numpy as np
from scipy.stats import norm
from ..sampler import Sampler

#  We first write two helper functions from X_eval to Gaussian copula.

def _to_gaussian_with_empirical_distribution_reference(sample, empirical_distribution_reference_sample,):
    # Since we don't have true CDF, we can only use empirical distribution
    reference_sample = np.asarray(empirical_distribution_reference_sample)
    sample = np.asarray(sample)
    # Estimate marginal probabilities using midranks, retaining a finite range for normal
    # quantiles.
    sorted_reference_sample = np.sort(reference_sample) 
    left = np.searchsorted(sorted_reference_sample,sample,side="left",)  # left point less than or equal to sample
    right = np.searchsorted(sorted_reference_sample,sample,side="right",)  # right point greater than sample
    probabilities = (left + right) / (2.0 * len(sorted_reference_sample))  # midrank/total references approx CDF
    probabilities = np.clip(probabilities, 0.5 / len(sorted_reference_sample),1.0 - 0.5 / len(sorted_reference_sample),)

    return norm.ppf(probabilities)


def _from_gaussian_with_empirical_distribution_reference(sample, empirical_distribution_reference_sample,):
    # Transform Gaussian sample back to the original empirical distribution
    reference_sample = np.asarray(empirical_distribution_reference_sample)
    sample = np.asarray(sample)
    sorted_reference_sample = np.sort(reference_sample)

    # Map Gaussian values back through interpolated quantiles of the reference marginal.
    probabilities = norm.cdf(sample)
    reference_probabilities = (np.arange(len(sorted_reference_sample)) + 0.5) / len(sorted_reference_sample)  # probabilities in the empirical distribution
    probabilities = np.clip(probabilities, 0.5 / len(sorted_reference_sample),1.0 - 0.5 / len(sorted_reference_sample),)

    return np.interp(probabilities,reference_probabilities,sorted_reference_sample)



class GCMRSampler(Sampler):
    def __init__(
        self,
        feature_types,
        predictive_model_class,
        predictive_model_kwargs=None,
        random_state=None,
    ):
        super().__init__(
            feature_types=feature_types,
            random_state=random_state,
        )

        self.predictive_model_class = predictive_model_class
        self.predictive_model_kwargs = (
            {} if predictive_model_kwargs is None
            else predictive_model_kwargs
        )
        self.X_train = None  # Store the training data for empirical distribution references, which are used in _sample_feature.

    def _fit_feature(self, X_train, feature_idx, feature_type):
        self.X_train = X_train

        # Gaussianize each training marginal before fitting the conditional feature-mean
        # model.
        Z_train = np.empty_like(X_train, dtype=float)
        for j in range(self.feature_size):
            Z_train[:, j] = (
                _to_gaussian_with_empirical_distribution_reference(
                    X_train[:, j],
                    X_train[:, j],
                )
            )

        Z_minus_j_train = np.delete(Z_train, feature_idx, axis=1)
        Z_j_train = Z_train[:, feature_idx]

        model = self.predictive_model_class(
            **self.predictive_model_kwargs
        )
        model.fit(Z_minus_j_train, Z_j_train)
        return model

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

        # Anchor both transformations to the training marginals.
        gaussian_X_eval = np.empty_like(X_eval, dtype=float)
        gaussian_X_background = np.empty_like(
            X_background, dtype=float
        )

        for j in range(self.feature_size):
            gaussian_X_eval[:, j] = (
                _to_gaussian_with_empirical_distribution_reference(
                    X_eval[:, j],
                    self.X_train[:, j],
                )
            )
            gaussian_X_background[:, j] = (
                _to_gaussian_with_empirical_distribution_reference(
                    X_background[:, j],
                    self.X_train[:, j],
                )
            )

        gaussian_X_eval_minus_j = np.delete(
            gaussian_X_eval, feature_idx, axis=1
        )
        conditional_mean_evaluation = fitted_model.predict(
            gaussian_X_eval_minus_j
        )

        gaussian_X_background_minus_j = np.delete(
            gaussian_X_background, feature_idx, axis=1
        )
        conditional_mean_background = fitted_model.predict(
            gaussian_X_background_minus_j
        )
        # Form the pooled residual distribution in Gaussianized space.
        residuals = (
            gaussian_X_background[:, feature_idx]
            - conditional_mean_background
        )
        sampled_residuals = self.rng.choice(
            residuals,
            size=(len(X_eval), n_resamples),
            replace=True,
        )

        gaussian_replacement_values = (
            conditional_mean_evaluation[:, None]
            + sampled_residuals
        )

        # Transform generated feature values back to the original training marginal.
        return _from_gaussian_with_empirical_distribution_reference(
            gaussian_replacement_values,
            self.X_train[:, feature_idx],
        )