import numpy as np

from scipy.stats import norm

from ..sampler import Sampler

#  We first write two helper functions from X_eval to Gaussian copula.

def _to_gaussian_with_empirical_distribution_reference(sample, empirical_distribution_reference_sample,):
    # Since we don't have true CDF, we can only use empirical distribution
    reference_sample = np.asarray(empirical_distribution_reference_sample)
    sample = np.asarray(sample)
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

    probabilities = norm.cdf(sample)
    reference_probabilities = (np.arange(len(sorted_reference_sample)) + 0.5) / len(sorted_reference_sample)  # probabilities in the empirical distribution
    probabilities = np.clip(probabilities, 0.5 / len(sorted_reference_sample),1.0 - 0.5 / len(sorted_reference_sample),)

    return np.interp(probabilities,reference_probabilities,sorted_reference_sample)



class GCMRSampler(Sampler):
    def __init__(
        self,
        predictive_model_class,
        predictive_model_kwargs=None,
        random_state=None,
    ):
        super().__init__(random_state)
        self.predictive_model_class = predictive_model_class
        if predictive_model_kwargs is None:
            predictive_model_kwargs = {}
        self.predictive_model_kwargs = predictive_model_kwargs
        self.models = {}
        self.residuals = {}
        self.feature_size = None
        self.X_train = None

    def fit(self, X_train, feature_idx, feature_type=None,):
        X_train = np.asarray(X_train)

        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array.")
        
        feature_size = X_train.shape[1]

        if not 0 <= feature_idx < feature_size:
            raise ValueError("feature_idx is out of range.")

        if self.feature_size is None:
            self.feature_size = feature_size

        elif self.feature_size != feature_size:
            raise ValueError("X_train has a different number of features from previous fitted data.")
        
        # fit model on z_-j, where z_j is the Gaussian-transformed feature j
        self.X_train = X_train
        Z_train = np.empty_like(X_train, dtype=float)
        for j in range(feature_size):
            Z_train[:, j] = _to_gaussian_with_empirical_distribution_reference(
                X_train[:, j],
                X_train[:, j],
            )
        Z_minus_j = np.delete(Z_train, feature_idx, axis=1)
        Z_j = Z_train[:, feature_idx]

        model = self.predictive_model_class(**self.predictive_model_kwargs)
        model.fit(Z_minus_j, Z_j)
        
        self.models[feature_idx] = model
        self.is_fitted = True

        return self

    def sample_batch(
        self,
        X_eval,
        feature_indices,
        n_resamples,
        X_background=None,
    ):
        if not self.is_fitted:
            raise RuntimeError("Sampler must be fitted before sampling.")

        X_eval, feature_indices, samples = self._prepare_sampling(
            X_eval,
            feature_indices,
            n_resamples,
            n_features=self.feature_size,
        )
        n_eval = len(X_eval)

        X_background = np.asarray(X_background)

        if X_background.ndim != 2:
            raise ValueError("X_background must be a 2D array.")

        if X_background.shape[1] != self.feature_size:
            raise ValueError("X_background has the wrong number of features.")
        # transform evaluation sample to Gaussian space
        gaussian_X_eval = np.empty_like(X_eval,dtype=float)
        # transform X_background sample to Gaussian space
        gaussian_X_background = np.empty_like(X_background,dtype=float)
        for j in range(self.feature_size):
            gaussian_X_eval[:, j] = _to_gaussian_with_empirical_distribution_reference(
                X_eval[:, j],
                self.X_train[:, j],
            )

            gaussian_X_background[:, j] = _to_gaussian_with_empirical_distribution_reference(
                X_background[:, j],
                self.X_train[:, j],
            )
        
        for local_j, feature_idx in enumerate(feature_indices):
            if feature_idx not in self.models:
                raise RuntimeError("The sampler has not been fitted for this feature.")

            model = self.models[feature_idx]
            
            # predict conditional mean for the whole evaluation batch
            gaussian_X_eval_minus_j = np.delete(gaussian_X_eval,feature_idx,axis=1,)
            conditional_mean_evaluation = model.predict(gaussian_X_eval_minus_j)  # shape (n_eval,)
            # calculate residual pool on the reference dataset
            gaussian_X_background_minus_j = np.delete(gaussian_X_background, feature_idx, axis=1)  # shape (n_eval, feature_size-1)
            conditional_mean_reference = model.predict(gaussian_X_background_minus_j)  # shape (n_eval,)
            reference_residuals = gaussian_X_background[:, feature_idx]-conditional_mean_reference
            sampled_residuals = self.rng.choice(reference_residuals,size=(n_eval,n_resamples,),replace=True,)  # shape (n_eval, n_resamples)

            gaussian_replacement_values = (conditional_mean_evaluation[:, None]+ sampled_residuals) # shape (n_eval, n_resamples)
            replacement_values = _from_gaussian_with_empirical_distribution_reference(
                gaussian_replacement_values,
                self.X_train[:, feature_idx],
            )

            samples[:, local_j, :] = replacement_values

        return samples
