import numpy as np

from scipy.stats import norm

from ..sampler import Sampler

#  We first write two helper functions from evaluation_samples to Gaussian copula.

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
            raise ValueError(
                "X_train must be a 2D array."
            )
        
        feature_size = X_train.shape[1]

        if not 0 <= feature_idx < feature_size:
            raise ValueError(
                "feature_idx is out of range."
            )

        if self.feature_size is None:
            self.feature_size = feature_size

        elif self.feature_size != feature_size:
            raise ValueError(
                "X_train has a different number of features "
                "from previous fitted data."
            )
        
        # fit model on z_-j
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
    
    def sample(self, x, feature_idx, replacement_sample_size, X_reference:np.ndarray):
        if not self.is_fitted:
            raise RuntimeError(
                "Sampler must be fitted before sampling."
            )

        if feature_idx not in self.models:
            raise RuntimeError(
                "The sampler has not been fitted for this feature."
            )

        if replacement_sample_size <= 0:
            raise ValueError(
                "replacement_sample_size must be positive."
            )

        x = np.asarray(x)

        if x.ndim != 1:
            raise ValueError(
                "x must be a 1D feature vector."
            )

        if len(x) != self.feature_size:
            raise ValueError(
                "x has the wrong number of features."
            )

        X_reference = np.asarray(X_reference)

        if X_reference.ndim != 2:
            raise ValueError(
                "X_reference must be a 2D array."
            )

        if X_reference.shape[1] != self.feature_size:
            raise ValueError(
                "X_reference has the wrong number of features."
            )

        model = self.models[feature_idx]
        z = np.empty(self.feature_size)

        for j in range(self.feature_size):
            z[j] = _to_gaussian_with_empirical_distribution_reference(
                x[j],
                self.X_train[:, j],
            )

        z_minus_j = np.delete(z,feature_idx,).reshape(1, -1)
        conditional_mean_z = model.predict(z_minus_j).item()
        
        Z_reference = np.empty_like(X_reference, dtype=float)

        for j in range(self.feature_size):
            Z_reference[:, j] = _to_gaussian_with_empirical_distribution_reference(
                X_reference[:, j],
                self.X_train[:, j],
            )
            
        Z_reference_minus_j = np.delete(Z_reference, feature_idx, axis=1)
        conditional_mean_reference = model.predict(Z_reference_minus_j)
        reference_residuals = (Z_reference[:, feature_idx] - conditional_mean_reference)
        sampled_residuals = self.rng.choice(reference_residuals,size=replacement_sample_size,replace=True,)
        replacement_samples_gaussian = (conditional_mean_z + sampled_residuals)
        replacement_samples = _from_gaussian_with_empirical_distribution_reference(
            replacement_samples_gaussian,
            self.X_train[:, feature_idx],
        )
        return replacement_samples

    # rewrite sample batch, since for each batch we do not want predict for each samples!
    def sample_batch(
        self,
        evaluation_sample,
        feature_indices,
        replacement_sample_size,
        X_reference=None,
    ):
        if not self.is_fitted:
            raise RuntimeError(
                "Sampler must be fitted before sampling."
            )


        evaluation_sample = np.asarray(evaluation_sample)

        if evaluation_sample.ndim != 2:
            raise ValueError(
                "evaluation_sample must be a 2D array."
            )


        if evaluation_sample.shape[1] != self.feature_size:
            raise ValueError(
                "evaluation_sample has the wrong number of features."
            )


        feature_indices = np.asarray(feature_indices)

        if feature_indices.ndim != 1:
            raise ValueError(
                "feature_indices must be a 1D array."
            )


        if replacement_sample_size <= 0:
            raise ValueError(
                "replacement_sample_size must be positive."
            )


        X_reference = np.asarray(X_reference)

        if X_reference.ndim != 2:
            raise ValueError(
                "X_reference must be a 2D array."
            )


        if X_reference.shape[1] != self.feature_size:
            raise ValueError(
                "X_reference has the wrong number of features."
            )


        evaluation_sample_size = evaluation_sample.shape[0]
        selected_feature_size = len(feature_indices)
        samples = np.empty(
                (evaluation_sample_size, selected_feature_size, replacement_sample_size)
            )
        # transform evaluation sample to Gaussian space
        gaussian_evaluation_sample = np.empty_like(evaluation_sample,dtype=float)
        # transform reference sample to Gaussian space
        gaussian_X_reference = np.empty_like(X_reference,dtype=float)
        for j in range(self.feature_size):
            gaussian_evaluation_sample[:, j] = _to_gaussian_with_empirical_distribution_reference(
                evaluation_sample[:, j],
                self.X_train[:, j],
            )

            gaussian_X_reference[:, j] = _to_gaussian_with_empirical_distribution_reference(
                X_reference[:, j],
                self.X_train[:, j],
            )
        

        for local_j, feature_idx in enumerate(feature_indices):
            if feature_idx not in self.models:
                raise RuntimeError(
                    "The sampler has not been fitted for this feature."
                )

            model = self.models[feature_idx]
            
            # predict conditional mean for the whole evaluation batch
            gaussian_evaluation_sample_minus_j = np.delete(gaussian_evaluation_sample,feature_idx,axis=1,)
            conditional_mean_evaluation = model.predict(gaussian_evaluation_sample_minus_j)  # shape (evaluation_sample_size,)
            # calculate residual pool on the reference dataset
            gaussian_X_reference_minus_j = np.delete(gaussian_X_reference, feature_idx, axis=1)  # shape (evaluation_sample_size, feature_size-1)
            conditional_mean_reference = model.predict(gaussian_X_reference_minus_j)  # shape (evaluation_sample_size,)
            reference_residuals = gaussian_X_reference[:, feature_idx]-conditional_mean_reference
            sampled_residuals = self.rng.choice(reference_residuals,size=(evaluation_sample_size,replacement_sample_size,),replace=True,)  # shape (evaluation_sample_size, replacement_sample_size)

            gaussian_replacement_samples = (conditional_mean_evaluation[:, None]+ sampled_residuals) # shape (evaluation_sample_size, replacement_sample_size)
            replacement_samples = _from_gaussian_with_empirical_distribution_reference(
                gaussian_replacement_samples,
                self.X_train[:, feature_idx],
            )

            samples[:, local_j, :] = replacement_samples

        return samples
