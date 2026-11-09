import numpy as np
from ..sampler import Sampler

class CMRSampler(Sampler):
    def __init__(
        self,
        predictive_model_class,
        feature_types,
        predictive_model_kwargs=None,
        random_state=None,
    ):
        super().__init__(feature_types=feature_types, random_state=random_state)
        # Model class for estimating E[X_j | X_-j]. 
        self.predictive_model_class = predictive_model_class
        if predictive_model_kwargs is None:
            predictive_model_kwargs = {}
        self.predictive_model_kwargs = predictive_model_kwargs
        self.models = {}
        self.residuals = {}

    def fit(self, X_train, feature_idx, feature_type=None,):
        X_train = self._prepare_fitting(X_train, feature_idx, feature_type)
        
        # fit model on x_-j
        X_minus_j = np.delete(X_train, feature_idx, axis=1)
        X_j = X_train[:, feature_idx]
        model = self.predictive_model_class(**self.predictive_model_kwargs)
        model.fit(X_minus_j, X_j)
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
        """Generate replacements using conditional means and resampled residuals.

        For each feature, predict its conditional mean at each evaluation
        observation and add residuals sampled with replacement from the
        background residual pool.

        `X_background` supplies the observations used to calculate residuals.
        Returns an array of shape
        (n_eval, n_selected_features, n_resamples).
        """
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

        for local_j, feature_idx in enumerate(feature_indices):
            if feature_idx not in self.models:
                raise RuntimeError("The sampler has not been fitted for this feature.")

            model = self.models[feature_idx]
            # predict conditional mean for the whole evaluation batch
            X_eval_minus_j = np.delete(X_eval,feature_idx,axis=1,)
            conditional_mean_evaluation = model.predict(X_eval_minus_j)  # shape (n_eval,)

            # calculate residual pool on the x_background dataset
            X_background_minus_j = np.delete(X_background, feature_idx, axis=1)  # shape (n_eval, feature_size-1)
            conditional_mean_reference = model.predict(X_background_minus_j)  # shape (n_eval,)
            reference_residuals = X_background[:, feature_idx]-conditional_mean_reference
            sampled_residuals = self.rng.choice(reference_residuals,size=(n_eval,n_resamples,),replace=True,)  # shape (n_eval, n_resamples)

            replacement_values = (conditional_mean_evaluation[:, None]+ sampled_residuals) # shape (n_eval, n_resamples)
            samples[:, local_j, :] = replacement_values

        return samples
