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
        super().__init__(
            feature_types=feature_types, 
            random_state=random_state
        )
        
        # Model used to estimate E[X_j | X_-j].
        self.predictive_model_class = predictive_model_class
        self.predictive_model_kwargs = (
            {} if predictive_model_kwargs is None
            else predictive_model_kwargs
        )

    def _fit_feature(self, X_train, feature_idx, feature_type):
        X_minus_j_train = np.delete(X_train, feature_idx, axis=1)
        X_j_train = X_train[:, feature_idx]

        model = self.predictive_model_class(
            **self.predictive_model_kwargs
        )
        model.fit(X_minus_j_train, X_j_train)
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

        X_eval_minus_j = np.delete(X_eval, feature_idx, axis=1)
        conditional_mean_evaluation = fitted_model.predict(
            X_eval_minus_j
        )

        # Background residuals supply the empirical noise distribution.
        X_background_minus_j = np.delete(
            X_background, feature_idx, axis=1
        )
        conditional_mean_background = fitted_model.predict(
            X_background_minus_j
        )
        residuals = (
            X_background[:, feature_idx]
            - conditional_mean_background
        )
        sampled_residuals = self.rng.choice(
            residuals,
            size=(len(X_eval), n_resamples),
            replace=True,
        )

        return conditional_mean_evaluation[:, None] + sampled_residuals
