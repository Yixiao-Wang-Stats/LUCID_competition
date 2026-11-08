import numpy as np
from ..sampler import Sampler

class CMRSampler(Sampler):
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
        
        # fit model on x_-j
        X_minus_j = np.delete(X_train, feature_idx, axis=1)
        X_j = X_train[:, feature_idx]
        model = self.predictive_model_class(**self.predictive_model_kwargs)
        model.fit(X_minus_j, X_j)
        self.models[feature_idx] = model
        self.is_fitted = True

        return self
    
    def sample(self, x, feature_idx, n_resamples, X_reference:np.ndarray):
        if not self.is_fitted:
            raise RuntimeError(
                "Sampler must be fitted before sampling."
            )

        if feature_idx not in self.models:
            raise RuntimeError(
                "The sampler has not been fitted for this feature."
            )

        if n_resamples <= 0:
            raise ValueError(
                "n_resamples must be positive."
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
        x_minus_j = np.delete(x,feature_idx,).reshape(1, -1)
        conditional_mean_x = model.predict(x_minus_j).item()

        X_reference_minus_j = np.delete(X_reference, feature_idx, axis=1)  # shape (n_eval, feature_size-1)
        conditional_mean_reference = model.predict(X_reference_minus_j)  # shape (n_eval,)
        reference_residuals = X_reference[:, feature_idx]-conditional_mean_reference
        sampled_residuals = self.rng.choice(reference_residuals,size=n_resamples,replace=True,)
        replacement_values = (conditional_mean_x + sampled_residuals)

        return replacement_values

    # rewrite sample batch, since for each batch we do not want predict for each samples!
    def sample_batch(
        self,
        X_eval,
        feature_indices,
        n_resamples,
        X_reference=None,
    ):
        if not self.is_fitted:
            raise RuntimeError(
                "Sampler must be fitted before sampling."
            )


        X_eval = np.asarray(X_eval)

        if X_eval.ndim != 2:
            raise ValueError(
                "X_eval must be a 2D array."
            )


        if X_eval.shape[1] != self.feature_size:
            raise ValueError(
                "X_eval has the wrong number of features."
            )


        feature_indices = np.asarray(feature_indices)

        if feature_indices.ndim != 1:
            raise ValueError(
                "feature_indices must be a 1D array."
            )


        if n_resamples <= 0:
            raise ValueError(
                "n_resamples must be positive."
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


        n_eval = X_eval.shape[0]
        selected_feature_size = len(feature_indices)
        samples = np.empty(
                (n_eval, selected_feature_size, n_resamples)
            )
        for local_j, feature_idx in enumerate(feature_indices):
            if feature_idx not in self.models:
                raise RuntimeError(
                    "The sampler has not been fitted for this feature."
                )

            model = self.models[feature_idx]
            # predict conditional mean for the whole evaluation batch
            X_eval_minus_j = np.delete(X_eval,feature_idx,axis=1,)
            conditional_mean_evaluation = model.predict(X_eval_minus_j)  # shape (n_eval,)

            # calculate residual pool on the reference dataset
            X_reference_minus_j = np.delete(X_reference, feature_idx, axis=1)  # shape (n_eval, feature_size-1)
            conditional_mean_reference = model.predict(X_reference_minus_j)  # shape (n_eval,)
            reference_residuals = X_reference[:, feature_idx]-conditional_mean_reference
            sampled_residuals = self.rng.choice(reference_residuals,size=(n_eval,n_resamples,),replace=True,)  # shape (n_eval, n_resamples)


            replacement_values = (conditional_mean_evaluation[:, None]+ sampled_residuals) # shape (n_eval, n_resamples)
            samples[:, local_j, :] = replacement_values

        return samples
