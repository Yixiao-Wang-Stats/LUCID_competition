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
        x_minus_j = np.delete(x,feature_idx,).reshape(1, -1)
        conditional_mean_x = model.predict(x_minus_j).item()

        X_reference_minus_j = np.delete(X_reference, feature_idx, axis=1)  # shape (evaluation_sample_size, feature_size-1)
        conditional_mean_reference = model.predict(X_reference_minus_j)  # shape (evaluation_sample_size,)
        reference_residuals = X_reference[:, feature_idx]-conditional_mean_reference
        sampled_residuals = self.rng.choice(reference_residuals,size=replacement_sample_size,replace=True,)
        replacement_samples = (conditional_mean_x + sampled_residuals)

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
        for local_j, feature_idx in enumerate(feature_indices):
            if feature_idx not in self.models:
                raise RuntimeError(
                    "The sampler has not been fitted for this feature."
                )

            model = self.models[feature_idx]
            # predict conditional mean for the whole evaluation batch
            evaluation_sample_minus_j = np.delete(evaluation_sample,feature_idx,axis=1,)
            conditional_mean_evaluation = model.predict(evaluation_sample_minus_j)  # shape (evaluation_sample_size,)

            # calculate residual pool on the reference dataset
            X_reference_minus_j = np.delete(X_reference, feature_idx, axis=1)  # shape (evaluation_sample_size, feature_size-1)
            conditional_mean_reference = model.predict(X_reference_minus_j)  # shape (evaluation_sample_size,)
            reference_residuals = X_reference[:, feature_idx]-conditional_mean_reference
            sampled_residuals = self.rng.choice(reference_residuals,size=(evaluation_sample_size,replacement_sample_size,),replace=True,)  # shape (evaluation_sample_size, replacement_sample_size)


            replacement_samples = (conditional_mean_evaluation[:, None]+ sampled_residuals) # shape (evaluation_sample_size, replacement_sample_size)
            samples[:, local_j, :] = replacement_samples

        return samples
