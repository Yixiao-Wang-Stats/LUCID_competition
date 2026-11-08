import numpy as np
from ..sampler import Sampler

class CliqueSampler(Sampler):
    def __init__(
        self,
        random_state=None,
    ):
        super().__init__(random_state)
        self.training_features = {}
        self.feature_types = {}
        self.feature_size = None

    def fit(
        self,
        X_train,
        feature_idx,
        feature_type=None,
    ):
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

        self.training_features[feature_idx] = X_train[:, feature_idx].copy()
        self.feature_types[feature_idx] = feature_type
        self.is_fitted = True

        return self

    def sample(
        self,
        x,
        feature_idx,
        n_resamples,
        X_reference:np.ndarray,
    ):
        if not self.is_fitted:
            raise RuntimeError(
                "Sampler must be fitted before sampling."
            )

        if feature_idx not in self.training_features:
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

        training_feature = self.training_features[feature_idx]
        probabilities = np.linspace(
            0.0,
            1.0,
            num=n_resamples,
        )
        replacement_values = np.quantile(
            training_feature,
            probabilities,
            method="linear",
        )

        if self.feature_types[feature_idx] != "continuous":
            categories = np.unique(training_feature)
            indices = np.abs(
                replacement_values[:, None]-categories[None, :]
            ).argmin(axis=1)
            replacement_values = categories[indices]

        return replacement_values

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

        n_eval = X_eval.shape[0]
        selected_feature_size = len(feature_indices)

        samples = np.empty(
            (
                n_eval,
                selected_feature_size,
                n_resamples,
            )
        )

        for local_j, feature_idx in enumerate(feature_indices):
            if feature_idx not in self.training_features:
                raise RuntimeError(
                    "The sampler has not been fitted for this feature."
                )

            training_feature = self.training_features[feature_idx]
            probabilities = np.linspace(
                0.0,
                1.0,
                num=n_resamples,
            )
            replacement_values = np.quantile(
                training_feature,
                probabilities,
                method="linear",
            )

            if self.feature_types[feature_idx] != "continuous":
                categories, counts = np.unique(training_feature, return_counts=True)
                category_counts = np.round(
                    n_resamples * counts / len(training_feature)
                ).astype(int)
                difference = n_resamples - category_counts.sum()
                category_counts[np.argmax(counts)] += difference  # make sure the total number equals the replacement samples.
                replacement_values = np.repeat(categories, category_counts)

            samples[:, local_j, :] = replacement_values[None, :]

        return samples