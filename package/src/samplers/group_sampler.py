"""Shared chain-rule fitting and sampling for ordered feature groups."""

import numpy as np

from .sampler import Sampler


class GroupSampler(Sampler):
    """Fit each group feature given outside features and preceding group features.

    Group order is part of the fitted context. Sampling follows that order,
    using generated values for preceding features on each resampling path.
    """

    def __init__(self, feature_types, random_state=None):
        super().__init__(feature_types, random_state=random_state)
        self.group_fits = {}

    def fit_group(self, X_train, feature_indices, feature_types=None):
        """Fit the conditional factors for one ordered group."""
        X_train, feature_indices, feature_types = self._prepare_group_fitting(
            X_train, feature_indices, feature_types
        )
        outside_group = [
            j for j in range(self.feature_size) if j not in feature_indices
        ]
        group_key = tuple(feature_indices)
        group_fits = []

        for local_j, feature_idx in enumerate(feature_indices):
            conditioning_indices = outside_group + feature_indices[:local_j]
            fitted_model = self._fit_group_feature(
                X_predictors=X_train[:, conditioning_indices],
                X_target=X_train[:, feature_idx],
                feature_idx=feature_idx,
                feature_type=feature_types[local_j],
                group_key=group_key,
            )
            group_fits.append({
                "feature_idx": feature_idx,
                "feature_type": feature_types[local_j],
                "conditioning_indices": conditioning_indices,
                "fitted_model": fitted_model,
            })

        self.group_fits[group_key] = group_fits
        self.is_fitted = True
        return self

    def sample_group_batch(
        self, 
        X_eval, 
        feature_indices, 
        n_resamples, 
        X_background=None,
        **sampling_kwargs,
    ):
        """Return joint replacements of shape (n_eval, group_size, n_resamples)."""
        if not self.is_fitted:
            raise RuntimeError("Sampler must be fitted before sampling.")
        X_eval, feature_indices, samples = self._prepare_sampling(
            X_eval, feature_indices, n_resamples, n_features=self.feature_size
        )
        group_key = tuple(feature_indices)
        if group_key not in self.group_fits:
            raise RuntimeError("The sampler has not been fitted for this feature group.")
        if len(X_eval) == 0:
            return samples

        feature_to_local = {
            feature_idx: local_j
            for local_j, feature_idx in enumerate(feature_indices)
        }
        n_eval = len(X_eval)

        for local_j, factor_fit in enumerate(self.group_fits[group_key]):
            conditioning_indices = factor_fit["conditioning_indices"]
            if local_j == 0:
                X_predictors = X_eval[:, conditioning_indices]
                count = n_resamples
            else:
                X_predictors = np.empty(
                    (n_eval, n_resamples, len(conditioning_indices))
                )
                for column, feature_idx in enumerate(conditioning_indices):
                    if feature_idx in feature_to_local:
                        X_predictors[:, :, column] = samples[
                            :, feature_to_local[feature_idx], :
                        ]
                    else:
                        X_predictors[:, :, column] = X_eval[:, feature_idx, None]
                # Keep each observation's resampling paths together.
                X_predictors = X_predictors.reshape(
                    n_eval * n_resamples, len(conditioning_indices)
                )
                count = 1

            replacement_values = self._sample_group_feature(
                factor_fit=factor_fit,
                X_predictors=X_predictors,
                n_resamples=count,
                X_background=X_background,
                **sampling_kwargs,
            )
            samples[:, local_j, :] = np.asarray(replacement_values).reshape(
                n_eval, n_resamples
            )

        return samples

    def _fit_group_feature(
        self, X_predictors, X_target, feature_idx, feature_type, group_key
    ):
        """Return the fitted backend context for one conditional factor."""
        raise NotImplementedError

    def _sample_group_feature(
        self, factor_fit, X_predictors, n_resamples, X_background, **sampling_kwargs
    ):
        """Return one factor's replacements with shape (n_rows, n_resamples)."""
        raise NotImplementedError

    def _prepare_group_fitting(
        self,
        X_train,
        feature_indices,
        feature_types=None,
    ):
        """Validate group fitting inputs and resolve target types."""
        feature_indices = np.asarray(feature_indices)

        if feature_indices.ndim != 1:
            raise ValueError("feature_indices must be a 1D array.")
        if len(feature_indices) == 0:
            raise ValueError("feature_indices must not be empty.")
        if len(np.unique(feature_indices)) != len(feature_indices):
            raise ValueError("feature_indices must not contain duplicates.")

        # Reuse full-dataset, schema, and feature-count validation.
        X_train = self._prepare_fitting(
            X_train,
            feature_indices[0],
        )

        if np.any(feature_indices < 0) or np.any(
            feature_indices >= self.feature_size
        ):
            raise ValueError("feature_indices contains an out-of-range feature index.")

        expected_types = [
            self.feature_types[feature_idx]
            for feature_idx in feature_indices
        ]
        if feature_types is not None and list(feature_types) != expected_types:
            raise ValueError("Group feature types do not match the stored feature schema.")

        feature_indices = [
            int(feature_idx) for feature_idx in feature_indices
        ]
        return X_train, feature_indices, expected_types