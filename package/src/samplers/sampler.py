# This file is a base class for all samplers.
# For all the samplers, here is the uniform sampling procedure.
# - 1. We train model on the training dataset, including the mean estimator, the conditional density estimator, a tree based model, or just a fake one for permutation sampling.
# - 2. We generate replacement samples on the evaluation dataset, including residual permutation, permutation.


from abc import ABC
import numpy as np
from pathlib import Path

class Sampler(ABC):
    def __init__(self, feature_types, random_state=None):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        if feature_types is None:
            raise ValueError("feature_types must be provided.")

        self.feature_types = list(feature_types)
        if any(
            feature_type not in {"continuous", "binary", "categorical"}
            for feature_type in self.feature_types
        ):
            raise ValueError("Feature types must be binary, categorical, or continuous.")

        self.feature_size = None
        self.is_fitted = False
        self.fits = {}

    def fit(self, X_train, feature_idx, feature_type=None):
        """Fit and retain the conditional sampler for one feature."""
        X_train = self._prepare_fitting(
            X_train,
            feature_idx,
            feature_type=feature_type,
        )
        feature_type = self.feature_types[feature_idx]

        fitted_model = self._fit_feature(
            X_train,
            feature_idx,
            feature_type,
        )

        self.fits[feature_idx] = fitted_model
        self.is_fitted = True
        return self

    def sample_batch(
        self,
        X_eval,
        feature_indices,
        n_resamples,
        X_background=None,
        **sampling_kwargs,
    ):
        """Generate replacements with shape (n_eval, n_selected_features, n_resamples)."""
        if not self.is_fitted:
            raise RuntimeError("Sampler must be fitted before sampling.")

        X_eval, feature_indices, samples = self._prepare_sampling(
            X_eval,
            feature_indices,
            n_resamples,
            n_features=self.feature_size,
        )

        for local_j, feature_idx in enumerate(feature_indices):
            if feature_idx not in self.fits:
                raise RuntimeError("The sampler has not been fitted for this feature.")

            samples[:, local_j, :] = self._sample_feature(
                self.fits[feature_idx],
                X_eval,
                feature_idx,
                n_resamples,
                X_background,
                **sampling_kwargs,
            )

        return samples  
        
    def _fit_feature(self, X_train, feature_idx, feature_type):
        """Return the fitted result needed to sample one feature."""
        raise NotImplementedError

    def _sample_feature(
        self,
        fitted_model,
        X_eval,
        feature_idx,
        n_resamples,
        X_background,
        **sampling_kwargs,
    ):
        """Return replacements with shape (n_eval, n_resamples)."""
        raise NotImplementedError

    @staticmethod
    def _resolve_feature_bounds(
        X_background,
        feature_types,
        bounds=None,
        padding=0.1,
    ):
        """Resolve explicit or background-derived continuous feature bounds."""
        X_background = np.asarray(X_background)
        bounds = {} if bounds is None else bounds
        resolved = {}

        for feature_idx, feature_type in enumerate(feature_types):
            if feature_type != "continuous":
                continue

            if feature_idx in bounds:
                lower, upper = bounds[feature_idx]
            else:
                values = X_background[:, feature_idx]
                observed_lower = float(values.min())
                observed_upper = float(values.max())

                # A small fallback gives constant features a nonzero interval.
                width = max(
                    observed_upper - observed_lower,
                    1e-8 * max(
                        1.0, abs(observed_lower), abs(observed_upper)
                    ),
                )
                lower = observed_lower - padding * width
                upper = observed_upper + padding * width

            if not np.isfinite([lower, upper]).all() or lower >= upper:
                raise ValueError(f"Invalid bounds for feature {feature_idx}.")

            resolved[feature_idx] = (float(lower), float(upper))

        return resolved

    def _next_r_seed(self):
        return int(self.rng.integers(0,2**31 - 1))

    def _prepare_fitting(
        self, 
        X_train, 
        feature_idx, 
        feature_type=None
    ):
        """Validate training data against the stored feature schema."""
        X_train = np.asarray(X_train)
        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array.")

        feature_size = X_train.shape[1]

        if not 0 <= feature_idx < feature_size:
            raise ValueError("feature_idx is out of range.")

        if len(self.feature_types) != feature_size:
            raise ValueError("feature_types must have one entry per feature.")

        if self.feature_size is not None and self.feature_size != feature_size:
            raise ValueError("X_train has a different number of features from previous fitted data.")

        if feature_type is not None and feature_type != self.feature_types[feature_idx]:
            raise ValueError("feature_type does not match feature_types[feature_idx].")

        self.feature_size = feature_size
        return X_train

        
    @staticmethod
    def _prepare_sampling(
        X_eval,
        feature_indices,
        n_resamples,
        n_features,
    ):
        """Validate sampling inputs and allocate the replacement array."""
        X_eval = np.asarray(X_eval)
        if X_eval.ndim != 2:
            raise ValueError("X_eval must be a 2D array.")
        if X_eval.shape[1] != n_features:
            raise ValueError("X_eval has the wrong number of features.")

        feature_indices = np.asarray(feature_indices)
        if feature_indices.ndim != 1:
            raise ValueError("feature_indices must be a 1D array.")

        if n_resamples <= 0:
            raise ValueError("n_resamples must be positive.")

        samples = np.empty(
            (len(X_eval), len(feature_indices), n_resamples)
        )
        return X_eval, feature_indices, samples
    
    @staticmethod
    def _prepare_workspace(output_dir):
        """Create the workspace used to exchange files with R."""
        output_dir = Path(output_dir)
        exchange_dir = output_dir / "exchange"
        exchange_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, exchange_dir