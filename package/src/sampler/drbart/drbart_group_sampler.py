# This is the sampler for drbart group feature importance
# We keep the sampling order as the feature original input order

import subprocess
import numpy as np
from pathlib import Path

from .drbart_sampler import DRBartSampler


class DRBartGroupSampler(DRBartSampler):
    def __init__(
        self,
        output_dir,
        feature_types,
        seed=2026,
        nburn=2000,
        nsim=2000,
        nthin=1,
        m_mean=200,
        m_var=100,
        variance="ux",
        random_state=None
    ):
        super().__init__(
            output_dir=output_dir,
            feature_types=feature_types,
            seed=seed,
            nburn=nburn,
            nsim=nsim,
            nthin=nthin,
            m_mean=m_mean,
            m_var=m_var,
            variance=variance,
            random_state=random_state,
        )
        self.group_fits = {}

    
    def fit_group(self, X_train, feature_indices, feature_types=None):
        X_train, feature_indices, feature_types = (
            self._prepare_group_fitting(
                X_train,
                feature_indices,
                feature_types,
            )
        )
        feature_size = self.feature_size
        group_key = tuple(feature_indices)
        group_output_dir = self.output_dir / (
            "group_" + "_".join(str(feature_idx) for feature_idx in feature_indices)
        )
        group_output_dir.mkdir(parents=True,exist_ok=True)

        outside_group_indices = [
            feature_idx
            for feature_idx in range(feature_size)
            if feature_idx not in feature_indices
        ]

        group_fits = []

        for local_j,feature_idx in enumerate(feature_indices):
            conditioning_indices = (outside_group_indices+ feature_indices[:local_j])  # remaning feature group indices for j-th feature in the group
            X_predictors = X_train[:,conditioning_indices]
            X_target = X_train[:,feature_idx]
            feature_output_dir = (group_output_dir / f"feature_{feature_idx}")
            feature_type = feature_types[local_j]
            if feature_type in {"binary", "categorical"}:
                self.bart_sampler._fit_model(
                    X_predictors=X_predictors,
                    X_target=X_target,
                    feature_type=feature_type,
                    feature_output_dir=feature_output_dir,
                )
            else:
                self._fit_model(
                    X_predictors=X_predictors,
                    X_target=X_target,
                    feature_output_dir=feature_output_dir,
                )
            posterior_count = int((feature_output_dir / "posterior_count.txt").read_text())
            group_fits.append(
                {
                    "feature_idx": feature_idx,
                    "feature_type": feature_type,
                    "conditioning_indices": conditioning_indices,
                    "fit_dir": feature_output_dir,
                    "posterior_count": posterior_count,
                }
            )
        self.group_fits[group_key] = group_fits  # shape: (len(feature_indices),)
        self.is_fitted = True

        return self


    def sample_group_batch(
        self,
        X_eval,
        feature_indices,
        n_resamples,
        X_background=None,
    ):
        return self._sample_group_batch(
            X_eval=X_eval,
            feature_indices=feature_indices,
            n_resamples=n_resamples,
            mode="lucid",
            X_background=X_background,
        )


    def _sample_factor(
        self,
        factor_fit,
        X_predictors,
        n_resamples,
        mode,
        X_background,
        posterior_id=None,
        row_posterior_ids=None,
    ):
        """Sample one group factor using its fitted backend."""
        if factor_fit["feature_type"] in {"binary", "categorical"}:
            return self.bart_sampler._sample_model(
                X_predictors=X_predictors,
                fit_dir=factor_fit["fit_dir"],
                n_resamples=n_resamples,
                posterior_id=posterior_id,
                row_posterior_ids=row_posterior_ids,
            )

        feature_idx = factor_fit["feature_idx"]
        bounds = self._resolve_feature_bounds(
            X_background=np.asarray(X_background)[:, [feature_idx]],
            feature_types=["continuous"],
        )[0]

        return self._sample_batch_from_r(
            X_eval_minus_j=X_predictors,
            fit_dir=factor_fit["fit_dir"],
            n_resamples=n_resamples,
            mode=mode,
            posterior_id=posterior_id,
            bounds=bounds,
        )

    def _sample_group_batch(
        self,
        X_eval,
        feature_indices,
        n_resamples,
        mode,
        factor_posterior_ids=None,
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
        if mode not in {"lucid","credible_interval"}:
            raise ValueError("mode must be either 'lucid' or 'credible_interval'.")

        feature_indices = [int(feature_idx) for feature_idx in feature_indices]
        group_key = tuple(feature_indices)

        if group_key not in self.group_fits:
            raise RuntimeError("The sampler has not been fitted for this feature group.")

        group_fits = self.group_fits[group_key]  # shape: (len(feature_indices),)

        if mode == "credible_interval":
            if factor_posterior_ids is None:
                raise ValueError("factor_posterior_ids is required for credible_interval mode.")
            if len(factor_posterior_ids) != len(feature_indices):
                raise ValueError("factor_posterior_ids must have the same length as feature_indices.")

        n_eval = len(X_eval)
        feature_to_local = {
            feature_idx: local_j
            for local_j,feature_idx in enumerate(feature_indices)
        }  # build a mapping from feature indices to local indices

        for local_j,factor_fit in enumerate(group_fits):
            posterior_id = (
                factor_posterior_ids[local_j]
                if mode == "credible_interval"
                else None
            )
            feature_type = factor_fit["feature_type"]
            conditioning_indices = factor_fit["conditioning_indices"]  # conditioning indices (remaining features),outside the group
            fit_dir = factor_fit["fit_dir"]
            feature_idx = factor_fit["feature_idx"]
            if local_j == 0:
                X_predictors = X_eval[:,conditioning_indices]  # all features reamining here 

                replacement_values = self._sample_factor(
                    factor_fit=factor_fit,
                    X_predictors=X_predictors,
                    n_resamples=n_resamples,
                    mode=mode,
                    X_background=X_background,
                    posterior_id=posterior_id,
                )

                samples[:, local_j, :] = replacement_values.reshape(
                    n_eval, n_resamples
                )
                continue
            
            # Start from the second loop, previous fitted factors are added into conditional indices as well. 
            X_predictors = np.empty(
                (
                    n_eval,
                    n_resamples,
                    len(conditioning_indices),
                )
            )  # shape: (n_eval, n_resamples, len(conditioning_indices))

            for conditioning_local_j,conditioning_idx in enumerate(conditioning_indices):
                # Use the already sampled value for features in the group.
                if conditioning_idx in feature_to_local:
                    previous_local_j = feature_to_local[conditioning_idx]
                    X_predictors[:,:,conditioning_local_j] = (
                        samples[:,previous_local_j,:]
                    )
                # Otherwise, use the original value for features outside the group.
                else:
                    X_predictors[:,:,conditioning_local_j] = (
                        X_eval[:,conditioning_idx,None]
                    )
            # So the output will look like (in 3D):
            #           -s[ repeat replacement sample_size times ]
            #       -s[ repeat replacement sample_size times ]
            #   x[x(1), ... , x(n_resamples)]
            # 
            # Then for each X_predictors[i, j, :]
            # is the obersation i; posterior j; 's remaining feature
            # | x_i_{in group}^{posterior=j}, -S_i

            X_predictors = X_predictors.reshape(
                n_eval * n_resamples,
                len(conditioning_indices),
            )  # We then flatten the data, so each row represents a | x_i_{in group}^{posterior=j}, -S_i

            row_posterior_ids = None
            if (
                feature_type in {"binary", "categorical"}
                and mode == "lucid"
            ):
                path_ids = self.rng.integers(
                    1,
                    factor_fit["posterior_count"] + 1,
                    size=n_resamples,
                )
                row_posterior_ids = np.tile(path_ids, n_eval)

            replacement_values = self._sample_factor(
                factor_fit=factor_fit,
                X_predictors=X_predictors,
                n_resamples=1,
                mode=mode,
                X_background=X_background,
                posterior_id=(
                    factor_posterior_ids[local_j]
                    if mode == "credible_interval"
                    else None
                ),
                row_posterior_ids=row_posterior_ids,
            )
            samples[:, local_j, :] = replacement_values.reshape(
                n_eval, n_resamples
            )

        return samples

    
    def sample_group_batch_posterior_credible_interval(
        self,
        X_eval,
        feature_indices,
        n_resamples,
        posterior_id,
        X_background=None,
    ):
        if posterior_id <= 0:
            raise ValueError("posterior_id must be positive.")

        feature_indices = np.asarray(feature_indices)

        if feature_indices.ndim != 1:
            raise ValueError("feature_indices must be a 1D array.")

        group_key = tuple(
            int(feature_idx)
            for feature_idx in feature_indices
        )

        if group_key not in self.group_fits:
            raise RuntimeError("The sampler has not been fitted for this feature group.")

        group_fits = self.group_fits[group_key]
        
        # Use posterior_id to deterministically choose
        # one posterior draw for each factor.
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    self.seed,
                    posterior_id,
                ]
            )
        )
        
        factor_posterior_ids = np.array([
            rng.integers(
                1,
                factor_fit["posterior_count"] + 1,
            )
            for factor_fit in group_fits
        ])

        return self._sample_group_batch(
            X_eval=X_eval,
            feature_indices=feature_indices,
            n_resamples=n_resamples,
            mode="credible_interval",
            factor_posterior_ids=factor_posterior_ids,
            X_background=X_background,
        )
