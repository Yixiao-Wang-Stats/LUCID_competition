# This is the sampler for drbart group feature importance
# We keep the sampling order as the feature original input order

import subprocess
import numpy as np
from pathlib import Path

from ..group_sampler import GroupSampler
from .drbart_sampler import DRBartSampler


class DRBartGroupSampler(GroupSampler):
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
        random_state=None,
    ):
        super().__init__(
            feature_types=feature_types,
            random_state=random_state,
        )

        self.drbart_sampler = DRBartSampler(
            output_dir=output_dir,
            feature_types=self.feature_types,
            seed=seed,
            nburn=nburn,
            nsim=nsim,
            nthin=nthin,
            m_mean=m_mean,
            m_var=m_var,
            variance=variance,
            random_state=random_state,
        )

        # Preserve one random stream for group posterior IDs and R sampling seeds.
        self.drbart_sampler.rng = self.rng

        self.seed = seed
        self.output_dir = self.drbart_sampler.output_dir
        self.exchange_dir = self.drbart_sampler.exchange_dir
        self.bart_sampler = self.drbart_sampler.bart_sampler

        
    def _fit_group_feature(
        self,
        X_predictors,
        X_target,
        feature_idx,
        feature_type,
        group_key,
    ):
        """Fit and retain one conditional factor's model files."""
        # Keep model files separate for each ordered feature group and each conditional
        # target feature.
        group_name = "group_" + "_".join(map(str, group_key))
        feature_output_dir = (
            self.output_dir / group_name / f"feature_{feature_idx}"
        )

        # Reuse the single-feature BART or DRBART fitting implementation with feature-group
        # conditioning columns.
        if feature_type in {"binary", "categorical"}:
            self.bart_sampler._fit_model(
                X_predictors=X_predictors,
                X_target=X_target,
                feature_type=feature_type,
                feature_output_dir=feature_output_dir,
            )
        else:
            self.drbart_sampler._fit_model(
                X_predictors=X_predictors,
                X_target=X_target,
                feature_output_dir=feature_output_dir,
            )

        # Retain the number of saved draws so uncertainty calls can select valid posterior
        # IDs.
        return {
            "fit_dir": feature_output_dir,
            "posterior_count": int(
                (feature_output_dir / "posterior_count.txt").read_text()
            ),
        }

    def _sample_group_feature(
        self,
        factor_fit,
        X_predictors,
        n_resamples,
        X_background,
        group_indices,
        n_eval,
        n_group_resamples,
        mode="lucid",
        factor_posterior_ids=None,
    ):
        """Prepare posterior options and sample one conditional factor."""
        if mode not in {"lucid", "credible_interval"}:
            raise ValueError(
                "mode must be either 'lucid' or 'credible_interval'."
            )

        feature_idx = factor_fit["feature_idx"]
        feature_type = factor_fit["feature_type"]
        fitted_model = factor_fit["fitted_model"]
        fit_dir = fitted_model["fit_dir"]
        local_j = group_indices.index(feature_idx)

        posterior_id = None
        row_posterior_ids = None

        # Fix one posterior draw per conditional factor for this feature-group uncertainty
        # replicate.
        if mode == "credible_interval":
            if (
                factor_posterior_ids is None
                or len(factor_posterior_ids) != len(group_indices)
            ):
                raise ValueError(
                    "Provide one posterior ID per group factor."
                )
            posterior_id = factor_posterior_ids[local_j]

        elif (
            local_j > 0
            and feature_type in {"binary", "categorical"}
        ):
            # Use one posterior ID per path, shared across observations.
            path_ids = self.rng.integers(
                1,
                fitted_model["posterior_count"] + 1,
                size=n_group_resamples,
            )
            # Match posterior IDs to flattened (evaluation observation, resampling path)
            # rows.
            row_posterior_ids = np.tile(path_ids, n_eval)

        if feature_type in {"binary", "categorical"}:
            return self.bart_sampler._sample_model(
                X_predictors=X_predictors,
                fit_dir=fit_dir,
                n_resamples=n_resamples,
                posterior_id=posterior_id,
                row_posterior_ids=row_posterior_ids,
            )

        # Bound this continuous feature using its own background marginal.
        bounds = self._resolve_feature_bounds(
            X_background=np.asarray(X_background)[:, [feature_idx]],
            feature_types=["continuous"],
        )[0]

        return self.drbart_sampler._sample_from_r(
            X_eval_minus_j=X_predictors,
            fit_dir=fit_dir,
            n_resamples=n_resamples,
            mode=mode,
            posterior_id=posterior_id,
            bounds=bounds,
        )
        
    def sample_group(
        self,
        X_eval,
        feature_indices,
        n_resamples,
        X_background=None,
        **sampling_kwargs,
    ):
        # Supply feature-group dimensions needed to interpret flattened resampling paths.
        return super().sample_group(
            X_eval=X_eval,
            feature_indices=feature_indices,
            n_resamples=n_resamples,
            X_background=X_background,
            group_indices=tuple(feature_indices),
            n_eval=len(X_eval),
            n_group_resamples=n_resamples,
            **sampling_kwargs,
        )

    def sample_group_posterior_credible_interval(
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
        
        # Choose one saved posterior draw for each conditional factor in the ordered feature
        # group.
        factor_posterior_ids = np.array([
            rng.integers(
                1,
                factor_fit["fitted_model"]["posterior_count"] + 1,
            )
            for factor_fit in group_fits
        ])

        return self.sample_group(
            X_eval=X_eval,
            feature_indices=feature_indices,
            n_resamples=n_resamples,
            X_background=X_background,
            mode="credible_interval",
            factor_posterior_ids=factor_posterior_ids,
        )
