"""Explain joint feature groups using conditional replacement sampling."""

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs, parallel_config

from .lucid_calculation.lucid_values import lucid_single_observation_group
from ._explainer import Explainer
from ._explanation import Explanation


class GroupExplainer(Explainer):
    """Compute group LUCID values; group members follow the supplied order."""

    def __init__(self, model, X_background, sampler="drbart", **kwargs):
        if sampler not in {"drbart", "tabpfn"}:
            raise ValueError("GroupExplainer supports drbart and tabpfn.")
        super().__init__(model, X_background, sampler=sampler, **kwargs)

    def _new_sampler(self):
        # Create the feature-group backend with the same feature schema and random seed.
        kwargs = {
            "random_state": self.random_state,
            **self.sampler_kwargs,
            "feature_types": self.feature_types,
        }
        if self.sampler_name == "tabpfn":
            from .samplers.tabpfn.tabpfn_group_sampler import TabPFNGroupSampler

            return TabPFNGroupSampler(
                _shared_model=self._tabpfn_model, **kwargs
            )

        from .samplers.drbart.drbart_group_sampler import DRBartGroupSampler

        kwargs["output_dir"] = self.output_dir / "drbart"
        kwargs.setdefault(
            "seed",
            self.random_state if self.random_state is not None else 2026,
        )
        return DRBartGroupSampler(**kwargs)

    def _get_group_sampler(self, group_key):
        # Cache by the ordered feature indices because order determines the conditional
        # factors.
        if group_key not in self._samplers:
            sampler = self._new_sampler()
            sampler.fit_group(self.X_background, group_key)
            self._samplers[group_key] = sampler
        return self._samplers[group_key]

    def _compute_group(
        self,
        group_key,
        X_eval,
        reference_predictions,
        n_resamples,
        n_eval_batch,
        n_resamples_batch,
        **sampling_kwargs,
    ):
        sampler = self._get_group_sampler(group_key)
        # Assemble one output column per named feature group and cache each returned
        # sampler.
        values = np.empty(len(X_eval))

        # Generate joint replacements for a batch of evaluation observations.
        for start in range(0, len(X_eval), n_eval_batch):
            stop = min(start + n_eval_batch, len(X_eval))
            X_eval_batch = X_eval[start:stop]

            replacements = sampler.sample_group(
                X_eval=X_eval_batch,
                feature_indices=group_key,
                n_resamples=n_resamples,
                X_background=self.X_background,
                **sampling_kwargs,
            )

            # Replace the selected features jointly for each evaluation observation and
            # compute one feature-group value.
            for local_row, x in enumerate(X_eval_batch):
                row = start + local_row
                values[row] = lucid_single_observation_group(
                    x=x,
                    y=reference_predictions[row],
                    replacement_values=replacements[local_row],
                    n_resamples_batch=n_resamples_batch,
                    loss_type="squared_error",
                    model=self._predict,
                    feature_indices=group_key,
                    baseline_loss=0.0,
                )

        return values, sampler

    def __call__(
        self,
        X_eval,
        groups,
        n_resamples=1000,
        n_resamples_batch=100,
        n_eval_batch=64,
    ):
        """Explain a dictionary mapping group names to feature names or indices."""
        if min(n_resamples, n_resamples_batch, n_eval_batch) <= 0:
            raise ValueError("Resample counts and batch sizes must be positive.")

        if isinstance(X_eval, pd.DataFrame) and self._uses_dataframe:
            X_eval = X_eval.loc[:, self.feature_names]

        X_eval = np.asarray(X_eval, dtype=float)
        if X_eval.ndim != 2 or X_eval.shape[1] != len(self.feature_names):
            raise ValueError("X_eval must have the same features as X_background.")

        # Keep user-facing feature-group labels separate from ordered indices used for
        # fitting.
        group_names = list(groups)
        group_keys = [
            tuple(
                self.feature_names.index(feature)
                if isinstance(feature, str)
                else int(feature)
                for feature in features
            )
            for features in groups.values()
        ]

        if not group_keys or any(not key for key in group_keys):
            raise ValueError("Provide at least one nonempty group.")
        if len(set(group_keys)) != len(group_keys):
            raise ValueError("Each ordered feature group must be unique.")

        # Use each unmodified prediction as the reference for all feature-group
        # replacements.
        reference_predictions = self._predict(X_eval)

        # Limit concurrent feature-group tasks to available workers and any Slurm
        # allocation.
        n_workers = effective_n_jobs(self.n_jobs)
        allocated = os.environ.get("SLURM_CPUS_PER_TASK")
        if allocated is not None:
            n_workers = min(n_workers, int(allocated))
        n_workers = max(1, min(n_workers, len(group_keys)))

        arguments = (
            X_eval,
            reference_predictions,
            n_resamples,
            n_eval_batch,
            n_resamples_batch,
        )

        # Run independent feature groups concurrently; conditional factors within a feature
        # group remain sequential.
        if n_workers > 1:
            backend = "threading" if self.sampler_name == "tabpfn" else "loky"
            options = (
                {} if backend == "threading"
                else {"inner_max_num_threads": 1}
            )

            with parallel_config(backend=backend, **options):
                results = Parallel(
                    n_jobs=n_workers,
                    temp_folder=str(self.output_dir),
                )(
                    delayed(self._feature_worker(key)._compute_group)(
                        key, *arguments
                    )
                    for key in group_keys
                )
        else:
            results = [
                self._compute_group(key, *arguments)
                for key in group_keys
            ]

        values = np.empty((len(X_eval), len(group_keys)))

        for column, (key, result) in enumerate(zip(group_keys, results)):
            group_values, sampler = result
            values[:, column] = group_values
            self._samplers[key] = sampler

        return Explanation(
            values=values,
            feature_names=group_names,
        )
