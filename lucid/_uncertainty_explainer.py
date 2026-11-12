from pathlib import Path

import numpy as np

from ._explainer import Explainer
from ._uncertainty_explanation import UncertaintyExplanation


class UncertaintyExplainer(Explainer):
    """Estimate LUCID uncertainty from saved DRBART/BART posterior draws."""

    def __init__(self, model, X_background, sampler="drbart", **kwargs):
        if sampler != "drbart":
            raise ValueError("UncertaintyExplainer requires sampler='drbart'.")

        super().__init__(model, X_background, sampler=sampler, **kwargs)
        self._posterior_ids = {}

    def _compute_feature(self, feature_idx, *args, **kwargs):
        return super()._compute_feature(
            feature_idx,
            *args,
            posterior_id=int(self._posterior_ids[feature_idx]),
            **kwargs,
        )

    def __call__(
        self,
        X_eval,
        features=None,
        n_posterior_draws=100,
        n_resamples=1000,
        n_resamples_batch=100,
        n_eval_batch=64,
    ):
        if n_posterior_draws < 2:
            raise ValueError("n_posterior_draws must be at least two.")

        indices = (
            list(range(len(self.feature_names)))
            if features is None
            else [
                self.feature_names.index(feature)
                if isinstance(feature, str)
                else int(feature)
                for feature in features
            ]
        )
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Select one or more unique features.")

        rng = np.random.default_rng(self.random_state)
        selected_ids = []

        for feature_idx in indices:
            sampler = self._get_sampler(feature_idx)
            fit_dir = Path(sampler.fits[feature_idx])
            count = int((fit_dir / "posterior_count.txt").read_text())

            if n_posterior_draws > count:
                raise ValueError(
                    f"Feature {self.feature_names[feature_idx]} has only "
                    f"{count} saved posterior draws."
                )

            selected_ids.append(
                rng.choice(
                    count,
                    size=n_posterior_draws,
                    replace=False,
                ) + 1
            )

        posterior_ids = np.column_stack(selected_ids)
        posterior_values = []

        try:
            for ids in posterior_ids:
                self._posterior_ids = dict(zip(indices, ids))

                explanation = super().__call__(
                    X_eval,
                    features=indices,
                    n_resamples=n_resamples,
                    n_resamples_batch=n_resamples_batch,
                    n_eval_batch=n_eval_batch,
                )
                posterior_values.append(explanation.values)
        finally:
            self._posterior_ids = {}

        return UncertaintyExplanation(
            posterior_values=np.stack(posterior_values),
            feature_names=[self.feature_names[j] for j in indices],
            posterior_ids=posterior_ids,
        )
