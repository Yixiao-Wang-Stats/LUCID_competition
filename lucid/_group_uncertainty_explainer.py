import numpy as np

from ._group_explainer import GroupExplainer
from ._uncertainty_explanation import UncertaintyExplanation


class GroupUncertaintyExplainer(GroupExplainer):
    """Estimate group LUCID uncertainty from DRBART/BART posterior draws."""

    def __init__(self, model, X_background, sampler="drbart", **kwargs):
        if sampler != "drbart":
            raise ValueError(
                "GroupUncertaintyExplainer requires sampler='drbart'."
            )

        super().__init__(model, X_background, sampler=sampler, **kwargs)
        self._factor_posterior_ids = {}

    def _compute_group(self, group_key, *args, **kwargs):
        return super()._compute_group(
            group_key,
            *args,
            mode="credible_interval",
            factor_posterior_ids=self._factor_posterior_ids[group_key],
            **kwargs,
        )

    def __call__(
        self,
        X_eval,
        groups,
        n_posterior_draws=100,
        n_resamples=1000,
        n_resamples_batch=100,
        n_eval_batch=64,
    ):
        if n_posterior_draws < 2:
            raise ValueError("n_posterior_draws must be at least two.")

        resolved = {
            name: tuple(
                self.feature_names.index(feature)
                if isinstance(feature, str)
                else int(feature)
                for feature in features
            )
            for name, features in groups.items()
        }

        keys = list(resolved.values())
        if not keys or any(not key for key in keys):
            raise ValueError("Provide at least one nonempty group.")
        if len(set(keys)) != len(keys):
            raise ValueError("Each ordered feature group must be unique.")

        rng = np.random.default_rng(self.random_state)
        posterior_ids = {}

        for name, key in resolved.items():
            sampler = self._get_group_sampler(key)
            selected = []

            for factor in sampler.group_fits[key]:
                count = factor["fitted_model"]["posterior_count"]

                if n_posterior_draws > count:
                    raise ValueError(
                        f"A factor in group {name} has only "
                        f"{count} saved posterior draws."
                    )

                selected.append(
                    rng.choice(
                        count,
                        size=n_posterior_draws,
                        replace=False,
                    ) + 1
                )

            posterior_ids[name] = np.column_stack(selected)

        posterior_values = []

        try:
            for draw in range(n_posterior_draws):
                self._factor_posterior_ids = {
                    key: posterior_ids[name][draw]
                    for name, key in resolved.items()
                }

                explanation = super().__call__(
                    X_eval,
                    groups=resolved,
                    n_resamples=n_resamples,
                    n_resamples_batch=n_resamples_batch,
                    n_eval_batch=n_eval_batch,
                )
                posterior_values.append(explanation.values)
        finally:
            self._factor_posterior_ids = {}

        return UncertaintyExplanation(
            posterior_values=np.stack(posterior_values),
            feature_names=list(resolved),
            posterior_ids=posterior_ids,
        )
