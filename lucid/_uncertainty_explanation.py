import numpy as np

from ._explanation import Explanation


class UncertaintyExplanation(Explanation):
    """LUCID values and credible intervals from conditional-model posterior draws."""

    def __init__(
        self,
        posterior_values,
        feature_names=None,
        posterior_ids=None,
    ):
        self.posterior_values = np.asarray(posterior_values, dtype=float)

        if (
            self.posterior_values.ndim != 3
            or self.posterior_values.shape[0] < 2
            or 0 in self.posterior_values.shape[1:]
        ):
            raise ValueError(
                "posterior_values must have shape "
                "(n_posterior_draws >= 2, n_eval, n_features_or_groups)."
            )
        if not np.isfinite(self.posterior_values).all():
            raise ValueError("posterior_values must be finite.")

        self.posterior_ids = posterior_ids

        # Expose posterior-mean values through the same interface as ordinary explanations.
        super().__init__(
            values=self.posterior_values.mean(axis=0),
            feature_names=feature_names,
        )

    def credible_interval(self, level=0.95):
        """Return lower and upper bounds for each observation and feature/group."""
        return self._interval(self.posterior_values, level)

    def global_credible_interval(self, level=0.95):
        """Return bounds after averaging observations within each posterior draw."""
        # Average evaluation observations within each posterior draw before taking global
        # interval quantiles.
        global_draws = self.posterior_values.mean(axis=1)
        return self._interval(global_draws, level)

    @staticmethod
    def _interval(values, level):
        if not 0 < level < 1:
            raise ValueError("level must be between zero and one.")

        # Split the remaining probability equally between the two tails of the posterior
        # draws.
        tail = (1 - level) / 2
        return np.quantile(values, [tail, 1 - tail], axis=0)