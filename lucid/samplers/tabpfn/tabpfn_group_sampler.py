"""Joint conditional replacement sampling with local or hosted TabPFN."""

import numpy as np

from ..group_sampler import GroupSampler
from .tabpfn_sampler import TabPFNSampler


class TabPFNGroupSampler(GroupSampler):
    """Reuse one TabPFN network owner across ordered group contexts.

    Each factor has a separate fitted context. Pretrained networks are shared
    for local inference; hosted inference retains separate API estimators.
    If a group includes every feature, its first, unconditional factor uses
    the empirical marginal distribution of that feature in the training data.
    """

    def __init__(
        self,
        feature_types,
        version="v3.5",
        backend="local",
        device="auto",
        random_state=None,
        _shared_model=None,
        **model_kwargs,
    ):
        super().__init__(feature_types, random_state=random_state)
        self.tabpfn_sampler = TabPFNSampler(
            feature_types=self.feature_types,
            version=version,
            backend=backend,
            device=device,
            random_state=random_state,
            _shared_model=_shared_model,
            **model_kwargs,
        )

    def _fit_group_feature(
        self, X_predictors, X_target, feature_idx, feature_type, group_key
    ):
        # TabPFN requires predictors; an unconditional factor needs only a pool.
        if X_predictors.shape[1] == 0:
            return {"marginal_values": np.asarray(X_target).copy()}
        return self.tabpfn_sampler._fit_model(
            X_predictors=X_predictors,
            X_target=X_target,
            feature_type=feature_type,
        )

    def _sample_group_feature(
        self, factor_fit, X_predictors, n_resamples, X_background
    ):
        fitted_model = factor_fit["fitted_model"]
        if "marginal_values" in fitted_model:
            return self.tabpfn_sampler.rng.choice(
                fitted_model["marginal_values"],
                size=(len(X_predictors), n_resamples),
                replace=True,
            )
        return self.tabpfn_sampler._sample_model(
            fitted_model=fitted_model,
            X_predictors=X_predictors,
            feature_type=factor_fit["feature_type"],
            n_resamples=n_resamples,
        )
