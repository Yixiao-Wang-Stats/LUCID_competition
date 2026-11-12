import numpy as np
import torch

from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from ..sampler import Sampler
from threading import Lock

class _SharedTabPFNModel:
    """Retain pretrained networks across feature contexts."""

    def __init__(self):
        self._lock = Lock()
        self._specs = {}

    def get_specs(self, model_class, version, options):
        from tabpfn import ModelSpecs

        # Older versions use separate regression/classification models.
        multitask = version.value in {"v3.5", "v3.5-fast"}
        key = (version.value, None if multitask else model_class)

        # Serialize first-time loading so concurrent feature tasks reuse the same retained
        # network specifications.
        with self._lock:
            if key not in self._specs:
                loader = model_class.create_default_for_version(
                    version=version, **options
                )
                inference_config = loader.get_inference_config()

                if len(loader.devices_) != 1:
                    raise ValueError("Shared feature execution currently requires one device.")

                for network in loader.models_:
                    network.to(loader.devices_[0])
                    if loader.forced_inference_dtype_ is not None:
                        network.to(dtype=loader.forced_inference_dtype_)
                    network.eval()

                # Older regressors need this distribution explicitly.
                norm_criterion = getattr(
                    loader, "znorm_space_bardist_", None
                )

                # Pass network objects and their inference settings to separate
                # feature-specific estimators.
                self._specs[key] = [
                    ModelSpecs(
                        model=network,
                        architecture_config=config,
                        inference_config=inference_config,
                        norm_criterion=norm_criterion,
                    )
                    for network, config in zip(
                        loader.models_, loader.configs_
                    )
                ]

            return self._specs[key]

            
class TabPFNSampler(Sampler):
    """Conditional replacement sampling with local or hosted TabPFN."""

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
        super().__init__(random_state=random_state, feature_types=feature_types)
        self.version = version
        self.backend = backend
        self.device = device
        self.model_kwargs = dict(model_kwargs)
        # Keep a local network owner only for local inference; hosted estimators use the
        # external service.
        self._shared_model = (
            _shared_model
            if _shared_model is not None
            else _SharedTabPFNModel()
        ) if backend == "local" else None

    def _fit_feature(self, X_train, feature_idx, feature_type):
        X = np.asarray(X_train, dtype=float)
        return self._fit_model(
            X_predictors=np.delete(X, feature_idx, axis=1),
            X_target=X[:, feature_idx],
            feature_type=feature_type,
        )

    def _fit_model(self, X_predictors, X_target, feature_type):
        """Fit one conditional context using the retained pretrained network."""
        context = np.asarray(X_predictors, dtype=float)
        target = np.asarray(X_target)

        classes = None
        categorical = feature_type != "continuous"

        # Encode discrete target values as class indices; a constant target needs no
        # classifier.
        if categorical:
            classes, target = np.unique(
                target, return_inverse=True
            )
            if len(classes) == 1:
                return {
                    "model": None,
                    "classes": classes,
                    "constant": classes[0],
                }

        options = {
            "random_state": self.random_state,
            **self.model_kwargs,
        }
        version = self.version

        # Choose the installed local estimator or hosted client and its supported
        # context-caching mode.
        if self.backend == "local":
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            from tabpfn.constants import ModelVersion

            version = ModelVersion(version)
            options.setdefault("device", self.device)
            options.setdefault("fit_mode", "fit_with_cache")
            options.setdefault("show_progress_bar", False)
            options.setdefault("n_jobs", 1)

        elif self.backend == "hosted":
            from tabpfn_client import (
                TabPFNClassifier,
                TabPFNRegressor,
            )

            options.setdefault(
                "fit_mode",
                "fit_preprocessors"
                if self.version in {"v2.5", "v2.6"}
                else "fit_with_cache",
            )
            options.setdefault("thinking_mode", False)

        else:
            raise ValueError("backend must be 'local' or 'hosted'.")

        model_class = (
            TabPFNClassifier if categorical else TabPFNRegressor
        )

        if self.backend == "local":
            # Initialize this feature context with retained local weights rather than
            # loading another network.
            options["model_path"] = self._shared_model.get_specs(
                model_class, version, options
            )

        model = model_class.create_default_for_version(
            version=version,
            **options,
        )
        # Fit preprocessing and the conditional context for this target feature.
        model.fit(context, target)

        return {
            "model": model,
            "classes": classes,
            "constant": None,
        }

    def _sample_feature(
        self,
        fitted_model,
        X_eval,
        feature_idx,
        n_resamples,
        X_background,
    ):
        X_eval = np.asarray(X_eval, dtype=float)
        return self._sample_model(
            fitted_model,
            X_predictors=np.delete(X_eval, feature_idx, axis=1),
            feature_type=self.feature_types[feature_idx],
            n_resamples=n_resamples,
        )

    def _sample_model(
        self,
        fitted_model,
        X_predictors,
        feature_type,
        n_resamples,
    ):
        """Draw target values conditional on the supplied predictor rows."""
        X_eval_minus_j = np.asarray(X_predictors, dtype=float)

        if fitted_model["constant"] is not None:
            return np.full(
                (len(X_eval_minus_j), n_resamples),
                fitted_model["constant"],
                dtype=float,
            )  # For constant features in binary or categorical feature cases, return the constant value for all resamples.

        model = fitted_model["model"]
        replacement_values = np.empty(
            (len(X_eval_minus_j), n_resamples)
        )

        if len(X_eval_minus_j) == 0:
            return replacement_values

        # Sample original category values using the classifier's conditional probability
        # mass.
        if feature_type != "continuous":
            probabilities = np.asarray(
                model.predict_proba(X_eval_minus_j)
            )  # shape: (len(X_eval), len(classes))
            labels = np.asarray(model.classes_, dtype=int)
            classes = fitted_model["classes"][labels]

            for row, p in enumerate(probabilities):
                replacement_values[row] = (
                    self.rng.choice(
                        classes,
                        size=n_resamples,
                        p=p / p.sum(),
                    )
                )

        else:
            # Read the full predictive distribution, rather than sampling around a point
            # prediction.
            full = model.predict(
                X_eval_minus_j, output_type="full"
            )
            logits = (
                torch.as_tensor(full["logits"])
                .detach().cpu().double()
            )
            # Accept distribution borders returned directly or attached to TabPFN's
            # criterion.
            borders = (
                full["criterion"].borders
                if "criterion" in full
                else full["borders"]
            )
            borders = (
                torch.as_tensor(borders)
                .detach().cpu().double()
            )
            criterion = FullSupportBarDistribution(borders)  # the bin distribution

            replacement_values[:] = self._draw(
                criterion,
                logits,
                n_resamples,
            )

        return replacement_values

    def _draw(self, criterion, logits, count):
        # Use a sampler-derived seed without modifying PyTorch's global random stream.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._next_r_seed())

        with torch.inference_mode():  # turn off gradient tracking
            # Draw distribution buckets for each evaluation observation from their predicted
            # probabilities.
            bins = torch.multinomial(
                logits.softmax(-1),
                count,
                replacement=True,
                generator=generator,
            )

            # Generate uniform values within interior buckets.
            widths = criterion.bucket_widths
            values = (
                criterion.borders[bins]
                + widths[bins] * torch.rand(
                    bins.shape,
                    dtype=logits.dtype,
                    generator=generator,
                )
            )

            # The two outer buckets use half-normal tails instead of bounded uniform draws.
            lower_scale = (
                criterion.halfnormal_with_p_weight_before(widths[0]).scale
            )
            upper_scale = (
                criterion.halfnormal_with_p_weight_before(widths[-1]).scale
            )

            lower = criterion.borders[1] - lower_scale * torch.randn(
                bins.shape,
                dtype=logits.dtype,
                generator=generator,
            ).abs()

            upper = criterion.borders[-2] + upper_scale * torch.randn(
                bins.shape,
                dtype=logits.dtype,
                generator=generator,
            ).abs()

            # Substitute tail draws only where an outer bucket was selected.
            values = torch.where(bins == 0, lower, values)
            values = torch.where(
                bins == logits.shape[-1] - 1, upper, values
            )
            return values.numpy()
