import numpy as np
import torch

from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from ..sampler import Sampler


class TabPFNSampler(Sampler):
    """Conditional replacement sampling with local or hosted TabPFN."""

    def __init__(
        self, version="v3.5", backend="local", device="auto",
        batch_size=256, random_state=None, **model_kwargs,
    ):
        super().__init__(random_state=random_state)
        self.version = version
        self.backend = backend
        self.device = device
        self.batch_size = batch_size
        self.model_kwargs = dict(model_kwargs)

    def fit(self, X_train, feature_idx, feature_type="continuous"):
        X = np.asarray(X_train, dtype=float)
        self.feature_idx = feature_idx
        self.feature_type = feature_type
        context = np.delete(X, feature_idx, axis=1)
        target = X[:, feature_idx]
        self._constant = None

        if feature_type not in {"continuous", "binary", "categorical"}:
            raise ValueError("Unknown feature type.")

        categorical = feature_type != "continuous"
        if categorical:
            self.classes_, target = np.unique(target, return_inverse=True)
            if len(self.classes_) == 1:
                self._constant = self.classes_[0]
                self.is_fitted = True
                return self

        options = {"random_state": self.random_state, **self.model_kwargs}
        version = self.version

        if self.backend == "local":
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            from tabpfn.constants import ModelVersion

            version = ModelVersion(version)
            options.setdefault("device", self.device)
            options.setdefault("fit_mode", "fit_with_cache")
            options.setdefault("show_progress_bar", False)
        elif self.backend == "hosted":
            from tabpfn_client import TabPFNClassifier, TabPFNRegressor

            options.setdefault(
                "fit_mode",
                "fit_preprocessors"
                if self.version in {"v2.5", "v2.6"} else "fit_with_cache",
            )
            options.setdefault("thinking_mode", False)
        else:
            raise ValueError("backend must be 'local' or 'hosted'.")

        model_class = TabPFNClassifier if categorical else TabPFNRegressor
        self.model_ = model_class.create_default_for_version(
            version=version, **options
        )
        self.model_.fit(context, target)
        self.is_fitted = True
        return self

    def sample_batch(
        self, X_eval, feature_indices,
        n_resamples, X_reference=None,
    ):
        if not self.is_fitted:
            raise RuntimeError("Call fit before sampling.")
        if list(feature_indices) != [self.feature_idx]:
            raise ValueError("Requested feature differs from fitted feature.")

        X = np.asarray(X_eval, dtype=float)
        context = np.delete(X, self.feature_idx, axis=1)
        samples = np.empty((len(X), 1, n_resamples))

        if self._constant is not None:
            samples.fill(self._constant)
            return samples

        for start in range(0, len(X), self.batch_size):
            stop = min(start + self.batch_size, len(X))
            batch = context[start:stop]

            if self.feature_type != "continuous":
                probabilities = np.asarray(self.model_.predict_proba(batch))
                labels = np.asarray(self.model_.classes_, dtype=int)
                classes = self.classes_[labels]
                for row, p in enumerate(probabilities):
                    samples[start + row, 0] = self.rng.choice(
                        classes, size=n_resamples, p=p / p.sum()
                    )
            else:
                full = self.model_.predict(batch, output_type="full")
                logits = torch.as_tensor(full["logits"]).detach().cpu().double()
                borders = (
                    full["criterion"].borders
                    if "criterion" in full else full["borders"]
                )
                borders = torch.as_tensor(borders).detach().cpu().double()
                criterion = FullSupportBarDistribution(borders)
                samples[start:stop, 0] = self._draw(
                    criterion, logits, n_resamples
                )

        return samples  # (n_eval, n_selected_features, n_resamples)

    def sample(
        self, x, feature_idx, n_resamples, X_reference=None,
    ):
        return self.sample_batch(
            np.asarray(x)[None, :], [feature_idx],
            n_resamples, X_reference,
        )[0, 0]

    def _draw(self, criterion, logits, count):
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(self._next_r_seed())
            bins = torch.multinomial(logits.softmax(-1), count, replacement=True)
            widths = criterion.bucket_widths
            values = (
                criterion.borders[bins]
                + widths[bins] * torch.rand(bins.shape, dtype=logits.dtype)
            )
            lower = criterion.borders[1] - (
                criterion.halfnormal_with_p_weight_before(widths[0])
                .sample(bins.shape)
            )
            upper = criterion.borders[-2] + (
                criterion.halfnormal_with_p_weight_before(widths[-1])
                .sample(bins.shape)
            )
            values = torch.where(bins == 0, lower, values)
            values = torch.where(bins == logits.shape[-1] - 1, upper, values)
            return values.numpy()