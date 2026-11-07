import numpy as np
import torch
from copy import deepcopy
from sklearn.utils.validation import check_array, check_X_y
from tabpfn.architectures.shared.bar_distribution import (
    FullSupportBarDistribution,
)


class TabPFNCDF:
    """Batched native conditional CDF for local or hosted TabPFN."""

    def __init__(
        self,
        version="v2",
        backend="local",
        device="cuda",
        batch_size=256,
        cut_batch_size=256,
        **model_kwargs,
    ):
        if version not in {"v2", "v2.5", "v2.6", "v3", "v3.5"}:
            raise ValueError("Unsupported TabPFN version.")

        if backend not in {"local", "hosted"}:
            raise ValueError("backend must be 'local' or 'hosted'.")

        for size in (batch_size, cut_batch_size):
            if not isinstance(size, (int, np.integer)) or size < 1:
                raise ValueError("Batch sizes must be positive integers.")

        self.version = version
        self.backend = backend
        self.device = device
        self.batch_size = batch_size
        self.cut_batch_size = cut_batch_size
        self.model_kwargs = model_kwargs

    def fit(self, context, z):
        context, z = check_X_y(context, z, y_numeric=True)

        options = dict(n_estimators="auto", random_state=2026)

        if self.backend == "local":
            from tabpfn import TabPFNRegressor
            from tabpfn.constants import ModelVersion

            options.update(
                device=self.device,
                fit_mode="fit_with_cache",  # is much helpful when you do many times of prediction
                # n_preprocessing_jobs=2,
                show_progress_bar=False,
            )
            options.update(self.model_kwargs)

            model = TabPFNRegressor.create_default_for_version(
                ModelVersion(self.version),
                **options,
            )

        else:
            if self.version == "v2":
                raise ValueError(
                    "Use the local backend for v2 in this adapter."
                )

            from tabpfn_client import TabPFNRegressor

            options.update(
                fit_mode=(
                    "fit_preprocessors"
                    if self.version in {"v2.5", "v2.6"}  # This is from hosted-API restriction.
                    else "fit_with_cache"
                ),
                thinking_mode=False,
            )
            options.update(self.model_kwargs)

            model = TabPFNRegressor.create_default_for_version(
                self.version,
                **options,
            )

        model.fit(context, z)

        self.model_ = model
        self.n_features_in_ = context.shape[1]

        return self

    @staticmethod
    def _native_distribution(full):
        """Normalize the native output without changing its distribution."""

        logits = torch.as_tensor(
            full["logits"]
        ).detach().cpu().double()

        borders = (
            full["criterion"].borders
            if "criterion" in full
            else full["borders"]
        )
        borders = torch.as_tensor(borders).detach().cpu().double()

        if (
            logits.ndim != 2
            or borders.ndim != 1
            or len(borders) != logits.shape[1] + 1
        ):
            raise ValueError("Unexpected native distribution dimensions.")

        if (
            not torch.isfinite(borders).all()
            or torch.any(torch.diff(borders) < 0)
        ):
            raise ValueError("Invalid native distribution borders.")

        if (
            torch.isnan(logits).any()
            or torch.isposinf(logits).any()
            or not torch.isfinite(logits).any(dim=1).all()
        ):
            raise ValueError("Invalid native distribution logits.")
        widths = torch.diff(borders)

        # The outer bins parameterize half-normal tails and need positive widths.
        if widths[0] <= 0 or widths[-1] <= 0:
            raise ValueError("Tail bins must have positive width.")

        # Interior zero-width bins represent point masses. Keep them and their logits.
        criterion = FullSupportBarDistribution(borders)

        return criterion, logits


        # empty = torch.diff(borders) == 0  # if the bin is empty, so tabpfn will treat it as point mass.

        # if empty.any():
        #     if (
        #         empty[0]
        #         or empty[-1]
        #         or not torch.isneginf(logits[:, empty]).all()
        #     ):
        #         raise ValueError("Cannot remove a nonempty or tail bin.")

        #     keep = ~empty
        #     logits = logits[:, keep]
        #     borders = torch.cat((borders[:-1][keep], borders[-1:]))

        # criterion = FullSupportBarDistribution(borders) 

        # return criterion, logits

    def predict_cdf(self, context, cuts):
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit before predict_cdf.")

        context = check_array(context)
        cuts = np.asarray(cuts, dtype=np.float64)

        if context.shape[1] != self.n_features_in_:
            raise ValueError("context has the wrong number of features.")

        if cuts.ndim != 1 or np.isnan(cuts).any():
            raise ValueError("cuts must be a 1D array without NaN.")

        result = np.empty((len(context), len(cuts)), dtype=np.float64)

        if len(cuts) == 0:
            return result

        for row in range(0, len(context), self.batch_size):
            batch = context[row:row + self.batch_size]  # row for evaluation

            # One inference call per context batch, not per threshold.
            full = self.model_.predict(batch, output_type="full")  # save all denstiy information
            criterion, logits = self._native_distribution(full)  # the logits shape: (batch_size, n_bins)

            if logits.shape[0] != len(batch):
                raise ValueError("Prediction returned the wrong row count.")
            # tabpfn will deal the zero bins as point masses.
            # We should put that mass to the left side for CDF <= border.
            borders = criterion.borders
            zero_width = torch.diff(borders) == 0
            atom_locations = borders[:-1][zero_width]

            cumulative = None
            if atom_locations.numel() > 0:
                probabilities = torch.softmax(logits, dim=-1)
                cumulative = torch.cat(
                    (
                        torch.zeros_like(probabilities[:, :1]),
                        torch.cumsum(probabilities, dim=-1),
                    ),
                    dim=-1,
                )


            with torch.inference_mode():
                for column in range(0, len(cuts), self.cut_batch_size):
                    thresholds = torch.as_tensor(
                        cuts[column:column + self.cut_batch_size],
                        dtype=logits.dtype,
                    )

                    values = criterion.cdf(logits, thresholds)

                    if cumulative is not None:
                        at_atom = torch.isin(thresholds, atom_locations)

                        if at_atom.any():
                            # Count bins whose upper border is <= the requested threshold.
                            counts = torch.searchsorted(
                                borders[1:].contiguous(),
                                thresholds[at_atom].contiguous(),
                                right=True,
                            )

                            # Include all mass at equality: F(t) = P(Z <= t).
                            values[:, at_atom] = cumulative[:, counts]

                    values = values.numpy()


                    result[
                        row:row + len(batch),
                        column:column + len(thresholds),
                    ] = values

        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite TabPFN CDF values.")

        return np.clip(result, 0.0, 1.0)