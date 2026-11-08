import numpy as np
import pandas as pd

from LUCID_calculation.lucid import (
    lucid_single_observation_single_feature,
)
from pathlib import Path
from tempfile import TemporaryDirectory 

from ._model import ModelAdapter
from ._explanation import Explanation


class Explainer:
    """Compute LUCID values for a fitted predictive model."""

    def __init__(
        self,
        model,
        X_background,
        cde="permutation",
        random_state=None,
        feature_types=None,
        cde_kwargs=None,
        output_dir=None,
    ):
        
        self.model = ModelAdapter(model)

        self._uses_dataframe = isinstance(X_background, pd.DataFrame)
        self.feature_names = (
            list(X_background.columns)
            if self._uses_dataframe
            else list(range(np.asarray(X_background).shape[1]))
        )
        self.X_background = np.asarray(X_background, dtype=float)

        self.cde = cde
        self.random_state = random_state
        self.cde_kwargs = dict(cde_kwargs or {})
        # Numeric features are continuous unless explicitly specified.
        self.feature_types = (
            list(feature_types)
            if feature_types is not None
            else ["continuous"] * len(self.feature_names)
        )

        self._temporary_directory = None

        if output_dir is None:
            self._temporary_directory = TemporaryDirectory(prefix="lucid_")
            self.output_dir = Path(self._temporary_directory.name)
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._samplers = {}


    def _new_sampler(self):
        kwargs = {
            "random_state": self.random_state,
            **self.cde_kwargs,
        }

        if self.cde == "permutation":
            from sampler.permutation.permutation_sampler import PermutationSampler

            return PermutationSampler(**kwargs)

        if self.cde in {"cmr", "gcmr"}:
            from predictive_model.xgboost_model import XGBoostModel

            kwargs.setdefault("predictive_model_class", XGBoostModel)
            kwargs.setdefault(
                "predictive_model_kwargs",
                {"random_state": self.random_state},
            )

            if self.cde == "cmr":
                from sampler.cmr.cmr_sampler import CMRSampler

                return CMRSampler(**kwargs)

            from sampler.gcmr.gcmr_sampler import GCMRSampler

            return GCMRSampler(**kwargs)

        if self.cde in {"drbart", "arf", "cspfi"}:
            workspace = self.output_dir / self.cde
            workspace.mkdir(parents=True, exist_ok=True)
            kwargs["output_dir"] = workspace

            if self.cde == "drbart":
                from sampler.drbart.drbart_sampler import DRBartSampler

                kwargs.setdefault(
                    "seed",
                    self.random_state if self.random_state is not None else 2026,
                )
                sampler = DRBartSampler(**kwargs)

            elif self.cde == "arf":
                from sampler.arf.arf_sampler import ARFSampler

                kwargs["feature_types"] = self.feature_types
                sampler = ARFSampler(**kwargs)

            else:
                from sampler.cspfi.cspfi_sampler import CSPFISampler

                kwargs["feature_types"] = self.feature_types
                kwargs.setdefault(
                    "seed",
                    self.random_state if self.random_state is not None else 2026,
                )
                sampler = CSPFISampler(**kwargs)

            return sampler

        if self.cde == "tabpfn":
            from sampler.tabpfn.tabpfn_sampler import TabPFNSampler

            return TabPFNSampler(**kwargs)

        raise ValueError(
            "cde must be permutation, drbart, arf, cspfi, cmr, gcmr, or tabpfn."
        )

    def _get_sampler(self, feature_idx):
        if feature_idx not in self._samplers:
            sampler = self._new_sampler()

            sampler.fit(
                self.X_background,
                feature_idx=feature_idx,
                feature_type=self.feature_types[feature_idx],
            )

            self._samplers[feature_idx] = sampler

        return self._samplers[feature_idx]

    def _predict(self, X):
        # Restore column names for models that expect a DataFrame.
        if self._uses_dataframe:
            X = pd.DataFrame(X, columns=self.feature_names)

        predictions = self.model(X)

        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions[:, 0]

        if predictions.shape != (len(X),):
            raise ValueError(
                "The model must return one scalar prediction per row."
            )

        return predictions

    def __call__(
        self,
        X_eval,
        features=None,
        n_resamples=1000,
        n_resamples_batch=100,  # this is the resample batch size.
        n_eval_batch=64,   # Observations per sampling call.
    ):
        if isinstance(X_eval, pd.DataFrame) and self._uses_dataframe:
            X_eval = X_eval.loc[:, self.feature_names]

        X_eval = np.asarray(X_eval, dtype=float)

        if X_eval.ndim != 2 or X_eval.shape[1] != len(self.feature_names):
            raise ValueError(
                "X_eval must have the same features as X_background."
            )

        if features is None:
            feature_indices = list(range(len(self.feature_names)))
        else:
            feature_indices = [
                self.feature_names.index(feature)
                if isinstance(feature, str)
                else int(feature)
                for feature in features
            ]

        reference_predictions = self._predict(X_eval)
        values = np.empty((len(X_eval), len(feature_indices)))

        for column, feature_idx in enumerate(feature_indices):
            sampler = self._get_sampler(feature_idx)

            for start in range(0, len(X_eval), n_eval_batch):
                stop = min(start + n_eval_batch, len(X_eval))
                X_eval_batch = X_eval[start:stop]

                replacement_values = sampler.sample_batch(
                    X_eval=X_eval_batch,
                    feature_indices=[feature_idx],
                    n_resamples=n_resamples,
                    X_reference=self.X_background,
                )

                for local_row, x in enumerate(X_eval_batch):
                    row = start + local_row

                    values[row, column] = (
                        lucid_single_observation_single_feature(
                            x=x,
                            y=reference_predictions[row],
                            replacement_sample=replacement_values[local_row, 0],
                            n_resamples_batch=n_resamples_batch,
                            loss_type="squared_error",
                            model=self._predict,
                            feature_idx=feature_idx,
                            baseline_loss=0.0,
                        )
                    )

        return Explanation(
            values=values,
            feature_names=[
                self.feature_names[j] for j in feature_indices
            ],
        )

    def close(self):
        """Remove automatically created working files."""
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None
            self._samplers.clear()