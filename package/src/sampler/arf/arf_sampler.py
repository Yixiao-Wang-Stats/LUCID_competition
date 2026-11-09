import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from ..sampler import Sampler


class ARFSampler(Sampler):
    def __init__(
        self,
        output_dir,
        random_state=None,
        feature_types=None,
    ):
        super().__init__(random_state=random_state)

        self.output_dir = Path(output_dir)
        self.exchange_dir = self.output_dir / "exchange"
        self.exchange_dir.mkdir(parents=True, exist_ok=True)
        self.fit_dir = None
        self.feature_size = None
        self.feature_types = list(feature_types) if feature_types is not None else None
        self.fits = {}


    def fit(
        self,
        X_train,
        feature_idx,
        feature_type=None,
    ):
    # note that arf train a whole tree at one time then use marginal to get CDE, so we do not need feature idx in fit for real case.
        X_train = np.asarray(X_train)

        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array.")

        if not 0 <= feature_idx < X_train.shape[1]:
            raise ValueError("feature_idx is out of range.")

        if self.output_dir is None:
            raise ValueError("output_dir must be provided.")
        feature_size = X_train.shape[1]
        
        if self.feature_types is None:
            raise ValueError("feature_types must be provided for all features.")
        if len(self.feature_types) != feature_size:
            raise ValueError("feature_types must have one entry per feature.")
        if any(t not in {"binary","categorical","continuous"} for t in self.feature_types):
            raise ValueError("Feature types must be binary, categorical, or continuous.")
        if feature_type is not None and feature_type != self.feature_types[feature_idx]:
            raise ValueError("feature_type does not match feature_types[feature_idx].")
    
        if self.feature_size is None:
            self.feature_size = feature_size

        self.fit_dir = self.output_dir / f"feature_{feature_idx}"
        self.fit_dir.mkdir(parents=True, exist_ok=True)

        fit_file = self.fit_dir / "fit.rds"
        psi_file = self.fit_dir / "psi.rds"
        schema_file = self.fit_dir / "schema.rds"
        X_train_file = self.fit_dir / "X_train.csv"
        feature_types_file = self.fit_dir / "feature_types.txt"
        feature_types_file.write_text("\n".join(self.feature_types) + "\n")
        np.savetxt(
            X_train_file,
            X_train,
            delimiter=",",
        )

        script_file = Path(__file__).with_name("arf_fit.R")

        seed = (
            int(self.random_state)
            if self.random_state is not None
            else 2026
        )

        subprocess.run(
            [
                "Rscript",
                str(script_file),
                str(X_train_file),
                str(self.fit_dir),
                str(seed),
                str(feature_types_file),
            ],
            check=True,
        )

        if not fit_file.exists() or not psi_file.exists() or not schema_file.exists():
            raise RuntimeError("ARF fitting did not generate fit.rds, psi.rds, and schema.rds.")

        self.fits[feature_idx] = self.fit_dir
        self.is_fitted = True

        return self

    def sample_batch(
        self,
        X_eval,
        feature_indices,
        n_resamples,
        X_background=None,
    ):
        if not self.is_fitted:
            raise RuntimeError("Sampler must be fitted before sampling.")
        
        X_eval, feature_indices, samples = self._prepare_sampling(
            X_eval,
            feature_indices,
            n_resamples,
            n_features=self.feature_size,
        )
        n_eval = len(X_eval)
        for local_j, feature_idx in enumerate(feature_indices):
            if feature_idx not in self.fits:
                raise RuntimeError("The ARF sampler has not been fitted for this feature.")

            fit_dir = self.fits[feature_idx]

            # sample replacement values using R
            temporary_dir = Path(tempfile.mkdtemp(dir=self.exchange_dir))
            input_file = temporary_dir / "evaluation_sample.csv"
            output_file = temporary_dir / "replacement_samples.csv"

            np.savetxt(input_file, X_eval, delimiter=",")

            subprocess.run(
                [
                    "Rscript",
                    str(Path(__file__).with_name("arf_sample.R")),
                    str(fit_dir),
                    str(input_file),
                    str(output_file),
                    str(feature_idx),
                    str(n_resamples),
                    str(self._next_r_seed()),
                ],
                check=True,
            )

            replacement_values = np.asarray(np.loadtxt(output_file, delimiter=",")).reshape(-1)

            if replacement_values.size != n_eval * n_resamples:
                raise RuntimeError("R returned an unexpected number of ARF replacement samples.")

            replacement_values = replacement_values.reshape(n_eval, n_resamples)
            samples[:, local_j, :] = replacement_values

        return samples
    
