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
        feature_types,
        random_state=None,
    ):
        super().__init__(feature_types=feature_types, random_state=random_state)

        # File workspace.
        self.output_dir, self.exchange_dir = self._prepare_workspace(
            output_dir
        )
        
        self.fit_dir = None
        self.fits = {}


    def fit(
        self,
        X_train,
        feature_idx,
        feature_type=None,
    ):
        # note that arf train a whole tree at one time then use marginal to get CDE, so we do not need feature idx in fit for real case.
        X_train = self._prepare_fitting(
            X_train,
            feature_idx,
            feature_type=feature_type,
        )

        if self.output_dir is None:
            raise ValueError("output_dir must be provided.")

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
    
