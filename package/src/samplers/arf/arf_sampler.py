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
        self._joint_fit = None

    def _fit_feature(self, X_train, feature_idx, feature_type):
        if self.output_dir is None:
            raise ValueError("output_dir must be provided.")
        if self._joint_fit is not None:
            return self._joint_fit
        fit_dir = self.output_dir / "joint"
        fit_dir.mkdir(parents=True, exist_ok=True)

        fit_file = fit_dir / "fit.rds"
        psi_file = fit_dir / "psi.rds"
        schema_file = fit_dir / "schema.rds"
        X_train_file = fit_dir / "X_train.csv"
        feature_types_file = fit_dir / "feature_types.txt"
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
                str(fit_dir),
                str(seed),
                str(feature_types_file),
            ],
            check=True,
        )

        if not fit_file.exists() or not psi_file.exists() or not schema_file.exists():
            raise RuntimeError("ARF fitting did not generate fit.rds, psi.rds, and schema.rds.")

        self._joint_fit = fit_dir
        return fit_dir

    def _sample_feature(
        self,
        fitted_model,
        X_eval,
        feature_idx,
        n_resamples,
        X_background,
    ):
        n_eval = len(X_eval)

        temporary_dir = Path(
            tempfile.mkdtemp(
                prefix=f"feature_{feature_idx}_",
                dir=self.exchange_dir,
            )
        )
        input_file = temporary_dir / "evaluation_sample.csv"
        output_file = temporary_dir / "replacement_samples.csv"

        np.savetxt(input_file, X_eval, delimiter=",")

        subprocess.run(
            [
                "Rscript",
                str(Path(__file__).with_name("arf_sample.R")),
                str(fitted_model),
                str(input_file),
                str(output_file),
                str(feature_idx),
                str(n_resamples),
                str(self._next_r_seed()),
            ],
            check=True,
        )

        replacement_values = np.asarray(
            np.loadtxt(output_file, delimiter=",")
        ).reshape(-1)

        if replacement_values.size != n_eval * n_resamples:
            raise RuntimeError("R returned an unexpected number of ARF replacement samples.")

        return replacement_values.reshape(n_eval, n_resamples)
    
