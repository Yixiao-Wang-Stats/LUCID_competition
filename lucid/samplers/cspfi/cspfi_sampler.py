import os
import subprocess
import tempfile
from pathlib import Path
import numpy as np

from ..sampler import Sampler

class CSPFISampler(Sampler):
    def __init__(
        self,
        output_dir,
        feature_types,
        seed=2026,
        random_state=None,
        fit_script=None,
        sample_script=None,
    ):
        super().__init__(feature_types=feature_types, random_state=random_state)
        
        # File workspace.
        self.output_dir, self.exchange_dir = self._prepare_workspace(
            output_dir
        )

        self.seed = seed
        self.fit_script = Path(fit_script) if fit_script is not None else Path(__file__).with_name("cspfi_fit.R")
        self.sample_script = Path(sample_script) if sample_script is not None else Path(__file__).with_name("cspfi_sample.R")


    def _fit_feature(self, X_train, feature_idx, feature_type):
        # Fit one tree partition per target feature and keep its feature-specific model
        # files.
        fit_dir = self.output_dir / f"feature_{feature_idx}"
        fit_dir.mkdir(parents=True, exist_ok=True)

        X_train_file = fit_dir / "X_train.csv"
        feature_types_file = fit_dir / "feature_types.txt"

        np.savetxt(X_train_file, X_train, delimiter=",")
        feature_types_file.write_text("\n".join(self.feature_types))

        subprocess.run(
            [
                "Rscript",
                str(self.fit_script),
                str(X_train_file),
                str(fit_dir),
                str(self.seed),
                str(feature_types_file),
                str(feature_idx),
            ],
            check=True,
        )

        return fit_dir
    
    def _sample_feature(
        self,
        fitted_model,
        X_eval,
        feature_idx,
        n_resamples,
        X_background,
    ):
        X_background = np.asarray(X_background)
        if X_background.ndim != 2:
            raise ValueError("X_background must be a 2D array.")

        temporary_dir = Path(
            tempfile.mkdtemp(
                prefix=f"feature_{feature_idx}_",
                dir=self.exchange_dir,
            )
        )
        X_eval_file = temporary_dir / "X_eval.csv"
        X_background_file = temporary_dir / "X_background.csv"
        output_file = temporary_dir / "samples.csv"

        # Send evaluation observations and the background replacement pool as separate
        # datasets.
        np.savetxt(X_eval_file, X_eval, delimiter=",")
        np.savetxt(X_background_file, X_background, delimiter=",")

        # Use the saved tree to resample background values from each evaluation
        # observation's leaf.
        subprocess.run(
            [
                "Rscript",
                str(self.sample_script),
                str(fitted_model / "fit.rds"),
                str(X_eval_file),
                str(X_background_file),
                str(output_file),
                str(n_resamples),
                str(self._next_r_seed()),
            ],
            check=True,
        )

        replacement_values = np.loadtxt(
            output_file, delimiter=","
        )
        # Restore (evaluation observations, replacement draws), including single-row CSV
        # results.
        return np.asarray(replacement_values).reshape(
            len(X_eval), n_resamples
        )
        