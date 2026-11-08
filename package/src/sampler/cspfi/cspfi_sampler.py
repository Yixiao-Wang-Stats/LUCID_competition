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
        super().__init__(random_state=random_state)

        self.output_dir = Path(output_dir)
        self.exchange_dir = self.output_dir / "exchange"
        self.exchange_dir.mkdir(parents=True, exist_ok=True)
        self.feature_types = feature_types
        self.seed = seed
        self.fit_script = Path(fit_script) if fit_script is not None else Path(__file__).with_name("cspfi_fit.R")
        self.sample_script = Path(sample_script) if sample_script is not None else Path(__file__).with_name("cspfi_sample.R")
        self.feature_idx = None
        self.fit_dir = None

    def fit(
        self,
        X_train,
        feature_idx,
        feature_type=None,
    ):
        X_train = np.asarray(X_train)
        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array.")
        if not 0 <= feature_idx < X_train.shape[1]:
            raise ValueError("feature_idx is out of range.")
        if len(self.feature_types) != X_train.shape[1]:
            raise ValueError("feature_types must have one entry per feature.")
        self.feature_idx = feature_idx
        self.fit_dir = self.output_dir / f"feature_{feature_idx}"
        self.fit_dir.mkdir(parents=True,exist_ok=True)
        fit_file = self.fit_dir / "fit.rds"
        X_train_file = self.fit_dir / "X_train.csv"
        feature_types_file = self.fit_dir / "feature_types.txt"
        np.savetxt(X_train_file,X_train,delimiter=",")
        feature_types_file.write_text("\n".join(self.feature_types))
        subprocess.run(
            [
                "Rscript",
                str(self.fit_script),
                str(X_train_file),
                str(self.fit_dir),
                str(self.seed),
                str(feature_types_file),
                str(feature_idx),
            ],
            check=True,
        )
        return self

    def sample(
        self,
        x,
        feature_idx,
        n_resamples,
        X_reference=None,
    ):
        return self.sample_batch(
            X_eval=np.asarray(x)[None, :],
            feature_indices=[feature_idx],
            n_resamples=n_resamples,
            X_reference=X_reference,
        )[0, 0]

    
    def sample_batch(
        self,
        X_eval,
        feature_indices,
        n_resamples,
        X_reference=None,
    ):
        if self.fit_dir is None:
            raise RuntimeError("fit must be called before sample_batch.")
        feature_indices = np.asarray(feature_indices, dtype=int)

        if feature_indices.shape != (1,):
            raise ValueError(
                "CSPFISampler supports one fitted feature at a time."
            )

        if feature_indices[0] != self.feature_idx:
            raise ValueError(
                "Requested feature does not match the fitted feature."
            )

        X = np.asarray(X_eval)
        X_reference = np.asarray(X_reference)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X_reference.ndim != 2:
            raise ValueError("X_reference must be a 2D array.")
        temporary_dir = Path(tempfile.mkdtemp(dir=self.exchange_dir))
        X_file = temporary_dir / "X.csv"
        X_reference_file = temporary_dir / "X_reference.csv"
        output_file = temporary_dir / "samples.csv"
        np.savetxt(X_file,X,delimiter=",")
        np.savetxt(X_reference_file,X_reference,delimiter=",")
        subprocess.run(
            [
                "Rscript",
                str(self.sample_script),
                str(self.fit_dir / "fit.rds"),
                str(X_file),
                str(X_reference_file),
                str(output_file),
                str(n_resamples),
                str(self._next_r_seed()),
            ],
            check=True,
        )
        replacement_values = np.loadtxt(output_file,delimiter=",")
        return np.asarray(replacement_values).reshape(len(X),1,n_resamples)