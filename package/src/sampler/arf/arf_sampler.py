import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from ..sampler import Sampler


class ARFSampler(Sampler):
    def __init__(
        self,
        random_state=None,
        output_dir=None,
        feature_types=None,
    ):
        super().__init__(random_state=random_state)

        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.exchange_dir = Path(os.environ.get("LUCID_EXCHANGE_DIR", "results/tmp"))
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

        # self.fit_dir = self.output_dir
        # fit seperately for now. Avoiding saving conflicts.
        self.fit_dir = self.output_dir / f"feature_{feature_idx}"
        self.fit_dir.mkdir(parents=True, exist_ok=True)

        fit_file = self.fit_dir / "fit.rds"
        psi_file = self.fit_dir / "psi.rds"

        # # ARF models the joint distribution of all features.
        # # Therefore all features share the same fitted ARF.
        # if fit_file.exists() and psi_file.exists():
        #     self.is_fitted = True
        #     return self

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

        command = [
            "Rscript",
            str(script_file),
            str(X_train_file),
            str(self.fit_dir),
            str(seed),
            str(feature_types_file),
        ]

        subprocess.run(
            command,
            check=True,
        )

        schema_file = self.fit_dir / "schema.rds"
        if not fit_file.exists() or not psi_file.exists() or not schema_file.exists():
            raise RuntimeError("ARF fitting did not generate fit.rds, psi.rds, and schema.rds.")

        self.fits[feature_idx] = self.fit_dir
        self.is_fitted = True

        return self


    def sample(
        self,
        x,
        feature_idx,
        replacement_sample_size,
        X_reference=None,
    ):
        if self.fit_dir is None:
            if self.output_dir is None:
                raise ValueError(
                    "output_dir must be provided."
                )

            self.fit_dir = self.output_dir

        if not (self.fit_dir / "fit.rds").exists():
            raise RuntimeError(
                "ARFSampler must be fitted before sampling."
            )

        x = np.asarray(x)

        if x.ndim != 1:
            raise ValueError(
                "x must be a 1D feature vector."
            )

        if not 0 <= feature_idx < x.shape[0]:
            raise ValueError(
                "feature_idx is out of range."
            )

        if replacement_sample_size <= 0:
            raise ValueError(
                "replacement_sample_size must be positive."
            )
        
        if feature_idx not in self.fits:
            raise RuntimeError(
                "The ARF sampler has not been fitted for this feature."
            )

        fit_dir = self.fits[feature_idx]

        replacement = self.sample_from_r(
            x=x,
            feature_idx=feature_idx,
            fit_dir=fit_dir,
            replacement_sample_size=replacement_sample_size,
        )

        if replacement.shape[0] != replacement_sample_size:
            raise RuntimeError(
                "ARF returned unexpected number of samples."
            )
        return replacement
    

    def sample_from_r(
        self,
        x,
        feature_idx,
        fit_dir,
        replacement_sample_size,
    ):
        script_file = Path(__file__).with_name("arf_sample.R")
        seed = int(self.rng.integers(0,np.iinfo(np.int32).max,))
        temporary_dir = Path(tempfile.mkdtemp(dir=self.exchange_dir))

        input_file = temporary_dir / "x.csv"
        output_file = temporary_dir / "replacement.csv"
        np.savetxt(
            input_file,
            x.reshape(1,-1),
            delimiter=",",
        )

        command = [
            "Rscript",
            str(script_file),
            str(fit_dir),
            str(input_file),
            str(output_file),
            str(feature_idx),
            str(replacement_sample_size),
            str(seed),
        ]

        subprocess.run(command,check=True,)
        if not output_file.exists():
            raise RuntimeError(
                "ARF sampling did not generate output."
            )

        replacement = np.asarray(np.loadtxt(output_file,delimiter=",",)).reshape(-1)

        return replacement


    def sample_batch(
        self,
        evaluation_sample,
        feature_indices,
        replacement_sample_size,
        X_reference=None,
    ):
        if not self.is_fitted:
            raise RuntimeError(
                "Sampler must be fitted before sampling."
            )

        evaluation_sample = np.asarray(evaluation_sample)

        if evaluation_sample.ndim != 2:
            raise ValueError(
                "evaluation_sample must be a 2D array."
            )

        if evaluation_sample.shape[1] != self.feature_size:
            raise ValueError(
                "evaluation_sample has the wrong number of features."
            )

        feature_indices = np.asarray(feature_indices,dtype=int,)

        if feature_indices.ndim != 1:
            raise ValueError(
                "feature_indices must be a 1D array."
            )

        if replacement_sample_size <= 0:
            raise ValueError(
                "replacement_sample_size must be positive."
            )

        evaluation_sample_size = evaluation_sample.shape[0]
        selected_feature_size = len(feature_indices)

        samples = np.empty(
            (
                evaluation_sample_size,
                selected_feature_size,
                replacement_sample_size,
            )
        )

        for local_j, feature_idx in enumerate(feature_indices):
            if feature_idx not in self.fits:
                raise RuntimeError(
                    "The ARF sampler has not been fitted "
                    "for this feature."
                )

            fit_dir = self.fits[feature_idx]

            replacement_samples = self._sample_batch_from_r(
                evaluation_sample=evaluation_sample,
                feature_idx=feature_idx,
                fit_dir=fit_dir,
                replacement_sample_size=replacement_sample_size,
            )

            samples[:, local_j, :] = replacement_samples

        return samples
    

    def _sample_batch_from_r(
        self,
        evaluation_sample,
        feature_idx,
        fit_dir,
        replacement_sample_size,
    ):
        evaluation_sample = np.asarray(evaluation_sample)

        if evaluation_sample.ndim != 2:
            raise ValueError(
                "evaluation_sample must be a 2D array."
            )

        evaluation_sample_size = evaluation_sample.shape[0]
        fit_dir = Path(fit_dir)

        temporary_dir = Path(tempfile.mkdtemp(dir=self.exchange_dir))

        input_file = temporary_dir / "evaluation_sample.csv"
        output_file = temporary_dir / "replacement_samples.csv"

        np.savetxt(
            input_file,
            evaluation_sample,
            delimiter=",",
        )

        script_path = Path(__file__).with_name(
            "arf_sample.R"
        )

        command = [
            "Rscript",
            str(script_path),
            str(fit_dir),
            str(input_file),
            str(output_file),
            str(feature_idx),
            str(replacement_sample_size),
            str(self._next_r_seed()),
        ]

        subprocess.run(
            command,
            check=True,
        )

        replacement_samples = np.loadtxt(
            output_file,
            delimiter=",",
        )
        replacement_samples = np.asarray(
            replacement_samples
        ).reshape(-1)

        expected_size = (
            evaluation_sample_size
            * replacement_sample_size
        )

        if replacement_samples.size != expected_size:
            raise RuntimeError(
                "R returned an unexpected number "
                "of ARF replacement samples."
            )

        replacement_samples = replacement_samples.reshape(
            evaluation_sample_size,
            replacement_sample_size,
        )

        return replacement_samples