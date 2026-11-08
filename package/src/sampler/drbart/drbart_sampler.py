import os
import numpy as np
import subprocess
from pathlib import Path

from ..sampler import Sampler
from .bart_sampler import BartSampler
import tempfile
import csv



class DRBartSampler(Sampler):
    def __init__(
        self,
        output_dir,
        seed=2026,
        nburn=2000,  # default is 5000
        nsim=2000,  # default is 5000
        nthin=1,
        m_mean=200,  # default is 200
        m_var=100,  # default is 100
        variance="ux",  
        random_state=None
    ):
        super().__init__(random_state=random_state)
        self.output_dir = Path(output_dir)
        self.exchange_dir = self.output_dir / "exchange"
        self.exchange_dir.mkdir(parents=True, exist_ok=True)  # just use to avoid conflicts when running in parallel
        self.seed = seed
        self.nburn = nburn
        self.nsim = nsim
        self.nthin = nthin
        self.m_mean = m_mean
        self.m_var = m_var
        self.variance = variance
        self.fits = {}  # fits is the same usage as models in cmr just use to save
        self.feature_size = None
        self.feature_types = {}  # if feature is binary or categorical, we change to bart
        self.feature_bounds = {}
        self.feature_names = {}
        self.bart_sampler = BartSampler(
            output_dir=self.output_dir,
            m=50,
            draws=1000,
            tune=1000,
            chains=1,
            cores=1,
            seed=self.seed,
            random_state=random_state,
        )


    def load_feature_bounds(self,path,feature_names,feature_types):
        with Path(path).open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != len(feature_names) or len(feature_types) != len(feature_names):
            raise ValueError("feature_bounds.csv must have one row per feature.")
        bounds = {}
        for j,(row,name,feature_type) in enumerate(zip(rows,feature_names,feature_types)):
            if int(row["feature_idx"]) != j or row["feature_name"] != name:
                raise ValueError("feature_bounds.csv must match the feature names and order.")
            if feature_type == "continuous":
                lower,upper = float(row["lower_bound"]),float(row["upper_bound"])
                if not np.isfinite([lower,upper]).all() or lower >= upper:
                    raise ValueError(f"Invalid continuous bounds for {name}.")
                bounds[j] = (lower,upper)
            elif row["lower_bound"] or row["upper_bound"]:
                raise ValueError(f"Categorical/binary bounds must be empty: {name}.")
        self.feature_bounds,self.feature_names = bounds,dict(enumerate(feature_names))
        print("DRBART continuous bounds:",{feature_names[j]:bound for j,bound in bounds.items()},flush=True)

    def _bounds_arguments(self,fit_dir):
        feature_idx = int(Path(fit_dir).name.removeprefix("feature_"))
        if feature_idx not in self.feature_bounds:
            return []
        lower,upper = self.feature_bounds[feature_idx]
        return [str(lower),str(upper),f"{self.feature_names[feature_idx]} (feature_{feature_idx})"]

    def fit(self, X_train, feature_idx, feature_type="continuous"):
        X_train = np.asarray(X_train)

        if X_train.ndim != 2:
            raise ValueError(
                "X_train must be a 2D array."
            )

        feature_size = X_train.shape[1]

        if not 0 <= feature_idx < feature_size:
            raise ValueError(
                "feature_idx is out of range."
            )

        if self.feature_size is None:
            self.feature_size = feature_size

        elif self.feature_size != feature_size:
            raise ValueError(
                "X_train has a different number of features "
                "from previous fitted data."
            )

        if feature_type not in {"continuous","binary","categorical"}:
            raise ValueError("feature_type must be 'continuous', 'binary', or 'categorical'.")

        self.feature_types[feature_idx] = feature_type

        if feature_type in {"binary","categorical"}:
            self.bart_sampler.fit(X_train,feature_idx,feature_type=feature_type)
            self.fits[feature_idx] = self.bart_sampler.fits[feature_idx]
            self.is_fitted = True
            return self

        X_minus_j = np.delete(X_train, feature_idx, axis=1)
        X_j = X_train[:, feature_idx]
        #  below is the same as cmr
        feature_output_dir = (self.output_dir/ f"feature_{feature_idx}")
        feature_output_dir.mkdir(parents=True,exist_ok=True)
        # save data use to fit
        X_train_file = feature_output_dir / "X_train.csv"
        y_train_file = feature_output_dir / "y_train.csv"
        np.savetxt(X_train_file,X_minus_j,delimiter=",")
        np.savetxt(y_train_file,X_j,delimiter=",")

        script_path = Path(__file__).with_name("drbart_fit.R")

        subprocess.run(
            [
                "Rscript",
                str(script_path),
                str(X_train_file),
                str(y_train_file),
                str(feature_output_dir),
                str(self.seed),
                str(self.nburn),
                str(self.nsim),
                str(self.nthin),
                str(self.m_mean),
                str(self.m_var),
                str(self.variance),
            ],
            check=True
        )
        X_train_file.unlink(missing_ok=True)
        y_train_file.unlink(missing_ok=True)
        self.fits[feature_idx] = feature_output_dir
        self.is_fitted = True

        return self


    def sample(self, x, feature_idx, n_resamples, X_reference=None):
        if not self.is_fitted:
            raise RuntimeError("Sampler must be fitted before sampling.")
        if feature_idx not in self.fits:
            raise RuntimeError("The sampler has not been fitted for this feature.")
        if n_resamples <= 0:
            raise ValueError("n_resamples must be positive.")
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be a 1D feature vector.")
        if len(x) != self.feature_size:
            raise ValueError("x has the wrong number of features.")

        feature_type = self.feature_types[feature_idx]

        if feature_type in {"binary","categorical"}:
            return self.bart_sampler.sample(
                x=x,
                feature_idx=feature_idx,
                n_resamples=n_resamples,
                X_reference=X_reference,
            )
        
        x_minus_j = np.delete(x, feature_idx)
        fit_dir = self.fits[feature_idx]
        return self._sample_from_r(
            x_minus_j=x_minus_j,
            fit_dir=fit_dir,
            n_resamples=n_resamples,
            mode="lucid"
        )

    def sample_posterior_credible_interval(self, x, feature_idx, n_resamples, posterior_id):
        if not self.is_fitted:
            raise RuntimeError("Sampler must be fitted before sampling.")
        if feature_idx not in self.fits:
            raise RuntimeError("The sampler has not been fitted for this feature.")
        if n_resamples <= 0:
            raise ValueError("n_resamples must be positive.")
        if posterior_id <= 0:
            raise ValueError("posterior_id must be positive.")
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be a 1D feature vector.")
        if len(x) != self.feature_size:
            raise ValueError("x has the wrong number of features.")

        feature_type = self.feature_types[feature_idx]

        if feature_type in {"binary","categorical"}:
            return self.bart_sampler.sample_posterior_credible_interval(
                x=x,
                feature_idx=feature_idx,
                n_resamples=n_resamples,
                posterior_id=posterior_id,
            )

        x_minus_j = np.delete(x, feature_idx)
        fit_dir = self.fits[feature_idx]
        return self._sample_from_r(
            x_minus_j=x_minus_j,
            fit_dir=fit_dir,
            n_resamples=n_resamples,
            mode="credible_interval",
            posterior_id=posterior_id
        )

    # sample function from R
    def _sample_from_r(self, x_minus_j, fit_dir, n_resamples, mode, posterior_id=None):
        if mode not in {"lucid", "credible_interval"}:
            raise ValueError("mode must be either 'lucid' or 'credible_interval'.")
        if mode == "credible_interval" and posterior_id is None:
            raise ValueError("posterior_id is required for credible_interval mode.")

        x_minus_j = np.asarray(x_minus_j)
        temporary_dir = Path(tempfile.mkdtemp(dir=self.exchange_dir))

        input_file = temporary_dir / "x_eval.csv"
        output_file = temporary_dir / f"replacement_samples_{mode}.csv"

        np.savetxt(
            input_file,
            x_minus_j.reshape(1, -1),
            delimiter=",",
        )

        script_path = Path(__file__).with_name("drbart_sample.R")

        command = [
            "Rscript",
            str(script_path),
            str(fit_dir),
            str(input_file),
            str(output_file),
            str(n_resamples),
            str(mode),
            str(self._next_r_seed()),
        ]

        if mode == "credible_interval":
            command.append(str(posterior_id))
        command.extend(self._bounds_arguments(fit_dir))
        subprocess.run(command,check=True,)

        replacement_values = np.loadtxt(output_file,delimiter=",",)
        replacement_values = np.asarray(replacement_values).reshape(-1)
        return replacement_values

    # sample_batch
    def sample_batch(
        self,
        X_eval,
        feature_indices,
        n_resamples,
        X_reference=None,
    ):
        if not self.is_fitted:
            raise RuntimeError("Sampler must be fitted before sampling.")
        X_eval = np.asarray(X_eval)
        if X_eval.ndim != 2:
            raise ValueError("X_eval must be a 2D array.")
        if X_eval.shape[1] != self.feature_size:
            raise ValueError("X_eval has the wrong number of features.")
        feature_indices = np.asarray(feature_indices)
        if feature_indices.ndim != 1:
            raise ValueError("feature_indices must be a 1D array.")
        if n_resamples <= 0:
            raise ValueError("n_resamples must be positive.")
        n_eval = X_eval.shape[0]
        selected_feature_size = len(feature_indices)
        samples = np.empty(
            (n_eval, selected_feature_size, n_resamples)
        )
        for local_j,feature_idx in enumerate(feature_indices):
            if feature_idx not in self.fits:
                raise RuntimeError("The sampler has not been fitted for this feature.")

            feature_type = self.feature_types[feature_idx]

            if feature_type in {"binary","categorical"}:
                replacement_values = self.bart_sampler.sample_batch(
                    X_eval=X_eval,
                    feature_indices=np.array([feature_idx]),
                    n_resamples=n_resamples,
                    X_reference=X_reference,
                )
                samples[:,local_j,:] = replacement_values[:,0,:]
                continue

            X_eval_minus_j = np.delete(X_eval,feature_idx,axis=1)
            fit_dir = self.fits[feature_idx]
            replacement_values = self._sample_batch_from_r(
                X_eval_minus_j=X_eval_minus_j,
                fit_dir=fit_dir,
                n_resamples=n_resamples,
                mode="lucid"
            )
            samples[:,local_j,:] = replacement_values
        return samples


    def sample_batch_posterior_credible_interval(
        self,
        X_eval,
        feature_indices,
        n_resamples,
        posterior_id,
        X_reference=None,
    ):
        if not self.is_fitted:
            raise RuntimeError("Sampler must be fitted before sampling.")
        X_eval = np.asarray(X_eval)
        if X_eval.ndim != 2:
            raise ValueError("X_eval must be a 2D array.")
        if X_eval.shape[1] != self.feature_size:
            raise ValueError("X_eval has the wrong number of features.")
        feature_indices = np.asarray(feature_indices)
        if feature_indices.ndim != 1:
            raise ValueError("feature_indices must be a 1D array.")
        if n_resamples <= 0:
            raise ValueError("n_resamples must be positive.")
        if posterior_id <= 0:
            raise ValueError("posterior_id must be positive.")
        n_eval = X_eval.shape[0]
        selected_feature_size = len(feature_indices)
        samples = np.empty(
            (n_eval, selected_feature_size, n_resamples)
        )
       
        for local_j,feature_idx in enumerate(feature_indices):
            if feature_idx not in self.fits:
                raise RuntimeError("The sampler has not been fitted for this feature.")

            feature_type = self.feature_types[feature_idx]

            if feature_type in {"binary","categorical"}:
                replacement_values = self.bart_sampler.sample_batch_posterior_credible_interval(
                    X_eval=X_eval,
                    feature_indices=np.array([feature_idx]),
                    n_resamples=n_resamples,
                    posterior_id=posterior_id,
                    X_reference=X_reference,
                )
                samples[:,local_j,:] = replacement_values[:,0,:]
                continue

            X_eval_minus_j = np.delete(X_eval,feature_idx,axis=1)
            fit_dir = self.fits[feature_idx]
            replacement_values = self._sample_batch_from_r(
                X_eval_minus_j=X_eval_minus_j,
                fit_dir=fit_dir,
                n_resamples=n_resamples,
                mode="credible_interval",
                posterior_id=posterior_id
            )
            samples[:,local_j,:] = replacement_values
        return samples


    def _sample_batch_from_r(
        self,
        X_eval_minus_j,
        fit_dir,
        n_resamples,
        mode,
        posterior_id=None,
    ):
        if mode not in {"lucid", "credible_interval"}:
            raise ValueError("mode must be either 'lucid' or 'credible_interval'.")
        if mode == "credible_interval" and posterior_id is None:
            raise ValueError("posterior_id is required for credible_interval mode.")
        X_eval_minus_j = np.asarray(X_eval_minus_j)
        if X_eval_minus_j.ndim != 2:
            raise ValueError(
                "X_eval_minus_j must be a 2D array."
            )
        n_eval = X_eval_minus_j.shape[0]


        temporary_dir = Path(tempfile.mkdtemp(dir=self.exchange_dir))

        input_file = temporary_dir / "evaluation_sample.csv"
        output_file = temporary_dir / "replacement_samples.csv"
        np.savetxt(input_file, X_eval_minus_j, delimiter=",")
        script_path = Path(__file__).with_name("drbart_sample.R")
        command = [
            "Rscript",
            str(script_path),
            str(fit_dir),
            str(input_file),
            str(output_file),
            str(n_resamples),
            str(mode),
            str(self._next_r_seed()),
        ]
        if mode == "credible_interval":
            command.append(str(posterior_id))
        command.extend(self._bounds_arguments(fit_dir))
        subprocess.run(command, check=True)
        replacement_values = np.loadtxt(output_file, delimiter=",")
        replacement_values = np.asarray(replacement_values)
        if replacement_values.ndim == 1:
            replacement_values = replacement_values.reshape(1, -1)
        
        return replacement_values


    # helper function for seed in order to make sure every call produces a different seed
    def _next_r_seed(self):
        return int(self.rng.integers(0,2**31 - 1))
