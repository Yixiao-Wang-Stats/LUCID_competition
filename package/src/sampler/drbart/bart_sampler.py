import numpy as np
import pymc as pm
import pymc_bart as pmb
from pymc_bart.pymc_bart import PosteriorSampler
from ..sampler import Sampler
from scipy.special import expit, softmax

import pickle
from pathlib import Path

#  Helper function to load BART fit from saving.
def _logit_reference(n_rows: int) -> np.ndarray:
    # Give categorical BART a centered, unit-variance logit reference.
    # BART uses Y for initialization/scaling only.
    # The actual categorical labels enter through pm.Categorical(observed=y_codes).

    if n_rows < 2:
        raise ValueError("Categorical BART requires at least two training rows.")

    reference = np.linspace(-1.0, 1.0, int(n_rows))
    reference -= reference.mean()
    reference /= reference.std()

    return reference
    
def load_bart_fit(fit_dir):
        fit_dir = Path(fit_dir)
        fit_file = fit_dir / "fit.pkl"

        if not fit_file.exists():
            raise RuntimeError("fit.pkl does not exist.")

        with open(fit_file,"rb") as file:
            fit = pickle.load(file)

        histories = fit["histories"]

        if len(histories) != 1:
            raise NotImplementedError("For now BART sampling only supports one chain.")

        baseline_forest,batches = histories[0]

        posterior_sampler = PosteriorSampler.from_history(
            batches,
            baseline_forest,
            fit["m"],
            fit["n_outputs"],
        )

        return posterior_sampler,fit

class BartSampler(Sampler):
    def __init__(
        self,
        output_dir,
        feature_types,
        m=50,
        draws=1000,
        tune=1000,
        cores=1,
        seed=2026,
        random_state=None,
        
    ):
        super().__init__(feature_types=feature_types, random_state=random_state)
        self.output_dir = Path(output_dir)
        self.m = m
        self.draws = draws
        self.tune = tune
        self.cores = cores
        self.seed = seed
        

    def _fit_feature(self, X_train, feature_idx, feature_type):
        return self._fit_model(
            X_predictors=np.delete(
                X_train, feature_idx, axis=1
            ),
            X_target=X_train[:, feature_idx],
            feature_type=feature_type,
            feature_output_dir=(
                self.output_dir / f"feature_{feature_idx}"
            ),
        )

    def _fit_model(
        self,
        X_predictors,
        X_target,
        feature_type,
        feature_output_dir,
    ):
        """Fit and save BART using explicitly supplied predictors and target."""
        X_predictors = np.asarray(X_predictors)
        X_target = np.asarray(X_target)
        if X_predictors.ndim != 2:
            raise ValueError("X_predictors must be a 2D array.")
        if X_target.ndim != 1:
            raise ValueError("X_target must be a 1D array.")
        if X_predictors.shape[0] != len(X_target):
            raise ValueError("X_predictors and X_target must have the same number of rows.")
        feature_output_dir = Path(feature_output_dir)
        feature_output_dir.mkdir(parents=True,exist_ok=True)
        categories = np.unique(X_target)
        if feature_type not in {"binary","categorical"}:
            raise ValueError("feature_type must be either 'binary' or 'categorical'.")
        if feature_type == "binary" and len(categories) != 2:
            raise ValueError("Binary BART requires exactly two categories.")
        if feature_type == "categorical" and len(categories) < 2:
            raise ValueError("Categorical BART requires at least two categories.")
        category_to_code = {category: code for code,category in enumerate(categories)}
        y_codes = np.array([category_to_code[value] for value in X_target],dtype=int)
        # how to use follow: https://www.pymc.io/projects/bart/en/stable/examples/bart_categorical_hawks.html
        with pm.Model():
            if feature_type == "binary":
                latent = pmb.BART("latent",X=X_predictors,Y=y_codes,m=self.m)
                probability = pm.math.sigmoid(latent)
                pm.Bernoulli("observed",p=probability,observed=y_codes)
            else:
                latent = pmb.BART(
                    "latent",
                    X=X_predictors,
                    Y=_logit_reference(len(X_target)),
                    m=self.m,
                    shape=(len(categories),len(X_target)),
                )
                probability = pm.math.softmax(latent,axis=0)
                pm.Categorical("observed",p=probability.T,observed=y_codes)
            pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=1,
                cores=self.cores,
                random_seed=self.seed,
                progressbar=False,
                compute_convergence_checks=False,
            )
        operation = latent.owner.op
        history = operation.all_trees[0]
        posterior_count = self.draws

        fit_artifact = {
            "histories": (history,),
            "m": self.m,
            "n_outputs": int(operation.n_outputs),
            "categories": categories,
            "feature_type": feature_type,
        }
        with open(feature_output_dir / "fit.pkl", "wb") as file:
            pickle.dump(fit_artifact, file, pickle.HIGHEST_PROTOCOL)

        (feature_output_dir / "posterior_count.txt").write_text(
            str(posterior_count)
        )
        return feature_output_dir

    def _sample_feature(
        self,
        fitted_model,
        X_eval,
        feature_idx,
        n_resamples,
        X_background,
        posterior_id=None,
    ):
        return self._sample_model(
            X_predictors=np.delete(
                np.asarray(X_eval, dtype=np.float64),
                feature_idx,
                axis=1,
            ),
            fit_dir=fitted_model,
            n_resamples=n_resamples,
            posterior_id=posterior_id,
        )

    # use X_predictors to replace the X_eval_minus_j, nothing fancy no need to read
    # We can change above sample and fit to use the batch, but we keep it here just avoid potential bugs.
    def _sample_model(
        self,
        X_predictors,
        fit_dir,
        n_resamples,
        posterior_id=None,
        row_posterior_ids=None,
    ):
        """Sample target values given explicitly supplied conditioning rows."""
        X_predictors = np.asarray(X_predictors,dtype=np.float64)
        if X_predictors.ndim != 2:
            raise ValueError("X_predictors must be a 2D array.")
        if n_resamples <= 0:
            raise ValueError("n_resamples must be positive.")
        n_eval = X_predictors.shape[0]
        posterior_sampler,fit = load_bart_fit(fit_dir)
        if row_posterior_ids is not None:
            ids = np.asarray(row_posterior_ids)
            if posterior_id is not None or n_resamples != 1:
                raise ValueError("row_posterior_ids requires one replacement per row and no fixed posterior_id.")
            if ids.shape != (n_eval,) or ids.dtype.kind not in "iu":
                raise ValueError("row_posterior_ids must contain one integer ID per conditioning row.")
            if np.any(ids < 1) or np.any(ids > posterior_sampler.n_draws):
                raise ValueError("row_posterior_ids contains an out-of-range posterior ID.")
            raw = np.empty((1,fit["n_outputs"],n_eval),dtype=float)  # (n_resamples, n_outputs, n_eval)
            for selected_id in np.unique(ids):
                rows = np.flatnonzero(ids == selected_id)  # rows sharing the selected posterior ID.
                raw[:,:,rows] = posterior_sampler.sample_posterior(
                    X_predictors[rows], [int(selected_id) - 1], None
                )  # (1, n_outputs, len(rows))
        elif posterior_id is None:
            posterior_ids = self.rng.integers(
                0,
                posterior_sampler.n_draws,
                size=n_resamples
            )
        else:
            if posterior_id < 1 or posterior_id > posterior_sampler.n_draws:
                raise ValueError("posterior_id is out of range.")
            posterior_ids = np.repeat(
                posterior_id - 1,
                n_resamples
            )
        if row_posterior_ids is None:
            raw = posterior_sampler.sample_posterior(
                X_predictors,
                posterior_ids.tolist(),
                None
            )  # (len(draw_indices), n_outputs, n_rows)
        categories = np.asarray(fit["categories"])
        raw = np.asarray(raw, dtype=float)

        if fit["feature_type"] == "binary":
            probability = expit(raw[:, 0, :])
            y_codes = self.rng.binomial(1, probability)

        elif fit["feature_type"] == "categorical":
            probability = softmax(raw.transpose(0, 2, 1), axis=-1)
            cumulative = np.cumsum(probability, axis=-1)
            cumulative[..., -1] = 1.0

            uniforms = self.rng.random(probability.shape[:-1])
            y_codes = (uniforms[..., None] > cumulative).sum(axis=-1)

        else:
            raise ValueError("Unsupported BART feature type.")

        return categories[y_codes].T