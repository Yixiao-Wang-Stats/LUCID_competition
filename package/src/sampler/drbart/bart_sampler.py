import numpy as np
import pymc as pm
import pymc_bart as pmb
from pymc_bart.pymc_bart import PosteriorSampler
from ..sampler import Sampler

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
        chains=1,
        cores=1,
        seed=2026,
        random_state=None,
        
    ):
        super().__init__(feature_types=feature_types, random_state=random_state)
        self.output_dir = Path(output_dir)
        self.m = m
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.cores = cores
        self.seed = seed
        self.fits = {}


    def fit(self, X_train, feature_idx, feature_type=None):
        #  feature type: contains both binary and categorical features 
        X_train = self._prepare_fitting(X_train,feature_idx,feature_type=feature_type,)
        feature_type = self.feature_types[feature_idx]

        X_minus_j = np.delete(X_train, feature_idx, axis=1)
        X_j = X_train[:, feature_idx]  
        #  below is the same as other methods
        feature_output_dir = self.output_dir / f"feature_{feature_idx}"
        feature_output_dir.mkdir(parents=True,exist_ok=True)
        categories = np.unique(X_j)

        if feature_type not in {"binary","categorical"}:
            raise ValueError("feature_type must be either 'binary' or 'categorical'.")
        if feature_type == "binary" and len(categories) != 2:
            raise ValueError("Binary BART requires exactly two categories.")
        if feature_type == "categorical" and len(categories) < 2:
            raise ValueError("Categorical BART requires at least two categories.")

        # If feature is not number then coding by myself
        category_to_code = {category: code for code, category in enumerate(categories)}
        y_codes = np.array([category_to_code[value] for value in X_j],dtype=int)  # shape (n_eval, )

        with pm.Model():
            X_data = pm.Data("X_data",X_minus_j)
            
            if feature_type == "binary":
                latent = pmb.BART("latent",X=X_data,Y=y_codes,m=self.m)
                probability = pm.math.sigmoid(latent)
                pm.Bernoulli("observed",p=probability,observed=y_codes)

            else:
                latent = pmb.BART("latent", X=X_data, Y=_logit_reference(len(X_train)), m=self.m, shape=(len(categories), len(X_train)))
                probability = pm.math.softmax(latent, axis=0)
                pm.Categorical("observed", p=probability.T, observed=y_codes)
           
            pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=self.cores,
                random_seed=self.seed,
                progressbar=False,
                compute_convergence_checks=False,
            )

        operation = latent.owner.op
        # operation.all_trees[chain_idx] is the chain_idx-th posterior record
        # it countains baseline_forest (intial state of all the trees, m trees here); batches (updated forest)
        histories = tuple(operation.all_trees[chain_idx] for chain_idx in range(len(operation.all_trees)))

        fit_artifact = {
            "histories": histories,
            "m": int(operation.m),
            "n_outputs": int(operation.n_outputs),
            "categories": categories,
            "feature_type": feature_type,
        }

        fit_file = feature_output_dir / "fit.pkl"
        with open(fit_file,"wb") as file:
            pickle.dump(fit_artifact,file,pickle.HIGHEST_PROTOCOL)    # save fit_artifact

        baseline_forest,batches = histories[0]
        posterior_sampler = PosteriorSampler.from_history(batches,baseline_forest,int(operation.m),int(operation.n_outputs))
        posterior_count = posterior_sampler.n_draws

        with open(feature_output_dir / "posterior_count.txt","w") as file:
            file.write(str(posterior_count))

        self.fits[feature_idx] = feature_output_dir
        self.is_fitted = True
        return self

    # this is for drbart we copy the fit, just add the X_conditioning to replace x_minus_j, not fancy no need to read.
    def fit_conditioning(
        self,
        X_conditioning,
        X_j,
        feature_idx,
        feature_type,
        feature_output_dir,
    ):
        X_conditioning = np.asarray(X_conditioning)
        X_j = np.asarray(X_j)
        if X_conditioning.ndim != 2:
            raise ValueError("X_conditioning must be a 2D array.")
        if X_j.ndim != 1:
            raise ValueError("X_j must be a 1D array.")
        if X_conditioning.shape[0] != len(X_j):
            raise ValueError("X_conditioning and X_j must have the same number of rows.")
        feature_output_dir = Path(feature_output_dir)
        feature_output_dir.mkdir(parents=True,exist_ok=True)
        categories = np.unique(X_j)
        if feature_type not in {"binary","categorical"}:
            raise ValueError("feature_type must be either 'binary' or 'categorical'.")
        if feature_type == "binary" and len(categories) != 2:
            raise ValueError("Binary BART requires exactly two categories.")
        if feature_type == "categorical" and len(categories) < 2:
            raise ValueError("Categorical BART requires at least two categories.")
        category_to_code = {category: code for code,category in enumerate(categories)}
        y_codes = np.array([category_to_code[value] for value in X_j],dtype=int)
        with pm.Model():
            X_data = pm.Data("X_data",X_conditioning)
            if feature_type == "binary":
                latent = pmb.BART("latent",X=X_data,Y=y_codes,m=self.m)
                probability = pm.math.sigmoid(latent)
                pm.Bernoulli("observed",p=probability,observed=y_codes)
            else:
                latent = pmb.BART(
                    "latent",
                    X=X_data,
                    Y=_logit_reference(len(X_j)),
                    m=self.m,
                    shape=(len(categories),len(X_j)),
                )
                probability = pm.math.softmax(latent,axis=0)
                pm.Categorical("observed",p=probability.T,observed=y_codes)
            pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=self.cores,
                random_seed=self.seed,
                progressbar=False,
                compute_convergence_checks=False,
            )
        operation = latent.owner.op
        histories = tuple(
            operation.all_trees[chain_idx]
            for chain_idx in range(len(operation.all_trees))
        )
        fit_artifact = {
            "histories": histories,
            "m": int(operation.m),
            "n_outputs": int(operation.n_outputs),
            "categories": categories,
            "feature_type": feature_type,
        }
        fit_file = feature_output_dir / "fit.pkl"
        with open(fit_file,"wb") as file:
            pickle.dump(fit_artifact,file,pickle.HIGHEST_PROTOCOL)
        baseline_forest,batches = histories[0]
        posterior_sampler = PosteriorSampler.from_history(
            batches,
            baseline_forest,
            int(operation.m),
            int(operation.n_outputs),
        )
        posterior_count = posterior_sampler.n_draws
        with open(feature_output_dir / "posterior_count.txt","w") as file:
            file.write(str(posterior_count))
        self.fits[feature_idx] = feature_output_dir
        self.is_fitted = True
        return self

        
    def sample(self, x, feature_idx, n_resamples, X_background=None):
        if not self.is_fitted:
            raise RuntimeError("Sampler must be fitted before sampling.")
        if feature_idx not in self.fits:
            raise RuntimeError("The sampler has not been fitted for this feature.")
        if n_resamples <= 0:
            raise ValueError("n_resamples must be positive.")
        x = np.asarray(x,dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("x must be a 1D feature vector.")
        if len(x) != self.feature_size:
            raise ValueError("x has the wrong number of features.")

        x_minus_j = np.delete(x,feature_idx).reshape(1,-1)
        posterior_sampler,fit = load_bart_fit(self.fits[feature_idx])
        posterior_ids = self.rng.integers(0,posterior_sampler.n_draws,size=n_resamples)

        raw = posterior_sampler.sample_posterior(x_minus_j,posterior_ids.tolist(),None)  # (X you want to predict, using the posterior samples, excluded feature)
        # output shape (posterior draws,n_output, observations); n_output is 1 here, while we deal with single observation so the observations dimension is also 1

        categories = np.asarray(fit["categories"])  # note that for categorical feature, the shap of raw is (n_resamples, n_categories, 1)

        if fit["feature_type"] == "binary":
            latent = np.asarray(raw,dtype=float)[:,0,0]
            probability = np.empty_like(latent,dtype=float)
            positive = latent >= 0
            probability[positive] = 1.0 / (1.0 + np.exp(-latent[positive]))
            exponentiated = np.exp(latent[~positive])
            probability[~positive] = exponentiated / (1.0 + exponentiated)
            y_codes = self.rng.binomial(1,probability)

        elif fit["feature_type"] == "categorical":
            latent = np.asarray(raw,dtype=float)[:,:,0]
            shifted = latent - np.max(latent,axis=1,keepdims=True)
            probability = np.exp(shifted)
            probability = probability / probability.sum(axis=1,keepdims=True)
            y_codes = np.array([
                self.rng.choice(len(categories),p=probability[draw_idx])
                for draw_idx in range(n_resamples)
            ])

        else:
            raise ValueError("Unsupported BART feature type.")
        replacement_values = categories[y_codes]

        return replacement_values


    def sample_posterior_credible_interval(self, x, feature_idx, n_resamples, posterior_id):
        # fixed posterior sample for credible interval
        if not self.is_fitted:
            raise RuntimeError("Sampler must be fitted before sampling.")
        if feature_idx not in self.fits:
            raise RuntimeError("The sampler has not been fitted for this feature.")
        if n_resamples <= 0:
            raise ValueError("n_resamples must be positive.")
        x = np.asarray(x,dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("x must be a 1D feature vector.")
        if len(x) != self.feature_size:
            raise ValueError("x has the wrong number of features.")

        x_minus_j = np.delete(x,feature_idx).reshape(1,-1)
        posterior_sampler,fit = load_bart_fit(self.fits[feature_idx])

        if posterior_id < 1 or posterior_id > posterior_sampler.n_draws:
            raise ValueError("posterior_id is out of range.")

        posterior_ids = np.repeat(posterior_id - 1,n_resamples)  # here is the difference, for each posterior id, we sample n_resamples times

        raw = posterior_sampler.sample_posterior(x_minus_j,posterior_ids.tolist(),None)  # shape (number of posterior id * n_resamples, 1, 1)

        categories = np.asarray(fit["categories"])  # note that for categorical feature, the shap of raw is (n_resamples, n_categories, 1)

        if fit["feature_type"] == "binary":
            latent = np.asarray(raw,dtype=float)[:,0,0]
            probability = np.empty_like(latent,dtype=float)
            positive = latent >= 0
            probability[positive] = 1.0 / (1.0 + np.exp(-latent[positive]))
            exponentiated = np.exp(latent[~positive])
            probability[~positive] = exponentiated / (1.0 + exponentiated)
            y_codes = self.rng.binomial(1,probability)

        elif fit["feature_type"] == "categorical":
            latent = np.asarray(raw,dtype=float)[:,:,0]
            shifted = latent - np.max(latent,axis=1,keepdims=True)
            probability = np.exp(shifted)
            probability = probability / probability.sum(axis=1,keepdims=True)
            y_codes = np.array([
                self.rng.choice(len(categories),p=probability[draw_idx])
                for draw_idx in range(n_resamples)
            ])

        else:
            raise ValueError("Unsupported BART feature type.")
        replacement_values = categories[y_codes]

        return replacement_values


    def sample_batch(
        self,
        X_eval,
        feature_indices,
        n_resamples,
        X_background=None,
    ):
        if not self.is_fitted:
            raise RuntimeError("Sampler must be fitted before sampling.")
        X_eval = np.asarray(X_eval,dtype=np.float64)
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
        samples = np.empty((n_eval,selected_feature_size,n_resamples))

        for local_j,feature_idx in enumerate(feature_indices):
            if feature_idx not in self.fits:
                raise RuntimeError("The sampler has not been fitted for this feature.")

            X_eval_minus_j = np.delete(X_eval,feature_idx,axis=1)
            posterior_sampler,fit = load_bart_fit(self.fits[feature_idx])

            posterior_ids = self.rng.integers(
                0,
                posterior_sampler.n_draws,
                size=n_resamples
            )

            raw = posterior_sampler.sample_posterior(
                X_eval_minus_j,
                posterior_ids.tolist(),
                None
            )  # (n_resamples, category, n_eval)

            categories = np.asarray(fit["categories"])

            if fit["feature_type"] == "binary":
                latent = np.asarray(raw,dtype=float)[:,0,:]
                probability = np.empty_like(latent,dtype=float)
                positive = latent >= 0
                probability[positive] = 1.0 / (1.0 + np.exp(-latent[positive]))
                exponentiated = np.exp(latent[~positive])
                probability[~positive] = exponentiated / (1.0 + exponentiated)
                y_codes = self.rng.binomial(1,probability)

            elif fit["feature_type"] == "categorical":
                latent = np.asarray(raw,dtype=float).transpose(0,2,1)  # (n_resamples, n_eval, category)
                shifted = latent - np.max(latent,axis=2,keepdims=True)
                probability = np.exp(shifted)
                probability = probability / probability.sum(axis=2,keepdims=True)
                uniforms = self.rng.random((n_resamples,n_eval))
                cumulative = np.cumsum(probability,axis=2)
                cumulative[:,:,-1] = 1.0
                y_codes = np.sum(uniforms[:,:,None] > cumulative,axis=2)
            
            else:
                raise ValueError("Unsupported BART feature type.")
            replacement_values = categories[y_codes]
            samples[:,local_j,:] = replacement_values.T  

        return samples



    def sample_batch_posterior_credible_interval(
        self,
        X_eval,
        feature_indices,
        n_resamples,
        posterior_id,
        X_background=None,
    ):
        if not self.is_fitted:
            raise RuntimeError("Sampler must be fitted before sampling.")
        X_eval = np.asarray(X_eval,dtype=np.float64)
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
        samples = np.empty((n_eval,selected_feature_size,n_resamples))

        for local_j,feature_idx in enumerate(feature_indices):
            if feature_idx not in self.fits:
                raise RuntimeError("The sampler has not been fitted for this feature.")

            X_eval_minus_j = np.delete(X_eval,feature_idx,axis=1)
            posterior_sampler,fit = load_bart_fit(self.fits[feature_idx])

            if posterior_id < 1 or posterior_id > posterior_sampler.n_draws:
                raise ValueError("posterior_id is out of range.")

            posterior_ids = np.repeat(posterior_id - 1,n_resamples)  # same thing as single sampling did

            raw = posterior_sampler.sample_posterior(
                X_eval_minus_j,
                posterior_ids.tolist(),
                None
            )

            categories = np.asarray(fit["categories"])

            if fit["feature_type"] == "binary":
                latent = np.asarray(raw,dtype=float)[:,0,:]
                probability = np.empty_like(latent,dtype=float)
                positive = latent >= 0
                probability[positive] = 1.0 / (1.0 + np.exp(-latent[positive]))
                exponentiated = np.exp(latent[~positive])
                probability[~positive] = exponentiated / (1.0 + exponentiated)
                y_codes = self.rng.binomial(1,probability)

            elif fit["feature_type"] == "categorical":
                latent = np.asarray(raw,dtype=float).transpose(0,2,1)
                shifted = latent - np.max(latent,axis=2,keepdims=True)
                probability = np.exp(shifted)
                probability = probability / probability.sum(axis=2,keepdims=True)
                uniforms = self.rng.random((n_resamples,n_eval))
                cumulative = np.cumsum(probability,axis=2)
                cumulative[:,:,-1] = 1.0
                y_codes = np.sum(uniforms[:,:,None] > cumulative,axis=2)

            else:
                raise ValueError("Unsupported BART feature type.")

            replacement_values = categories[y_codes]
            samples[:,local_j,:] = replacement_values.T

        return samples


    # use conditioning_data to replace the X_eval_minus_j, nothing fancy no need to read
    # We can change above sample and fit to use the batch, but we keep it here just avoid potential bugs.
    def sample_conditioning_batch(
        self,
        conditioning_data,
        fit_dir,
        n_resamples,
        posterior_id=None,
        row_posterior_ids=None,
    ):
        conditioning_data = np.asarray(conditioning_data,dtype=np.float64)
        if conditioning_data.ndim != 2:
            raise ValueError("conditioning_data must be a 2D array.")
        if n_resamples <= 0:
            raise ValueError("n_resamples must be positive.")
        n_eval = conditioning_data.shape[0]
        posterior_sampler,fit = load_bart_fit(fit_dir)
        if row_posterior_ids is not None:
            ids = np.asarray(row_posterior_ids)
            if posterior_id is not None or n_resamples != 1:
                raise ValueError("row_posterior_ids requires one replacement per row and no fixed posterior_id.")
            if ids.shape != (n_eval,) or ids.dtype.kind not in "iu":
                raise ValueError("row_posterior_ids must contain one integer ID per conditioning row.")
            if np.any(ids < 1) or np.any(ids > posterior_sampler.n_draws):
                raise ValueError("row_posterior_ids contains an out-of-range posterior ID.")
            raw = np.empty((1,fit["n_outputs"],n_eval),dtype=float)
            for selected_id in np.unique(ids):
                rows = np.flatnonzero(ids == selected_id)
                raw[:,:,rows] = posterior_sampler.sample_posterior(
                    conditioning_data[rows], [int(selected_id) - 1], None
                )
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
                conditioning_data,
                posterior_ids.tolist(),
                None
            )
        categories = np.asarray(fit["categories"])
        if fit["feature_type"] == "binary":
            latent = np.asarray(raw,dtype=float)[:,0,:]
            probability = np.empty_like(latent,dtype=float)
            positive = latent >= 0
            probability[positive] = 1.0 / (1.0 + np.exp(-latent[positive]))
            exponentiated = np.exp(latent[~positive])
            probability[~positive] = exponentiated / (1.0 + exponentiated)
            y_codes = self.rng.binomial(1,probability)
        elif fit["feature_type"] == "categorical":
            latent = np.asarray(raw,dtype=float).transpose(0,2,1)
            shifted = latent - np.max(latent,axis=2,keepdims=True)
            probability = np.exp(shifted)
            probability = probability / probability.sum(axis=2,keepdims=True)
            uniforms = self.rng.random(
                (n_resamples,n_eval)
            )
            cumulative = np.cumsum(probability,axis=2)
            cumulative[:,:,-1] = 1.0
            y_codes = np.sum(
                uniforms[:,:,None] > cumulative,
                axis=2
            )
        else:
            raise ValueError("Unsupported BART feature type.")
        replacement_values = categories[y_codes]
        return replacement_values.T
