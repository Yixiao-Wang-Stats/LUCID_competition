import os
import numpy as np
import subprocess
from pathlib import Path

from ..sampler import Sampler
from .bart_sampler import BartSampler
import tempfile


class DRBartSampler(Sampler):
    def __init__(
        self,
        output_dir,
        feature_types,
        seed=2026,
        nburn=2000,  # default is 5000
        nsim=2000,  # default is 5000
        nthin=1,
        m_mean=200,  # default is 200
        m_var=100,  # default is 100
        variance="ux",  
        random_state=None,
    ):
        super().__init__(feature_types=feature_types,random_state=random_state)
        
        # Method configuration.
        self.seed = seed
        self.nburn = nburn
        self.nsim = nsim
        self.nthin = nthin
        self.m_mean = m_mean
        self.m_var = m_var
        self.variance = variance
        
        # File workspace.
        self.output_dir, self.exchange_dir = self._prepare_workspace(
            output_dir
        )

        # Conditional models for binary and categorical targets.
        self.bart_sampler = BartSampler(
            output_dir=self.output_dir,
            feature_types=self.feature_types,
            m=50,
            draws=1000,
            tune=1000,
            cores=1,
            seed=self.seed,
            random_state=random_state,
        )

    def _fit_feature(self, X_train, feature_idx, feature_type):
        if feature_type in {"binary", "categorical"}:
            self.bart_sampler.fit(
                X_train,
                feature_idx,
                feature_type=feature_type,
            )
            return self.bart_sampler.fits[feature_idx]

        return self._fit_model(
            X_predictors=np.delete(X_train, feature_idx, axis=1),
            X_target=X_train[:, feature_idx],
            feature_output_dir=(
                self.output_dir / f"feature_{feature_idx}"
            ),
        )

    def _fit_model(
        self,
        X_predictors,
        X_target,
        feature_output_dir,
    ):
        """Fit and save continuous DRBART using prepared predictors and target."""
        feature_output_dir = Path(feature_output_dir)
        feature_output_dir.mkdir(parents=True, exist_ok=True)

        X_train_file = feature_output_dir / "X_train.csv"
        y_train_file = feature_output_dir / "y_train.csv"
        np.savetxt(X_train_file, X_predictors, delimiter=",")
        np.savetxt(y_train_file, X_target, delimiter=",")

        subprocess.run(
            [
                "Rscript",
                str(Path(__file__).with_name("drbart_fit.R")),
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
            check=True,
        )

        X_train_file.unlink(missing_ok=True)
        y_train_file.unlink(missing_ok=True)
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
        feature_type = self.feature_types[feature_idx]

        if feature_type in {"binary", "categorical"}:
            replacement_values = self.bart_sampler.sample(
                X_eval=X_eval,
                feature_indices=[feature_idx],
                n_resamples=n_resamples,
                X_background=X_background,
                posterior_id=posterior_id,
            )
            return replacement_values[:, 0, :]

        X_eval_minus_j = np.delete(
            X_eval, feature_idx, axis=1
        )
        # resolve the feature bounds for the continuous feature
        bounds = self._resolve_feature_bounds(
            X_background=np.asarray(X_background)[:, [feature_idx]],
            feature_types=["continuous"],
        )[0]  # (lower, upper)

        return self._sample_from_r(
            X_eval_minus_j=X_eval_minus_j,
            fit_dir=fitted_model,
            n_resamples=n_resamples,
            mode=(
                "lucid"
                if posterior_id is None
                else "credible_interval"
            ),
            posterior_id=posterior_id,
            bounds=bounds,
        )


    def sample_posterior_credible_interval(
        self,
        X_eval,
        feature_indices,
        n_resamples,
        posterior_id,
        X_background=None,
    ):
        if posterior_id <= 0:
            raise ValueError("posterior_id must be positive.")

        return self.sample(
            X_eval,
            feature_indices,
            n_resamples,
            X_background=X_background,
            posterior_id=posterior_id,
        )


    def _sample_from_r(
        self,
        X_eval_minus_j,
        fit_dir,
        n_resamples,
        mode,
        posterior_id=None,
        bounds=None,
    ):
        if mode not in {"lucid", "credible_interval"}:
            raise ValueError("mode must be either 'lucid' or 'credible_interval'.")
        if mode == "credible_interval" and posterior_id is None:
            raise ValueError("posterior_id is required for credible_interval mode.")
        X_eval_minus_j = np.asarray(X_eval_minus_j)
        if X_eval_minus_j.ndim != 2:
            raise ValueError("X_eval_minus_j must be a 2D array.")
        n_eval = X_eval_minus_j.shape[0]

        temporary_dir = Path(
            tempfile.mkdtemp(
                prefix=f"{Path(fit_dir).name}_",
                dir=self.exchange_dir,
            )
        )

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
        if bounds is not None:
            lower, upper = bounds
            command.extend([
                str(lower),
                str(upper),
                str(Path(fit_dir).name),
            ])
        subprocess.run(command, check=True)
        replacement_values = np.asarray(
            np.loadtxt(output_file, delimiter=",")
        ).reshape(n_eval, n_resamples)

        return replacement_values
