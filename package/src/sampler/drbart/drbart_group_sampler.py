# This is the sampler for drbart group feature importance
# We keep the sampling order as the feature original input order

import subprocess
import numpy as np
from pathlib import Path

from .drbart_sampler import DRBartSampler


class DRBartGroupSampler(DRBartSampler):
    def __init__(
        self,
        output_dir,
        seed=2026,
        nburn=2000,
        nsim=2000,
        nthin=1,
        m_mean=200,
        m_var=100,
        variance="ux",
        random_state=None
    ):
        super().__init__(
            output_dir=output_dir,
            seed=seed,
            nburn=nburn,
            nsim=nsim,
            nthin=nthin,
            m_mean=m_mean,
            m_var=m_var,
            variance=variance,
            random_state=random_state,
        )
        self.group_fits = {}

    
    def fit_group(self, X_train, feature_indices, feature_types=None):
        X_train = np.asarray(X_train)

        if X_train.ndim != 2:
            raise ValueError(
                "X_train must be a 2D array."
            )

        feature_size = X_train.shape[1]

        feature_indices = np.asarray(feature_indices)
        if feature_indices.ndim != 1:
            raise ValueError("feature_indices must be a 1D array.")
        if len(feature_indices) == 0:
            raise ValueError("feature_indices must not be empty.")
        if len(np.unique(feature_indices)) != len(feature_indices):
            raise ValueError("feature_indices must not contain duplicates.")
        if np.any(feature_indices < 0) or np.any(feature_indices >= feature_size):
            raise ValueError("feature_indices contains an out-of-range feature index.")

        if self.feature_size is None:
            self.feature_size = feature_size
        elif self.feature_size != feature_size:
            raise ValueError("X_train has a different number of features from previous fitted data.")

        # default version of feature type is continuous.
        if feature_types is None:
            feature_types = ["continuous"] * len(feature_indices)

        if len(feature_types) != len(feature_indices):
            raise ValueError("feature_types must have the same length as feature_indices.")

        # for feature_type in feature_types:
        #     if feature_type != "continuous":
        #         raise NotImplementedError(
        #             "DRBartGroupSampler currently supports continuous group features only."
        #         )
        for feature_type in feature_types:
            if feature_type not in {"continuous","binary","categorical"}:
                raise ValueError(
                    "feature_type must be 'continuous', 'binary', or 'categorical'."
                )

        feature_indices = [int(feature_idx) for feature_idx in feature_indices]
        group_key = tuple(feature_indices)
        group_output_dir = self.output_dir / (
            "group_" + "_".join(str(feature_idx) for feature_idx in feature_indices)
        )
        group_output_dir.mkdir(parents=True,exist_ok=True)

        outside_group_indices = [
            feature_idx
            for feature_idx in range(feature_size)
            if feature_idx not in feature_indices
        ]

        group_fits = []

        for local_j,feature_idx in enumerate(feature_indices):
            conditioning_indices = (outside_group_indices+ feature_indices[:local_j])  # remaning feature group indices for j-th feature in the group
            X_conditioning = X_train[:,conditioning_indices]
            X_j = X_train[:,feature_idx]
            feature_output_dir = (group_output_dir / f"feature_{feature_idx}")
            feature_type = feature_types[local_j]
            if feature_type in {"binary","categorical"}:
                self.bart_sampler.fit_conditioning(
                    X_conditioning=X_conditioning,
                    X_j=X_j,
                    feature_idx=feature_idx,
                    feature_type=feature_type,
                    feature_output_dir=feature_output_dir,
                )

                with open(
                    feature_output_dir / "posterior_count.txt",
                    "r",
                ) as file:
                    posterior_count = int(file.read().strip())

            else:
                feature_output_dir.mkdir(parents=True,exist_ok=True)
                X_train_file = feature_output_dir / "X_train.csv"
                y_train_file = feature_output_dir / "y_train.csv"
                np.savetxt(X_train_file,X_conditioning,delimiter=",")
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
            posterior_count = int((feature_output_dir / "posterior_count.txt").read_text())
            group_fits.append(
                {
                    "feature_idx": feature_idx,
                    "feature_type": feature_type,
                    "conditioning_indices": conditioning_indices,
                    "fit_dir": feature_output_dir,
                    "posterior_count": posterior_count,
                }
            )
        self.group_fits[group_key] = group_fits  # shape: (len(feature_indices),)
        self.is_fitted = True

        return self


    def sample_group_batch(
        self,
        evaluation_sample,
        feature_indices,
        replacement_sample_size,
        X_reference=None,
    ):
        return self._sample_group_batch(
            evaluation_sample=evaluation_sample,
            feature_indices=feature_indices,
            replacement_sample_size=replacement_sample_size,
            mode="lucid",
        )


    def _sample_group_batch(
        self,
        evaluation_sample,
        feature_indices,
        replacement_sample_size,
        mode,
        factor_posterior_ids=None,
    ):
        if not self.is_fitted:
            raise RuntimeError("Sampler must be fitted before sampling.")

        evaluation_sample = np.asarray(evaluation_sample)

        if evaluation_sample.ndim != 2:
            raise ValueError("evaluation_sample must be a 2D array.")
        if evaluation_sample.shape[1] != self.feature_size:
            raise ValueError("evaluation_sample has the wrong number of features.")

        feature_indices = np.asarray(feature_indices)
        if feature_indices.ndim != 1:
            raise ValueError("feature_indices must be a 1D array.")
        if replacement_sample_size <= 0:
            raise ValueError("replacement_sample_size must be positive.")
        if mode not in {"lucid","credible_interval"}:
            raise ValueError("mode must be either 'lucid' or 'credible_interval'.")

        feature_indices = [int(feature_idx) for feature_idx in feature_indices]
        group_key = tuple(feature_indices)

        if group_key not in self.group_fits:
            raise RuntimeError("The sampler has not been fitted for this feature group.")

        group_fits = self.group_fits[group_key]  # shape: (len(feature_indices),)

        if mode == "credible_interval":
            if factor_posterior_ids is None:
                raise ValueError("factor_posterior_ids is required for credible_interval mode.")
            if len(factor_posterior_ids) != len(feature_indices):
                raise ValueError(
                    "factor_posterior_ids must have the same length as feature_indices."
                )

        evaluation_sample_size = evaluation_sample.shape[0]
        selected_feature_size = len(feature_indices)

        samples = np.empty(
            (
                evaluation_sample_size,
                selected_feature_size,
                replacement_sample_size,
            )
        )  # save the whole replacement samples as samples for group feature 

        feature_to_local = {
            feature_idx: local_j
            for local_j,feature_idx in enumerate(feature_indices)
        }  # build a mapping from feature indices to local indices

        for local_j,factor_fit in enumerate(group_fits):
            feature_type = factor_fit["feature_type"]
            conditioning_indices = factor_fit["conditioning_indices"]  # conditioning indices (remaining features),outside the group
            fit_dir = factor_fit["fit_dir"]

            if local_j == 0:
                conditioning_data = evaluation_sample[:,conditioning_indices]  # all features reamining here 

                if feature_type in {"binary","categorical"}:
                    replacement_samples = self.bart_sampler.sample_conditioning_batch(
                        conditioning_data=conditioning_data,
                        fit_dir=fit_dir,
                        replacement_sample_size=replacement_sample_size,
                        posterior_id=(
                            factor_posterior_ids[local_j]
                            if mode == "credible_interval"
                            else None
                        ),
                    )
                else:
                    replacement_samples = self._sample_batch_from_r(
                        evaluation_sample_minus_j=conditioning_data,
                        fit_dir=fit_dir,
                        replacement_sample_size=replacement_sample_size,
                        mode=mode,
                        posterior_id=(
                            factor_posterior_ids[local_j]
                            if mode == "credible_interval"
                            else None
                        ),
                    )

                samples[:,local_j,:] = replacement_samples  # shape: (evaluation_sample_size, 1, replacement_sample_size)
                continue

            conditioning_data = np.empty(
                (
                    evaluation_sample_size,
                    replacement_sample_size,
                    len(conditioning_indices),
                )
            )  # shape: (evaluation_sample_size, replacement_sample_size, len(conditioning_indices))

            for conditioning_local_j,conditioning_idx in enumerate(conditioning_indices):
                if conditioning_idx in feature_to_local:
                    previous_local_j = feature_to_local[conditioning_idx]
                    conditioning_data[:,:,conditioning_local_j] = (
                        samples[:,previous_local_j,:]
                    )
                else:
                    conditioning_data[:,:,conditioning_local_j] = (
                        evaluation_sample[:,conditioning_idx,None]
                    )
            # So the output will look like (in 3D):
            #           -s[ repeat replacement sample_size times ]
            #       -s[ repeat replacement sample_size times ]
            #   x[x(1), ... , x(replacement_sample_size)]
            # 
            # Then for each conditioning_data[i, j, :]
            # is the obersation i; posterior j; 's remaining feature
            # | x_i_{in group}^{posterior=j}, -S_i

            conditioning_data = conditioning_data.reshape(
                evaluation_sample_size * replacement_sample_size,
                len(conditioning_indices),
            )  # We then flatten the data, so each row represents a | x_i_{in group}^{posterior=j}, -S_i

            if feature_type in {"binary","categorical"}:
                row_posterior_ids = None
                if mode == "lucid":
                    path_ids = self.rng.integers(
                        1, factor_fit["posterior_count"] + 1,
                        size=replacement_sample_size,
                    )
                    row_posterior_ids = np.tile(path_ids,evaluation_sample_size)
                replacement_samples = self.bart_sampler.sample_conditioning_batch(
                    conditioning_data=conditioning_data,
                    fit_dir=fit_dir,
                    replacement_sample_size=1,  # we have flattened the data
                    row_posterior_ids=row_posterior_ids,
                    posterior_id=(
                        factor_posterior_ids[local_j]
                        if mode == "credible_interval"
                        else None
                    ),
                )
            else:
                replacement_samples = self._sample_batch_from_r(
                    evaluation_sample_minus_j=conditioning_data,
                    fit_dir=fit_dir,
                    replacement_sample_size=1,
                    mode=mode,
                    posterior_id=(
                        factor_posterior_ids[local_j]
                        if mode == "credible_interval"
                        else None
                    ),
                )
            replacement_samples = replacement_samples.reshape(
                evaluation_sample_size,
                replacement_sample_size,
            )
            samples[:,local_j,:] = replacement_samples

        return samples

    
    def sample_group_batch_posterior_credible_interval(
        self,
        evaluation_sample,
        feature_indices,
        replacement_sample_size,
        posterior_id,
        X_reference=None,
    ):
        if posterior_id <= 0:
            raise ValueError("posterior_id must be positive.")

        feature_indices = np.asarray(feature_indices)

        if feature_indices.ndim != 1:
            raise ValueError("feature_indices must be a 1D array.")

        group_key = tuple(
            int(feature_idx)
            for feature_idx in feature_indices
        )

        if group_key not in self.group_fits:
            raise RuntimeError(
                "The sampler has not been fitted for this feature group."
            )

        group_fits = self.group_fits[group_key]
        
        # Use posterior_id to deterministically choose
        # one posterior draw for each factor.
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    self.seed,
                    posterior_id,
                ]
            )
        )
        
        factor_posterior_ids = np.array([
            rng.integers(
                1,
                factor_fit["posterior_count"] + 1,
            )
            for factor_fit in group_fits
        ])

        return self._sample_group_batch(
            evaluation_sample=evaluation_sample,
            feature_indices=feature_indices,
            replacement_sample_size=replacement_sample_size,
            mode="credible_interval",
            factor_posterior_ids=factor_posterior_ids,
        )
