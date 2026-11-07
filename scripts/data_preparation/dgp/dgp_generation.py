"""Five baseline DGPs and their known prediction/reduced models.

Run from LUCID_supplementary with ``python -m scripts.data_preparation.dgp.dgp_generation``.
Prepared XGBoost artifacts are produced by the prepare_lucid pipeline;
oracle models are saved separately as small JSON specifications.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri

from scripts.data_preparation.data_prepare_lucid import prepare_lucid


DGP_NAMES = (
    "regime_dependent",
    "strong_conditional_dependence",
    "heteroskedastic_conditional_dependence",
    "interaction_conditional_dependence",
    "extreme_tail",
)
FEATURE_COUNTS = dict(zip(DGP_NAMES, (3, 2, 2, 3, 2)))


class KnownDGPModel:
    """Known response function and conditional prediction moments."""

    def __init__(self, name, irrelevant_features=0):
        if name not in DGP_NAMES:
            raise ValueError(f"Unknown DGP: {name}")
        self.name = name
        if not isinstance(irrelevant_features,(int,np.integer)) or irrelevant_features < 0:
            raise ValueError("irrelevant_features must be a nonnegative integer.")
        self.irrelevant_features = int(irrelevant_features)
        self.feature_size = FEATURE_COUNTS[name]+self.irrelevant_features

    def _matrix(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.feature_size:
            raise ValueError(f"Expected a 2D array with {self.feature_size} features.")
        return X

    def predict(self, X):
        X = self._matrix(X)
        if self.name == "regime_dependent":
            return np.where(X[:, 2] > 0, X[:, 0], X[:, 1])
        if self.name == "extreme_tail":
            return X[:, 0] + 20 * (np.abs(X[:, 1]) > ndtri(.99))
        return X[:, 0]

    __call__ = predict

    def conditional_moments(self, X, feature_idx):
        X = self._matrix(X)
        if not isinstance(feature_idx, (int, np.integer)) or not 0 <= feature_idx < self.feature_size:
            raise ValueError("feature_idx is out of range.")
        mean = self.predict(X).copy()
        variance = np.zeros(len(X))
        if feature_idx >= FEATURE_COUNTS[self.name]:
            return mean,variance
        if self.name == "regime_dependent":
            if feature_idx in (0, 1):
                active = (X[:, 2] > 0) if feature_idx == 0 else (X[:, 2] <= 0)
                mean[active] = 0
                variance[active] = 1
            else:
                mean = (X[:, 0] + X[:, 1]) / 2
                variance = (X[:, 0] - X[:, 1]) ** 2 / 4
        elif self.name == "extreme_tail":
            if feature_idx == 0:
                mean = 20. * (np.abs(X[:, 1]) > ndtri(.99))
                variance = 1. + (X[:, 1] > 0).astype(float)
            else:
                # Given either sign of X2, P(|X2|>c)=.02. X1 depends
                # only on that sign, so conditioning on X1 keeps this .02.
                mean = X[:, 0] + .4
                variance[:] = 400 * .02 * .98
        elif feature_idx == 0:
            if self.name == "strong_conditional_dependence":
                mean = .9 * X[:, 1]
                variance[:] = 1 - .9 ** 2
            elif self.name == "heteroskedastic_conditional_dependence":
                mean[:] = 0
                variance = X[:, 1] ** 2
            else:
                mean = np.where(np.logical_xor(X[:, 1] > 0, X[:, 2] > 0), 2., -2.)
                variance[:] = 1
        return mean, variance

    def reduced_model(self, feature_idx):
        if not isinstance(feature_idx, (int, np.integer)) or not 0 <= feature_idx < self.feature_size:
            raise ValueError("feature_idx is out of range.")
        return KnownReducedModel(self, feature_idx)

    def save(self, path):
        spec = {"name": self.name}
        if self.irrelevant_features:
            spec["irrelevant_features"] = self.irrelevant_features
        _save_spec(path, spec)

    @classmethod
    def load(cls, path):
        spec = json.loads(Path(path).read_text())
        return cls(spec["name"],spec.get("irrelevant_features",0))


class KnownReducedModel:
    """E[g(X) | X_-j], taking only the retained columns as input."""

    def __init__(self, model, feature_idx):
        self.model = model
        self.feature_idx = feature_idx

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.model.feature_size - 1:
            raise ValueError("Reduced model input has the wrong number of features.")
        full = np.insert(X, self.feature_idx, 0., axis=1)
        return self.model.conditional_moments(full, self.feature_idx)[0]

    def save(self, path):
        spec = {"name": self.model.name, "feature_idx": int(self.feature_idx)}
        if self.model.irrelevant_features:
            spec["irrelevant_features"] = self.model.irrelevant_features
        _save_spec(path, spec)

    @classmethod
    def load(cls, path):
        spec = json.loads(Path(path).read_text())
        return KnownDGPModel(spec["name"],spec.get("irrelevant_features",0)).reduced_model(spec["feature_idx"])


def _save_spec(path, spec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2) + "\n")


def generate_dgp(name, n=5000, seed=2026):
    """Generate features and responses for a baseline DGP with a fixed RNG seed."""
    model = KnownDGPModel(name)
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer of at least 2.")
    rng = np.random.default_rng(seed)
    if name == "regime_dependent":
        X = rng.normal(size=(n, 3))
    elif name == "strong_conditional_dependence":
        X = rng.multivariate_normal(np.zeros(2), [[1., .9], [.9, 1.]], size=n)
    elif name == "heteroskedastic_conditional_dependence":
        x2 = rng.normal(size=n)
        X = np.column_stack((rng.normal(size=n) * np.abs(x2), x2))
    elif name == "interaction_conditional_dependence":
        x2, x3 = rng.normal(size=n), rng.normal(size=n)
        mu = 4 * np.logical_xor(x2 > 0, x3 > 0).astype(float) - 2
        X = np.column_stack((rng.normal(mu, 1.), x2, x3))
    else:
        x2 = rng.normal(size=n)
        variance = 1. + (x2 > 0).astype(float)
        x1 = rng.normal(size=n) * np.sqrt(variance)
        X = np.column_stack((x1, x2))
    frame = pd.DataFrame(X, columns=[f"X{i + 1}" for i in range(model.feature_size)])
    frame["target"] = model.predict(X)
    return frame


def prepare_dgp(name, output_dir, n=5000, random_state=2026, n_splits=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / f"{name}.csv"
    generated = not input_path.is_file()
    if generated:
        generate_dgp(name, n=n, seed=random_state).to_csv(input_path, index=False)
    prepared_dir = output_dir / "prepared"
    if generated or not (all((prepared_dir / file).is_file() for file in ("feature_types.csv", "fold_indices.csv"))
            and all((prepared_dir / f"fold_{k}" / file).is_file() for k in range(n_splits)
                    for file in ("train.csv", "outcome_reference.csv", "prediction_reference.csv", "predictive_model.json"))):
        prepare_lucid(input_path=input_path, output_dir=prepared_dir, target_name="target",
                     feature_types=["continuous"] * FEATURE_COUNTS[name],
                     n_splits=n_splits, random_state=random_state)
    model = KnownDGPModel(name)
    for fold_idx in range(n_splits):
        fold_dir = prepared_dir / f"fold_{fold_idx}"
        model.save(fold_dir / "optimal_model.json")
        reference = pd.read_csv(fold_dir / "outcome_reference.csv").drop(columns="target")
        reference["prediction"] = model.predict(reference.to_numpy())
        reference.to_csv(fold_dir / "prediction_reference.csv", index=False)
        model.save(fold_dir / "prediction_reference_model.json")
        for feature_idx in range(model.feature_size):
            model.reduced_model(feature_idx).save(fold_dir / f"optimal_reduced_model_{feature_idx}.json")
    return prepared_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dgp", nargs="+", choices=DGP_NAMES, default=list(DGP_NAMES))
    parser.add_argument("--output_dir", default="datasets/dgp")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=2026)
    args = parser.parse_args()
    for name in args.dgp:
        prepare_dgp(name, Path(args.output_dir) / name, n=args.n,
                    random_state=args.random_state, n_splits=args.n_splits)


if __name__ == "__main__":
    main()
