#  This is the file offering some helper functions for data perparation process

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

def get_kfold_indices(
    X,
    n_splits=5,
    random_state=2026,
    y=None,
):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y is not None:
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("Binary targets must be 1D and match the number of observations.")
        classes,counts = np.unique(y,return_counts=True)
        if not np.array_equal(classes,[0,1]):
            raise ValueError("binary:logistic requires both target classes 0 and 1, without missing values.")
        if counts.min() < n_splits:
            raise ValueError("Each binary target class must contain at least n_splits observations.")
    kfold = (StratifiedKFold if y is not None else KFold)(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    fold_indices = []
    for train_indices,evaluation_indices in kfold.split(X,y):
        fold_indices.append(
            (train_indices,evaluation_indices)
        )
    return fold_indices


def split_dataset(
    X,
    y,
    test_size=0.2,
    random_state=2026,
):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of observations.")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test

def fit_predictive_model(
    model,
    X_train,
    y_train,
):
    model.fit(X_train,y_train,)
    return model

def save_predictive_model(
    model,
    output_path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True,)
    model.save(output_path)

def save_dataset(
    X,
    y,
    feature_names,
    target_name,
    output_path,
):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of observations.")
    if len(feature_names) != X.shape[1]:
        raise ValueError("feature_names must have one entry for every feature.")
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    dataset = pd.DataFrame(
        X,
        columns=feature_names,
    )
    dataset[target_name] = y
    dataset.to_csv(
        output_path,
        index=False,
    )

def prepare_prediction_reference(
    X,
    model,
):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return np.asarray(model.predict(X))


def get_feature_type(
    X,
    feature_idx,
    feature_type=None,
):
    if feature_type is not None:
        if feature_type not in {"binary","continuous"}:
            raise ValueError("feature_type must be binary or continuous.")
        return feature_type
    feature_values = np.asarray(X[:,feature_idx])
    unique_values = np.unique(feature_values[~pd.isna(feature_values)])
    if len(unique_values) == 2:
        return "binary"
    return "continuous"

def save_feature_types(
    feature_names,
    feature_types,
    output_path,
):
    feature_type_dataset = pd.DataFrame({
        "feature_idx":range(len(feature_names)),
        "feature_name":feature_names,
        "feature_type":feature_types,
    })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    feature_type_dataset.to_csv(output_path,index=False)


def save_feature_bounds(X,feature_names,feature_types,output_path,feature_bounds=None,padding=.1):
    """Save full-dataset continuous bounds; physical bounds override padding."""
    X = np.asarray(X)
    if X.ndim != 2 or len(X) == 0 or X.shape[1] != len(feature_names) or len(feature_types) != len(feature_names):
        raise ValueError("X, feature_names and feature_types must describe the same nonempty dataset.")
    if not np.isfinite(padding) or padding <= 0:
        raise ValueError("padding must be finite and positive.")
    feature_bounds = {} if feature_bounds is None else feature_bounds
    if not isinstance(feature_bounds,dict) or set(feature_bounds)-set(feature_names):
        raise ValueError("feature_bounds must map known feature names to [lower,upper].")
    records = []
    for feature_idx,(name,feature_type) in enumerate(zip(feature_names,feature_types)):
        if feature_type not in {"continuous","binary","categorical"}:
            raise ValueError("Unknown feature type.")
        lower,upper,source = None,None,feature_type
        if feature_type != "continuous" and name in feature_bounds:
            raise ValueError(f"Bounds only apply to continuous features: {name}.")
        if feature_type == "continuous":
            values = np.asarray(X[:,feature_idx],dtype=float)
            values = values[~np.isnan(values)]
            if len(values) == 0 or not np.isfinite(values).all():
                raise ValueError(f"{name} must have finite observed values.")
            observed_lower,observed_upper = float(values.min()),float(values.max())
            if name in feature_bounds:
                bounds = np.asarray(feature_bounds[name],dtype=float)
                if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                    raise ValueError(f"Invalid physical bounds for {name}.")
                lower,upper = map(float,bounds)
                if observed_lower < lower or observed_upper > upper:
                    raise ValueError(f"Observed {name} values fall outside physical bounds.")
                source = "physical"
            else:
                width = max(observed_upper-observed_lower,1e-8*max(1,abs(observed_lower),abs(observed_upper)))
                lower,upper = observed_lower-padding*width,observed_upper+padding*width
                if not np.isfinite([lower,upper]).all() or lower >= upper:
                    raise ValueError(f"Cannot construct finite padded bounds for {name}.")
                source = "padded"
        records.append({"feature_idx":feature_idx,"feature_name":name,"lower_bound":lower,"upper_bound":upper,"bound_source":source})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(output_path,index=False)


def save_fold_indices(
    fold_indices,
    output_path,
):
    records = []
    for fold_idx,(_,evaluation_indices) in enumerate(fold_indices):
        for observation_idx in evaluation_indices:
            records.append({
                "observation_idx":observation_idx,
                "fold_idx":fold_idx,
            })
    fold_indices_dataset = pd.DataFrame(records)
    fold_indices_dataset = fold_indices_dataset.sort_values("observation_idx")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    fold_indices_dataset.to_csv(output_path,index=False)
