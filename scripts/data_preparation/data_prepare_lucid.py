from pathlib import Path
import argparse
import json
import pandas as pd

from package.src.predictive_model.xgboost_model import XGBoostModel
from scripts.data_preparation.data_helper_functions import (
    get_kfold_indices,
    # split_dataset,
    fit_predictive_model,
    save_predictive_model,
    save_dataset,
    prepare_prediction_reference,
    get_feature_type,
    save_feature_types,
    save_feature_bounds,
    save_fold_indices
)

def prepare_lucid(
    input_path,
    output_dir,
    target_name,
    feature_types=None,
    # test_size=0.2,
    n_splits=5,
    random_state=2026,
    feature_bounds=None,
    bounds_only=False,
    objective="reg:squarederror",
):
    if objective not in {"reg:squarederror","binary:logistic"}:
        raise ValueError("objective must be reg:squarederror or binary:logistic.")
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    dataset = pd.read_csv(input_path)
    if target_name not in dataset.columns:
        raise ValueError(f"{target_name} is not a column in the dataset.")
    feature_names = [
        column for column in dataset.columns
        if column != target_name
    ]
    X = dataset[feature_names].to_numpy()
    y = dataset[target_name].to_numpy()
    if bounds_only and feature_types is None:
        feature_types = pd.read_csv(output_dir / "feature_types.csv")["feature_type"].tolist()
    if feature_types is None:
        feature_types = [get_feature_type(X=X,feature_idx=feature_idx,) for feature_idx in range(X.shape[1])]
    else:
        feature_types = [feature_type.strip().lower() for feature_type in feature_types]
        if len(feature_types) != X.shape[1]:
            raise ValueError("feature_types must have one entry for every feature.")
        if any(feature_type not in {"binary","continuous","categorical"} for feature_type in feature_types):
            raise ValueError("Each feature type must be binary, categorical, or continuous.")
    if bounds_only:
        saved_types = pd.read_csv(output_dir / "feature_types.csv")
        if saved_types["feature_name"].tolist() != feature_names or saved_types["feature_type"].tolist() != feature_types:
            raise ValueError("bounds_only requires feature_types matching the existing prepared dataset.")
    if not bounds_only:
        fold_indices = get_kfold_indices(X=X,n_splits=n_splits,random_state=random_state,y=y if objective == "binary:logistic" else None)
    save_feature_bounds(X=X,feature_names=feature_names,feature_types=feature_types,output_path=output_dir / "feature_bounds.csv",feature_bounds=feature_bounds)
    if bounds_only:
        return
    feature_types_output_path = output_dir / "feature_types.csv"
    save_feature_types(feature_names=feature_names,feature_types=feature_types,output_path=feature_types_output_path,)
    # X_train, X_test, y_train, y_test = split_dataset(X=X,y=y,test_size=test_size,random_state=random_state,)
    #  change to k-fold
    fold_indices_output_path = output_dir / "fold_indices.csv"
    save_fold_indices(
        fold_indices=fold_indices,
        output_path=fold_indices_output_path,
    )
    for fold_idx, (train_indices, evaluation_indices) in enumerate(fold_indices):
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[evaluation_indices]
        y_test = y[evaluation_indices]
        model = XGBoostModel(objective=objective,random_state=random_state,)
        model = fit_predictive_model(model=model,X_train=X_train,y_train=y_train,)
        fold_output_dir = output_dir / f"fold_{fold_idx}"
        model_output_path = fold_output_dir / "predictive_model.json"
        save_predictive_model(model=model,output_path=model_output_path,)
        train_output_path = fold_output_dir / "train.csv"
        save_dataset(X=X_train,y=y_train,feature_names=feature_names,target_name=target_name,output_path=train_output_path,)
        outcome_reference_output_path = fold_output_dir / "outcome_reference.csv"
        save_dataset(
            X=X_test,
            y=y_test,
            feature_names=feature_names,
            target_name=target_name,
            output_path=outcome_reference_output_path,
        )
        prediction_reference = prepare_prediction_reference(
            X=X_test,
            model=model,
        )
        prediction_reference_output_path = fold_output_dir / "prediction_reference.csv"
        save_dataset(
            X=X_test,
            y=prediction_reference,
            feature_names=feature_names,
            target_name="prediction",
            output_path=prediction_reference_output_path,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",default="datasets/auto_mpg/auto_mpg.csv")
    parser.add_argument("--output_dir",default="datasets/auto_mpg/prepared")
    parser.add_argument("--target_name",default="mpg")
    parser.add_argument("--objective",choices=["reg:squarederror","binary:logistic"],default="reg:squarederror",help="binary:logistic saves class-1 probabilities as prediction reference and uses stratified folds.")
    parser.add_argument("--feature_types",nargs="+",default=None)
    parser.add_argument("--feature_bounds",type=json.loads,default=None,help='Physical bounds JSON, e.g. {"temp":[0,1],"hum":[0,1]}; other continuous features receive 10%% padding.')
    parser.add_argument("--bounds_only",action="store_true",help="Only save feature_bounds.csv for an existing prepared dataset; do not fit models or rewrite other files.")
    parser.add_argument("--n_splits",type=int,default=5)
    parser.add_argument("--random_state",type=int,default=2026)
    args = parser.parse_args()
    prepare_lucid(
        input_path=args.input_path,
        output_dir=args.output_dir,
        target_name=args.target_name,
        objective=args.objective,
        feature_types=args.feature_types,
        feature_bounds=args.feature_bounds,
        bounds_only=args.bounds_only,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

if __name__ == "__main__":
    main()
