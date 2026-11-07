# This is the file to run group LUCID end to end
# load prepared train data
# load evaluation data
# load frozen predictive model
# construct group sampler
# run lucid_group_end_to_end
# save this group's LUCID

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from package.src.predictive_model.xgboost_model import XGBoostModel
from package.src.sampler.drbart.drbart_group_sampler import DRBartGroupSampler
from package.src.LUCID_calculation.lucid import lucid_group_end_to_end
from scripts.data_preparation.dgp.dgp_generation import KnownDGPModel


def run_group_lucid_experiment(
    train_path,
    evaluation_path,
    model_path,
    output_dir,
    target_name,
    evaluation_target_name,
    fold_idx,
    feature_indices,
    feature_types_path,
    sampler_name,
    evaluation_sample_batch_size,
    replacement_sample_size,
    replacement_sample_batch_size,
    loss_type,
    random_state,
):
    train_dataset = pd.read_csv(train_path)
    evaluation_dataset = pd.read_csv(evaluation_path)

    feature_names = [column for column in train_dataset.columns if column != target_name]

    X_train = train_dataset[feature_names].to_numpy()
    evaluation_sample = evaluation_dataset[feature_names].to_numpy()
    evaluation_target = evaluation_dataset[evaluation_target_name].to_numpy()

    feature_indices = [int(feature_idx) for feature_idx in feature_indices]

    if len(feature_indices) == 0:
        raise ValueError("feature_indices must not be empty.")

    if len(set(feature_indices)) != len(feature_indices):
        raise ValueError("feature_indices must not contain duplicates.")

    for feature_idx in feature_indices:
        if not 0 <= feature_idx < len(feature_names):
            raise ValueError("feature_indices contains an out-of-range feature index.")

    feature_types = get_feature_types_from_file(
        feature_types_path=feature_types_path,
        feature_size=len(feature_names),
    )

    group_feature_types = [feature_types[feature_idx] for feature_idx in feature_indices]

    if "dgp" in Path(train_path).parts or Path(train_path).with_name("prediction_reference_model.json").is_file():
        model_path = Path(train_path).with_name("optimal_model.json")

    model_class = KnownDGPModel if Path(model_path).name == "optimal_model.json" else XGBoostModel
    model = model_class.load(model_path)

    fold_output_dir = Path(output_dir) / f"fold_{fold_idx}"
    sampler_output_dir = fold_output_dir / "sampler_fit"

    sampler = build_group_sampler(
        sampler_name=sampler_name,
        random_state=random_state,
        sampler_output_dir=sampler_output_dir,
    )

    sampler.load_feature_bounds(Path(feature_types_path).with_name("feature_bounds.csv"),feature_names,feature_types)
    lucid_result = lucid_group_end_to_end(
        X_train=X_train,
        evaluation_sample=evaluation_sample,
        evaluation_sample_batch_size=evaluation_sample_batch_size,
        evaluation_target=evaluation_target,
        feature_indices=feature_indices,
        feature_types=group_feature_types,
        sampler=sampler,
        replacement_sample_size=replacement_sample_size,
        replacement_sample_batch_size=replacement_sample_batch_size,
        loss_type=loss_type,
        model=model,
    )

    fold_output_dir.mkdir(parents=True,exist_ok=True)

    group_name = "_".join(str(feature_idx) for feature_idx in feature_indices)
    output_path = fold_output_dir / f"group_{group_name}.npy"

    np.save(output_path,lucid_result)

    print("Sampler:",sampler_name)
    print("Fold index:",fold_idx)
    print("Feature indices:",feature_indices)
    print("Feature names:",[feature_names[feature_idx] for feature_idx in feature_indices])
    print("Feature types:",group_feature_types)
    print("Group LUCID result shape:",lucid_result.shape)
    print("Saved to:",output_path)


def get_feature_types_from_file(feature_types_path,feature_size):
    feature_types_dataset = pd.read_csv(feature_types_path)
    feature_types = feature_types_dataset["feature_type"].tolist()

    if len(feature_types) != feature_size:
        raise ValueError("feature_types.csv must have one entry per feature.")

    return feature_types


def build_group_sampler(sampler_name,random_state,sampler_output_dir):
    if sampler_name == "drbart":
        return DRBartGroupSampler(
            output_dir=sampler_output_dir,
            seed=random_state,
            random_state=random_state,
        )

    raise ValueError("sampler_name must be drbart.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prepared_dir",default="LUCID/datasets/auto_mpg/prepared")
    parser.add_argument("--train_path",default=None)
    parser.add_argument("--evaluation_path",default=None)
    parser.add_argument("--model_path",default=None)
    parser.add_argument("--output_dir",default=None)
    parser.add_argument("--target_name",default="mpg")
    parser.add_argument("--evaluation_target_name",default=None)
    parser.add_argument("--fold_idx",type=int,default=0)
    parser.add_argument("--feature_indices",type=int,nargs="+",required=True)
    parser.add_argument("--feature_types_path",default="LUCID/datasets/auto_mpg/prepared/feature_types.csv")
    parser.add_argument("--sampler_name",choices=["drbart"],default="drbart")
    parser.add_argument("--evaluation_sample_batch_size",type=int,default=100)
    parser.add_argument("--replacement_sample_size",type=int,default=1000)
    parser.add_argument("--replacement_sample_batch_size",type=int,default=100)
    parser.add_argument("--loss_type",default="squared_error")
    parser.add_argument("--random_state",type=int,default=2026)

    args = parser.parse_args()

    
    fold_dir = Path(args.prepared_dir) / f"fold_{args.fold_idx}"
    is_dgp = "dgp" in fold_dir.parts or (fold_dir / "prediction_reference_model.json").is_file()
    reference_name = "prediction" if is_dgp else "outcome"

    if args.output_dir is None:
        args.output_dir = f"results/auto_mpg/{args.sampler_name}/group/{reference_name}/seed_{args.random_state}"

    if args.train_path is None:
        args.train_path = fold_dir / "train.csv"

    if args.evaluation_path is None:
        args.evaluation_path = fold_dir / ("prediction_reference.csv" if is_dgp else "outcome_reference.csv")

    if args.model_path is None:
        model_name = "optimal_model.json" if is_dgp else "predictive_model.json"
        args.model_path = fold_dir / model_name

    if args.evaluation_target_name is None:
        args.evaluation_target_name = "prediction" if is_dgp else args.target_name

    run_group_lucid_experiment(
        train_path=args.train_path,
        evaluation_path=args.evaluation_path,
        model_path=args.model_path,
        feature_types_path=args.feature_types_path,
        output_dir=args.output_dir,
        target_name=args.target_name,
        evaluation_target_name=args.evaluation_target_name,
        sampler_name=args.sampler_name,
        fold_idx=args.fold_idx,
        feature_indices=args.feature_indices,
        evaluation_sample_batch_size=args.evaluation_sample_batch_size,
        replacement_sample_size=args.replacement_sample_size,
        replacement_sample_batch_size=args.replacement_sample_batch_size,
        loss_type=args.loss_type,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
