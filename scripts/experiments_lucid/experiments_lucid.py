# This is the file to run lucid end to end 
# load prepared train data
# load evaluation data
# load frozen predictive model
# construct sampler
# run lucid_one_feature_end_to_end
# save this feature's LUCID

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from package.src.predictive_model.xgboost_model import XGBoostModel
from package.src.sampler.permutation.permutation_sampler import PermutationSampler
from package.src.sampler.drbart.drbart_sampler import DRBartSampler
from package.src.sampler.cmr.cmr_sampler import CMRSampler
from package.src.sampler.gcmr.gcmr_sampler import GCMRSampler
from package.src.sampler.arf.arf_sampler import ARFSampler
from package.src.sampler.cspfi.cspfi_sampler import CSPFISampler
from package.src.sampler.clique.clique_sampler import CliqueSampler
from package.src.sampler.loco.loco_sampler import LOCOSampler
from package.src.LUCID_calculation.lucid import lucid_one_feature_end_to_end
from package.src.sampler.optimal.optimal_sampler import OptimalLUCIDSampler, OptimalLOCOSampler
from scripts.data_preparation.dgp.dgp_generation import KnownDGPModel, KnownReducedModel

def run_lucid_experiment(
    train_path,
    evaluation_path,
    model_path,
    output_dir,
    target_name,
    evaluation_target_name,
    fold_idx,
    feature_idx,
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
    if not 0 <= feature_idx < len(feature_names):
        raise ValueError("feature_idx is out of range.")
    feature_types = get_feature_types_from_file(
        feature_types_path=feature_types_path,
        feature_size=len(feature_names),
    )
    feature_type = feature_types[feature_idx]
    if "dgp" in Path(train_path).parts or Path(train_path).with_name("prediction_reference_model.json").is_file():
        model_path = Path(train_path).with_name("optimal_model.json")
    model_class = KnownDGPModel if Path(model_path).name == "optimal_model.json" else XGBoostModel
    model = model_class.load(model_path)
    fold_output_dir = Path(output_dir) / f"fold_{fold_idx}"
    sampler_output_dir = fold_output_dir / "sampler_fit"
    # sampler_output_dir = Path(output_dir) / "sampler_fit"  # some methods like DRBartSampler require an output directory
    sampler = build_sampler(
        sampler_name=sampler_name,
        random_state=random_state,
        sampler_output_dir=sampler_output_dir,
        feature_types=feature_types,
        model_path=model_path,
        feature_idx=feature_idx,
    )
    if sampler_name == "drbart":
        sampler.load_feature_bounds(Path(feature_types_path).with_name("feature_bounds.csv"),feature_names,feature_types)
    lucid_result = sampler.calculate_batch(
        evaluation_sample, evaluation_target, feature_idx, loss_type,
    ) if sampler_name == "optimal_lucid" else lucid_one_feature_end_to_end(
        X_train=X_train,
        evaluation_sample=evaluation_sample,
        evaluation_sample_batch_size=evaluation_sample_batch_size,
        evaluation_target=evaluation_target,
        feature_idx=feature_idx,
        feature_type=feature_type,
        sampler=sampler,
        replacement_sample_size=replacement_sample_size,
        replacement_sample_batch_size=replacement_sample_batch_size,
        loss_type=loss_type,
        model=model,
    )
    # output_dir = Path(output_dir)
    # output_dir.mkdir(parents=True,exist_ok=True)
    # output_path = output_dir / f"feature_{feature_idx}.npy"
    fold_output_dir.mkdir(parents=True,exist_ok=True)
    output_path = fold_output_dir / f"feature_{feature_idx}.npy"
    np.save(output_path,lucid_result)
    print("Sampler:",sampler_name)
    print("Fold index:",fold_idx)
    print("Feature index:",feature_idx)
    print("Feature name:",feature_names[feature_idx])
    print("Feature type:",feature_type)
    print("LUCID result shape:",lucid_result.shape)
    print("Saved to:",output_path)


def get_feature_types_from_file(
    feature_types_path,
    feature_size,
):
    feature_types_dataset = pd.read_csv(feature_types_path)
    feature_types = feature_types_dataset["feature_type"].tolist()
    # arf and cspfi needs all feature types
    if len(feature_types) != feature_size:
        raise ValueError(
            "feature_types.csv must have one entry per feature."
        )
    return feature_types

def build_sampler(
    sampler_name,
    random_state,
    sampler_output_dir,
    feature_types,
    model_path=None,
    feature_idx=None,
):
    if sampler_name == "optimal_lucid":
        return OptimalLUCIDSampler(KnownDGPModel.load(model_path), random_state=random_state)
    if sampler_name == "optimal_loco":
        reduced_path = Path(model_path).with_name(f"optimal_reduced_model_{feature_idx}.json")
        return OptimalLOCOSampler({feature_idx: KnownReducedModel.load(reduced_path)}, random_state=random_state)
    if sampler_name == "permutation":
        return PermutationSampler(random_state=random_state)
    if sampler_name == "drbart":
        return DRBartSampler(
            output_dir=sampler_output_dir,
            seed=random_state,
            random_state=random_state,
        )
    if sampler_name == "cmr":
        return CMRSampler(
            predictive_model_class=XGBoostModel,
            random_state=random_state,
        )
    if sampler_name == "gcmr":
        return GCMRSampler(
            predictive_model_class=XGBoostModel,
            random_state=random_state,
        )
    if sampler_name == "arf":
        return ARFSampler(
            output_dir=sampler_output_dir,
            feature_types=feature_types,
            random_state=random_state,
        )
    if sampler_name == "cspfi":
        return CSPFISampler(
            output_dir=sampler_output_dir,
            feature_types=feature_types,
            seed=random_state,
            random_state=random_state,
        )
    if sampler_name == "clique":
        return CliqueSampler(
            random_state=random_state,
        )
    if sampler_name == "loco":
        return LOCOSampler(
            predictive_model_class=XGBoostModel,
            random_state=random_state,
        )
    raise ValueError(
        "sampler_name must be permutation, drbart, cmr, gcmr, arf, cspfi, clique, loco, optimal_lucid, or optimal_loco."
    )


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
    parser.add_argument("--feature_idx",type=int,default=0)
    parser.add_argument("--feature_types_path",default="LUCID/datasets/auto_mpg/prepared/feature_types.csv")
    parser.add_argument("--sampler_name",choices=["permutation","drbart","cmr","gcmr","arf","cspfi","clique","loco","optimal_lucid","optimal_loco"],default="permutation")
    parser.add_argument("--evaluation_sample_batch_size",type=int,default=100)
    parser.add_argument("--replacement_sample_size",type=int,default=1000)
    parser.add_argument("--replacement_sample_batch_size",type=int,default=100)
    parser.add_argument("--loss_type",default="squared_error")
    parser.add_argument("--random_state",type=int,default=2026)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f"results/auto_mpg/{args.sampler_name}/outcome/seed_{args.random_state}"
    fold_dir = Path(args.prepared_dir) / f"fold_{args.fold_idx}"
    is_dgp = "dgp" in fold_dir.parts or (fold_dir / "prediction_reference_model.json").is_file()

    if args.train_path is None:
        args.train_path = fold_dir / "train.csv"

    if args.evaluation_path is None:
        args.evaluation_path = fold_dir / ("prediction_reference.csv" if is_dgp else "outcome_reference.csv")

    if args.model_path is None:
        model_name = "optimal_model.json" if is_dgp or args.sampler_name in {"optimal_lucid", "optimal_loco"} else "predictive_model.json"
        args.model_path = fold_dir / model_name
    if args.evaluation_target_name is None:
        args.evaluation_target_name = "prediction" if is_dgp else args.target_name

    run_lucid_experiment(
        train_path=args.train_path,
        evaluation_path=args.evaluation_path,
        model_path=args.model_path,
        feature_types_path=args.feature_types_path,
        output_dir=args.output_dir,
        target_name=args.target_name,
        evaluation_target_name=args.evaluation_target_name,
        sampler_name=args.sampler_name,
        fold_idx=args.fold_idx,
        feature_idx=args.feature_idx,
        evaluation_sample_batch_size=args.evaluation_sample_batch_size,
        replacement_sample_size=args.replacement_sample_size,
        replacement_sample_batch_size=args.replacement_sample_batch_size,
        loss_type=args.loss_type,
        random_state=args.random_state,
    )

if __name__ == "__main__":
    main()
