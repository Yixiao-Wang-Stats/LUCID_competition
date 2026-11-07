from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def aggregate_one_fold(
    fold_input_dir,
    feature_size,
):
    fold_input_dir = Path(fold_input_dir)
    lucid_results = []
    evaluation_size = None
    for feature_idx in range(feature_size):
        feature_path = fold_input_dir / f"feature_{feature_idx}.npy"
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing LUCID result: {feature_path}")
        feature_result = np.load(feature_path)
        if feature_result.ndim != 1:
            raise ValueError(f"feature_{feature_idx}.npy must be a 1D array.")
        if evaluation_size is None:
            evaluation_size = len(feature_result)
        elif len(feature_result) != evaluation_size:
            raise ValueError("All feature LUCID results must have the same number of observations.")
        lucid_results.append(feature_result)
    return np.column_stack(lucid_results)

def aggregate_lucid(
    input_dir,
    output_dir,
    fold_indices_path,
    feature_size,
    n_splits,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    fold_indices_dataset = pd.read_csv(fold_indices_path)
    if not {"observation_idx","fold_idx"}.issubset(fold_indices_dataset.columns):
        raise ValueError("fold_indices.csv must contain observation_idx and fold_idx.")
    evaluation_size = len(fold_indices_dataset)
    lucid_result = np.empty((evaluation_size,feature_size))
    for fold_idx in range(n_splits):
        fold_input_dir = input_dir / f"fold_{fold_idx}"
        fold_lucid_result = aggregate_one_fold(
            fold_input_dir=fold_input_dir,
            feature_size=feature_size,
        )
        evaluation_indices = fold_indices_dataset.loc[
            fold_indices_dataset["fold_idx"] == fold_idx,
            "observation_idx",
        ].to_numpy(dtype=int)
        if len(evaluation_indices) != fold_lucid_result.shape[0]:
            raise ValueError(f"Fold {fold_idx} LUCID size does not match fold_indices.csv.")
        lucid_result[evaluation_indices,:] = fold_lucid_result
        fold_npy_output_path = fold_input_dir / "lucid.npy"
        fold_csv_output_path = fold_input_dir / "lucid.csv"
        np.save(fold_npy_output_path,fold_lucid_result)
        pd.DataFrame(fold_lucid_result).to_csv(fold_csv_output_path,index=False)
        print("Fold index:",fold_idx)
        print("Fold LUCID result shape:",fold_lucid_result.shape)
    output_dir.mkdir(parents=True,exist_ok=True)
    npy_output_path = output_dir / "lucid.npy"
    csv_output_path = output_dir / "lucid.csv"
    np.save(npy_output_path,lucid_result)
    pd.DataFrame(lucid_result).to_csv(csv_output_path,index=False)
    print("Full LUCID result shape:",lucid_result.shape)
    print("Saved npy to:",npy_output_path)
    print("Saved csv to:",csv_output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",default="auto_mpg")
    parser.add_argument("--sampler_name",choices=["permutation","drbart","cmr","gcmr","arf","cspfi","clique","loco","optimal_lucid","optimal_loco"],default="permutation")
    parser.add_argument("--reference_name",default="outcome")
    parser.add_argument("--random_state",type=int,default=2026)
    parser.add_argument("--feature_size",type=int,default=7)
    parser.add_argument("--n_splits",type=int,default=5)
    parser.add_argument("--input_dir",default=None)
    parser.add_argument("--output_dir",default=None)
    parser.add_argument("--fold_indices_path",default=None)
    args = parser.parse_args()
    if args.input_dir is None:
        args.input_dir = f"results/{args.dataset_name}/{args.sampler_name}/{args.reference_name}/seed_{args.random_state}"
    if args.output_dir is None:
        args.output_dir = args.input_dir
    if args.fold_indices_path is None:
        args.fold_indices_path = f"LUCID/datasets/{args.dataset_name}/prepared/fold_indices.csv"
    aggregate_lucid(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fold_indices_path=args.fold_indices_path,
        feature_size=args.feature_size,
        n_splits=args.n_splits,
    )

if __name__ == "__main__":
    main()