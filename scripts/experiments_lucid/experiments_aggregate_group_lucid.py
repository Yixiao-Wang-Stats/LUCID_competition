from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def aggregate_group_lucid(input_dir,output_dir,fold_indices_path,feature_indices,n_splits):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    group_name = "_".join(str(feature_idx) for feature_idx in feature_indices)
    fold_indices_dataset = pd.read_csv(fold_indices_path)

    if not {"observation_idx","fold_idx"}.issubset(fold_indices_dataset.columns):
        raise ValueError("fold_indices.csv must contain observation_idx and fold_idx.")

    evaluation_size = len(fold_indices_dataset)
    lucid_result = np.empty(evaluation_size)

    for fold_idx in range(n_splits):
        fold_input_dir = input_dir / f"fold_{fold_idx}"
        group_path = fold_input_dir / f"group_{group_name}.npy"

        if not group_path.exists():
            raise FileNotFoundError(f"Missing group LUCID result: {group_path}")

        fold_lucid_result = np.load(group_path)

        if fold_lucid_result.ndim != 1:
            raise ValueError(f"group_{group_name}.npy must be a 1D array.")

        evaluation_indices = fold_indices_dataset.loc[
            fold_indices_dataset["fold_idx"] == fold_idx,
            "observation_idx",
        ].to_numpy(dtype=int)

        if len(evaluation_indices) != len(fold_lucid_result):
            raise ValueError(f"Fold {fold_idx} LUCID size does not match fold_indices.csv.")

        lucid_result[evaluation_indices] = fold_lucid_result

        np.save(fold_input_dir / f"group_{group_name}_lucid.npy",fold_lucid_result)
        pd.DataFrame({"lucid":fold_lucid_result}).to_csv(fold_input_dir / f"group_{group_name}_lucid.csv",index=False)

        print("Fold index:",fold_idx)
        print("Fold LUCID result shape:",fold_lucid_result.shape)

    output_dir.mkdir(parents=True,exist_ok=True)

    npy_output_path = output_dir / f"group_{group_name}_lucid.npy"
    csv_output_path = output_dir / f"group_{group_name}_lucid.csv"

    np.save(npy_output_path,lucid_result)
    pd.DataFrame({"lucid":lucid_result}).to_csv(csv_output_path,index=False)

    print("Full LUCID result shape:",lucid_result.shape)
    print("Saved npy to:",npy_output_path)
    print("Saved csv to:",csv_output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",default="auto_mpg")
    parser.add_argument("--sampler_name",choices=["drbart"],default="drbart")
    parser.add_argument("--reference_name",default="outcome")
    parser.add_argument("--random_state",type=int,default=2026)
    parser.add_argument("--feature_indices",type=int,nargs="+",required=True)
    parser.add_argument("--n_splits",type=int,default=5)
    parser.add_argument("--input_dir",default=None)
    parser.add_argument("--output_dir",default=None)
    parser.add_argument("--fold_indices_path",default=None)

    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = f"results/{args.dataset_name}/{args.sampler_name}/group/{args.reference_name}/seed_{args.random_state}"

    if args.output_dir is None:
        args.output_dir = args.input_dir

    if args.fold_indices_path is None:
        args.fold_indices_path = f"datasets/{args.dataset_name}/prepared/fold_indices.csv"

    aggregate_group_lucid(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fold_indices_path=args.fold_indices_path,
        feature_indices=args.feature_indices,
        n_splits=args.n_splits,
    )


if __name__ == "__main__":
    main()