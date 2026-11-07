from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def aggregate_attributions(output_dir,fold_indices_path,n_splits):
    fold_indices_dataset = pd.read_csv(fold_indices_path)
    attribution_result = None
    for fold_idx in range(n_splits):
        fold_path = Path(output_dir) / f"fold_{fold_idx}" / "attributions.npy"
        fold_result = np.load(fold_path)
        evaluation_indices = fold_indices_dataset.loc[fold_indices_dataset["fold_idx"] == fold_idx,"observation_idx"].to_numpy()
        if fold_result.shape[0] != len(evaluation_indices):
            raise ValueError(f"Fold {fold_idx} attribution size does not match fold indices.")
        if attribution_result is None:
            attribution_result = np.empty((len(fold_indices_dataset),fold_result.shape[1]),dtype=float)
        attribution_result[evaluation_indices] = fold_result
    npy_output_path = Path(output_dir) / "attributions.npy"
    csv_output_path = Path(output_dir) / "attributions.csv"
    np.save(npy_output_path,attribution_result)
    pd.DataFrame(attribution_result).to_csv(csv_output_path,index=False)
    print("Aggregated attribution shape:",attribution_result.shape)
    print("Saved to:",npy_output_path)
    print("Saved to:",csv_output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",required=True)
    parser.add_argument("--fold_indices_path",required=True)
    parser.add_argument("--n_splits",type=int,default=5)
    args = parser.parse_args()
    aggregate_attributions(
        output_dir=args.output_dir,
        fold_indices_path=args.fold_indices_path,
        n_splits=args.n_splits,
    )

if __name__ == "__main__":
    main()