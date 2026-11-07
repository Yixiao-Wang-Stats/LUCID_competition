from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scripts.experiments_lucid.experiments_lucid_uncertainty import summarize_posterior_vi


def aggregate_lucid_uncertainty(input_dir,output_dir,fold_indices_path,feature_indices,n_splits,group=False):
    input_dir,output_dir = Path(input_dir),Path(output_dir)
    name = ("group_" if group else "feature_") + "_".join(str(j) for j in feature_indices)
    fold_indices_dataset = pd.read_csv(fold_indices_path)
    if not {"observation_idx","fold_idx"}.issubset(fold_indices_dataset.columns):
        raise ValueError("fold_indices.csv must contain observation_idx and fold_idx.")
    observation_indices = fold_indices_dataset["observation_idx"].to_numpy()
    fold_ids = fold_indices_dataset["fold_idx"].to_numpy()
    if n_splits < 1 or not np.array_equal(np.sort(observation_indices),np.arange(len(observation_indices))) or not np.isin(fold_ids,np.arange(n_splits)).all():
        raise ValueError("Fold indices must assign every observation exactly once to a valid fold.")
    protocol = ["feature_indices","feature_names","group","sampler_name","reference_name","random_state","posterior_sample_size","replacement_sample_size","credible_level","evaluation_sample_batch_size","replacement_sample_batch_size","loss_type","source_dir"]
    metadata,posterior_vi,posterior_ids,posterior_counts,model_paths,fit_dirs = None,None,[],[],[],[]
    for fold_idx in range(n_splits):
        path = input_dir / f"fold_{fold_idx}" / f"{name}_uncertainty.npz"
        with np.load(path,allow_pickle=False) as result:
            current = {key:result[key].copy() for key in protocol}
            if not np.array_equal(current["feature_indices"],feature_indices) or bool(current["group"]) != group or int(result["fold_idx"]) != fold_idx:
                raise ValueError(f"Feature/group/fold mismatch: {path}")
            if metadata is None:
                metadata = current
                posterior_vi = np.empty((int(current["posterior_sample_size"]),len(observation_indices)))
            elif any(not np.array_equal(metadata[key],current[key]) for key in protocol):
                raise ValueError(f"Uncertainty experiment parameters differ between folds: {path}")
            evaluation_indices = fold_indices_dataset.loc[fold_indices_dataset["fold_idx"] == fold_idx,"observation_idx"].to_numpy(dtype=int)
            values = result["posterior_vi"]
            if values.shape != (posterior_vi.shape[0],len(evaluation_indices)):
                raise ValueError(f"Posterior VI shape does not match fold indices: {path}")
            expected = summarize_posterior_vi(values,float(metadata["credible_level"]))
            for key in expected:
                if not np.array_equal(result[key],expected[key]):
                    raise ValueError(f"Saved interval summary does not match posterior VI: {path}")
            ids,counts = result["posterior_ids"],result["posterior_counts"]
            if ids.shape != (posterior_vi.shape[0],len(feature_indices)) or counts.shape != (len(feature_indices),) or ids.dtype.kind not in "iu" or counts.dtype.kind not in "iu" or np.any(counts < 1) or np.any(ids < 1) or np.any(ids > counts):
                raise ValueError(f"Invalid saved posterior IDs: {path}")
            posterior_vi[:,evaluation_indices] = values
            posterior_ids.append(ids.copy())
            posterior_counts.append(counts.copy())
            model_paths.append(str(result["model_path"]))
            fit_dirs.append(result["fit_dirs"].copy())
    summary = summarize_posterior_vi(posterior_vi,float(metadata["credible_level"]))
    output_dir.mkdir(parents=True,exist_ok=True)
    path = output_dir / f"{name}_uncertainty.npz"
    np.savez_compressed(path,posterior_vi=posterior_vi,posterior_ids=np.stack(posterior_ids),posterior_counts=np.stack(posterior_counts),model_paths=np.asarray(model_paths),fit_dirs=np.stack(fit_dirs),fold_ids=fold_ids[np.argsort(observation_indices)],**metadata,**summary)
    pd.DataFrame(summary).to_csv(output_dir / f"{name}_uncertainty.csv",index=False)
    print("Full posterior VI shape:",posterior_vi.shape)
    print("Saved to:",path)
    return posterior_vi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",required=True)
    parser.add_argument("--output_dir",default=None)
    parser.add_argument("--fold_indices_path",required=True)
    parser.add_argument("--feature_indices",type=int,nargs="+",required=True)
    parser.add_argument("--n_splits",type=int,default=5)
    parser.add_argument("--group",action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir
    aggregate_lucid_uncertainty(**vars(args))


if __name__ == "__main__":
    main()
