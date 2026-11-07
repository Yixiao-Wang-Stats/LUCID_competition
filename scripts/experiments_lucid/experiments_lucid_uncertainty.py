# Load saved DRBART fits and calculate LUCID for fixed posterior draws.
from pathlib import Path
import argparse
import hashlib
import numpy as np
import pandas as pd

from scripts.experiments_lucid.experiments_lucid import build_sampler
from scripts.experiments_lucid.experiments_group_lucid import build_group_sampler, get_feature_types_from_file, XGBoostModel, KnownDGPModel
from package.src.LUCID_calculation.lucid import lucid_all_evaluations_one_feature, lucid_all_evaluations_group


class FixedPosteriorSampler:
    def __init__(self,sampler,posterior_id):
        self.sampler = sampler
        self.posterior_id = int(posterior_id)

    def sample_batch(self,evaluation_sample,feature_indices,replacement_sample_size,X_reference=None):
        return self.sampler.sample_batch_posterior_credible_interval(evaluation_sample=evaluation_sample,feature_indices=feature_indices,replacement_sample_size=replacement_sample_size,posterior_id=self.posterior_id,X_reference=X_reference)

    def sample_group_batch(self,evaluation_sample,feature_indices,replacement_sample_size,X_reference=None):
        return self.sampler.sample_group_batch_posterior_credible_interval(evaluation_sample=evaluation_sample,feature_indices=feature_indices,replacement_sample_size=replacement_sample_size,posterior_id=self.posterior_id,X_reference=X_reference)


def load_existing_sampler(sampler_input_dir,feature_indices,feature_types,random_state,group=False):
    sampler_input_dir = Path(sampler_input_dir)
    sampler = build_group_sampler("drbart",random_state,sampler_input_dir) if group else build_sampler("drbart",random_state,sampler_input_dir,feature_types)
    sampler.feature_size = len(feature_types)
    sampler.bart_sampler.feature_size = len(feature_types)
    outside = [j for j in range(len(feature_types)) if j not in feature_indices]
    group_name = "group_" + "_".join(str(j) for j in feature_indices)
    counts, fits = [], []
    for position,feature_idx in enumerate(feature_indices):
        feature_type = feature_types[feature_idx]
        if feature_type not in {"continuous","binary","categorical"}:
            raise ValueError("Unknown feature type.")
        fit_dir = sampler_input_dir / group_name / f"feature_{feature_idx}" if group else sampler_input_dir / f"feature_{feature_idx}"
        required = ["posterior_count.txt","fit.pkl"] if feature_type in {"binary","categorical"} else ["posterior_count.txt","fit.rds","dr_bart_mean.txt"]
        for filename in required:
            if not (fit_dir / filename).is_file():
                raise FileNotFoundError(f"Missing saved sampler fit: {fit_dir / filename}. Run point LUCID first.")
        posterior_count = int((fit_dir / "posterior_count.txt").read_text())
        if posterior_count < 1:
            raise ValueError("Saved posterior_count must be positive.")
        counts.append(posterior_count)
        fits.append({"feature_idx":feature_idx,"feature_type":feature_type,"conditioning_indices":outside + feature_indices[:position],"fit_dir":fit_dir,"posterior_count":posterior_count})
        if not group:
            sampler.fits[feature_idx] = fit_dir
            sampler.feature_types[feature_idx] = feature_type
            if feature_type in {"binary","categorical"}:
                sampler.bart_sampler.fits[feature_idx] = fit_dir
    if group:
        sampler.group_fits[tuple(feature_indices)] = fits
    sampler.is_fitted = True
    sampler.bart_sampler.is_fitted = True
    return sampler,np.asarray(counts,dtype=int),[str(fit["fit_dir"]) for fit in fits]


def _metadata_path(path):
    """Record project-relative paths or opaque identifiers for external files."""
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[2]
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return "external/" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()


def summarize_posterior_vi(posterior_vi,credible_level):
    posterior_vi = np.asarray(posterior_vi,dtype=float)
    if posterior_vi.ndim != 2 or posterior_vi.shape[0] < 2 or not np.isfinite(posterior_vi).all():
        raise ValueError("posterior_vi must be a finite (posterior_draws, observations) array with at least two draws.")
    if not 0 < credible_level < 1:
        raise ValueError("credible_level must be between zero and one.")
    lower,upper = np.quantile(posterior_vi,[(1-credible_level)/2,(1+credible_level)/2],axis=0)
    return {"mean":posterior_vi.mean(axis=0),"median":np.median(posterior_vi,axis=0),"lower":lower,"upper":upper,"std":posterior_vi.std(axis=0,ddof=1)}


def run_lucid_uncertainty_experiment(train_path,evaluation_path,model_path,source_dir,output_dir,target_name,evaluation_target_name,fold_idx,feature_indices,feature_types_path,sampler_name="drbart",group=False,posterior_sample_size=100,replacement_sample_size=500,credible_level=.95,evaluation_sample_batch_size=100,replacement_sample_batch_size=100,loss_type="squared_error",random_state=2026,reference_name="prediction"):
    if sampler_name != "drbart" or reference_name not in {"prediction","outcome"}:
        raise ValueError("Uncertainty currently supports drbart with prediction or outcome reference.")
    if posterior_sample_size < 2 or replacement_sample_size < 1 or not 0 < credible_level < 1:
        raise ValueError("Use at least two posterior draws, positive replacement size, and a credible level between zero and one.")
    train_dataset = pd.read_csv(train_path)
    evaluation_dataset = pd.read_csv(evaluation_path)
    feature_names = [column for column in train_dataset.columns if column != target_name]
    X_train = train_dataset[feature_names].to_numpy()
    evaluation_sample = evaluation_dataset[feature_names].to_numpy()
    evaluation_target = evaluation_dataset[evaluation_target_name].to_numpy()
    feature_indices = [int(j) for j in feature_indices]
    if not feature_indices or len(set(feature_indices)) != len(feature_indices) or any(j < 0 or j >= X_train.shape[1] for j in feature_indices):
        raise ValueError("feature_indices must be nonempty, unique, and in range.")
    if not group and len(feature_indices) != 1:
        raise ValueError("Single-feature uncertainty requires exactly one feature index.")
    feature_types = get_feature_types_from_file(feature_types_path,len(feature_names))
    is_dgp = "dgp" in Path(train_path).parts or Path(train_path).with_name("prediction_reference_model.json").is_file()
    if is_dgp:
        if reference_name != "prediction":
            raise ValueError("DGP uncertainty requires prediction reference from the known model.")
        model_path = Path(train_path).with_name("optimal_model.json")
    model_class = KnownDGPModel if Path(model_path).name == "optimal_model.json" else XGBoostModel
    model = model_class.load(model_path)
    sampler_input_dir = Path(source_dir) / f"fold_{fold_idx}" / "sampler_fit"
    sampler,posterior_counts,fit_dirs = load_existing_sampler(sampler_input_dir,feature_indices,feature_types,random_state,group)
    sampler.load_feature_bounds(Path(feature_types_path).with_name("feature_bounds.csv"),feature_names,feature_types)
    if not group and posterior_sample_size > posterior_counts[0]:
        raise ValueError("posterior_sample_size exceeds the saved single-feature posterior count.")
    selected = np.arange(1,posterior_sample_size+1) if group else np.random.default_rng(np.random.SeedSequence([random_state,1])).choice(posterior_counts[0],size=posterior_sample_size,replace=False)+1
    posterior_ids = np.empty((posterior_sample_size,len(feature_indices)),dtype=int)
    posterior_vi = np.empty((posterior_sample_size,len(evaluation_sample)))
    for position,posterior_id in enumerate(selected):
        fixed_sampler = FixedPosteriorSampler(sampler,posterior_id)
        if group:
            rng = np.random.default_rng(np.random.SeedSequence([sampler.seed,int(posterior_id)]))
            posterior_ids[position] = [rng.integers(1,count+1) for count in posterior_counts]
            posterior_vi[position] = lucid_all_evaluations_group(evaluation_sample=evaluation_sample,evaluation_sample_batch_size=evaluation_sample_batch_size,evaluation_target=evaluation_target,feature_indices=feature_indices,sampler=fixed_sampler,replacement_sample_size=replacement_sample_size,replacement_sample_batch_size=replacement_sample_batch_size,loss_type=loss_type,model=model.predict)
        else:
            posterior_ids[position,0] = posterior_id
            posterior_vi[position] = lucid_all_evaluations_one_feature(evaluation_sample=evaluation_sample,evaluation_sample_batch_size=evaluation_sample_batch_size,evaluation_target=evaluation_target,feature_idx=feature_indices[0],sampler=fixed_sampler,replacement_sample_size=replacement_sample_size,replacement_sample_batch_size=replacement_sample_batch_size,loss_type=loss_type,model=model.predict)
        print("Posterior draw:",position+1,"/",posterior_sample_size,flush=True)
    summary = summarize_posterior_vi(posterior_vi,credible_level)
    fold_output_dir = Path(output_dir) / f"fold_{fold_idx}"
    fold_output_dir.mkdir(parents=True,exist_ok=True)
    name = ("group_" if group else "feature_") + "_".join(str(j) for j in feature_indices)
    output_path = fold_output_dir / f"{name}_uncertainty.npz"
    np.savez_compressed(output_path,posterior_vi=posterior_vi,posterior_ids=posterior_ids,posterior_counts=posterior_counts,feature_indices=np.asarray(feature_indices),feature_names=np.asarray([feature_names[j] for j in feature_indices]),group=group,fold_idx=fold_idx,sampler_name=sampler_name,reference_name=reference_name,random_state=random_state,posterior_sample_size=posterior_sample_size,replacement_sample_size=replacement_sample_size,credible_level=credible_level,evaluation_sample_batch_size=evaluation_sample_batch_size,replacement_sample_batch_size=replacement_sample_batch_size,loss_type=loss_type,source_dir=_metadata_path(source_dir),model_path=_metadata_path(model_path),fit_dirs=np.asarray([_metadata_path(path) for path in fit_dirs]),**summary)
    pd.DataFrame(summary).to_csv(fold_output_dir / f"{name}_uncertainty.csv",index=False)
    print("Saved to:",output_path)
    return posterior_vi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_dir",default="datasets/auto_mpg/prepared")
    parser.add_argument("--train_path",default=None)
    parser.add_argument("--evaluation_path",default=None)
    parser.add_argument("--model_path",default=None)
    parser.add_argument("--source_dir",required=True)
    parser.add_argument("--output_dir",required=True)
    parser.add_argument("--target_name",default="mpg")
    parser.add_argument("--evaluation_target_name",default=None)
    parser.add_argument("--fold_idx",type=int,default=0)
    parser.add_argument("--feature_indices",type=int,nargs="+",required=True)
    parser.add_argument("--feature_types_path",default=None)
    parser.add_argument("--sampler_name",choices=["drbart"],default="drbart")
    parser.add_argument("--reference_name",choices=["prediction","outcome"],default="prediction")
    parser.add_argument("--group",action="store_true")
    parser.add_argument("--posterior_sample_size",type=int,default=100)
    parser.add_argument("--replacement_sample_size",type=int,default=500)
    parser.add_argument("--credible_level",type=float,default=.95)
    parser.add_argument("--evaluation_sample_batch_size",type=int,default=100)
    parser.add_argument("--replacement_sample_batch_size",type=int,default=100)
    parser.add_argument("--loss_type",default="squared_error")
    parser.add_argument("--random_state",type=int,default=2026)
    args = parser.parse_args()
    fold_dir = Path(args.prepared_dir) / f"fold_{args.fold_idx}"
    is_dgp = "dgp" in fold_dir.parts or (fold_dir / "prediction_reference_model.json").is_file()
    if args.train_path is None:
        args.train_path = fold_dir / "train.csv"
    if args.evaluation_path is None:
        args.evaluation_path = fold_dir / f"{args.reference_name}_reference.csv"
    if args.model_path is None:
        args.model_path = fold_dir / ("optimal_model.json" if is_dgp else "predictive_model.json")
    if args.feature_types_path is None:
        args.feature_types_path = Path(args.prepared_dir) / "feature_types.csv"
    if args.evaluation_target_name is None:
        args.evaluation_target_name = "prediction" if args.reference_name == "prediction" else args.target_name
    arguments = vars(args)
    arguments.pop("prepared_dir")
    run_lucid_uncertainty_experiment(**arguments)


if __name__ == "__main__":
    main()
