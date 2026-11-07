from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from package.src.predictive_model.xgboost_model import XGBoostModel
from scripts.data_preparation.dgp.dgp_generation import KnownDGPModel

def run_attribution_experiment(train_path,evaluation_path,model_path,output_dir,target_name,fold_idx,feature_types_path,method_name,replacement_sample_size,random_state):
    train_dataset = pd.read_csv(train_path)
    evaluation_dataset = pd.read_csv(evaluation_path)
    feature_names = [column for column in train_dataset.columns if column != target_name]
    X_train = train_dataset[feature_names].to_numpy()
    evaluation_sample = evaluation_dataset[feature_names].to_numpy()
    feature_types = get_feature_types_from_file(feature_types_path=feature_types_path,feature_size=len(feature_names))
    if "dgp" in Path(train_path).parts or Path(train_path).with_name("prediction_reference_model.json").is_file():
        model_path = Path(train_path).with_name("optimal_model.json")
    model_class = KnownDGPModel if Path(model_path).name == "optimal_model.json" else XGBoostModel
    model = model_class.load(model_path)
    # calculate attributions
    attribution_result = calculate_attributions(method_name=method_name,model=model,X_train=X_train,evaluation_sample=evaluation_sample,feature_names=feature_names,feature_types=feature_types,replacement_sample_size=replacement_sample_size,random_state=random_state)
    fold_output_dir = Path(output_dir) / f"fold_{fold_idx}"
    fold_output_dir.mkdir(parents=True,exist_ok=True)
    output_path = fold_output_dir / "attributions.npy"
    np.save(output_path,attribution_result)
    print("Method:",method_name)
    print("Fold index:",fold_idx)
    print("Attribution result shape:",attribution_result.shape)
    print("Saved to:",output_path)

def get_feature_types_from_file(feature_types_path,feature_size):
    feature_types_dataset = pd.read_csv(feature_types_path)
    feature_types = feature_types_dataset["feature_type"].tolist()
    if len(feature_types) != feature_size:
        raise ValueError("feature_types.csv must have one entry per feature.")
    return feature_types

def calculate_attributions(method_name,model,X_train,evaluation_sample,feature_names,feature_types,replacement_sample_size,random_state):
    if method_name == "shap":
        return calculate_shap(model=model,evaluation_sample=evaluation_sample,feature_names=feature_names,random_state=random_state)
    if method_name == "lime":
        return calculate_lime(model=model,X_train=X_train,evaluation_sample=evaluation_sample,feature_names=feature_names,feature_types=feature_types,replacement_sample_size=replacement_sample_size,random_state=random_state)
    if method_name == "shapr":
        return calculate_shapr(model=model,X_train=X_train,evaluation_sample=evaluation_sample,feature_names=feature_names,feature_types=feature_types,random_state=random_state)
    raise ValueError("method_name must be shap, lime, or shapr.")

def calculate_shap(model,evaluation_sample,feature_names,random_state):
    import shap
    explainer = shap.Explainer(
        model.predict,
        evaluation_sample,
        feature_names=feature_names,
        seed=random_state,
    )
    explanation = explainer(evaluation_sample)
    attribution_result = np.asarray(explanation.values,dtype=float)
    if attribution_result.shape != evaluation_sample.shape:
        raise ValueError("Unexpected SHAP output shape.")
    return attribution_result

def calculate_lime(model,X_train,evaluation_sample,feature_names,feature_types,replacement_sample_size,random_state):
    from lime.lime_tabular import LimeTabularExplainer
    categorical_features = [feature_idx for feature_idx,feature_type in enumerate(feature_types) if feature_type != "continuous"]
    explainer = LimeTabularExplainer(X_train,feature_names=feature_names,categorical_features=categorical_features or None,mode="regression",random_state=random_state)
    attribution_result = np.zeros(evaluation_sample.shape,dtype=float)
    for row_idx,row in enumerate(evaluation_sample):
        explanation = explainer.explain_instance(row,model.predict,num_features=len(feature_names),num_samples=replacement_sample_size)
        local_exp = next(iter(explanation.local_exp.values()))  # [(feature_idx, weight)]
        for feature_idx, weight in local_exp:
            attribution_result[row_idx,int(feature_idx)] = float(weight)
    return attribution_result

def calculate_shapr(model,X_train,evaluation_sample,feature_names,feature_types,random_state):
    from pyshapr import explain
    X_train = pd.DataFrame(X_train,columns=feature_names)
    evaluation_sample = pd.DataFrame(evaluation_sample,columns=feature_names)
    for feature_idx,feature_type in enumerate(feature_types):
        if feature_type == "continuous":
            continue
        feature_name = feature_names[feature_idx]
        categories = np.unique(X_train[feature_name])
        X_train[feature_name] = pd.Categorical(X_train[feature_name],categories=categories)
        evaluation_sample[feature_name] = pd.Categorical(evaluation_sample[feature_name],categories=categories)
        if evaluation_sample[feature_name].isna().any():
            raise ValueError(f"{feature_name} contains categories not present in training data.")

    def predict_model(model,newdata):
        return model.predict(newdata)

    phi0 = float(np.mean(model.predict(X_train)))

    explanation = explain(
        model=model,
        predict_model=predict_model,
        x_train=X_train,
        x_explain=evaluation_sample,
        approach="arf",  # arf is the most updated method they also have some kernel methods.
        phi0=phi0,
        seed=random_state,
    )

    attribution_result = np.asarray(explanation.to_shap().values,dtype=float)
    if attribution_result.shape != evaluation_sample.shape:
        raise ValueError("Unexpected SHAPr output shape.")
    return attribution_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_dir",default="LUCID/datasets/auto_mpg/prepared")
    parser.add_argument("--train_path",default=None)
    parser.add_argument("--evaluation_path",default=None)
    parser.add_argument("--model_path",default=None)
    parser.add_argument("--output_dir",default=None)
    parser.add_argument("--target_name",default="mpg")
    parser.add_argument("--fold_idx",type=int,default=0)
    parser.add_argument("--feature_types_path",default="LUCID/datasets/auto_mpg/prepared/feature_types.csv")
    parser.add_argument("--method_name",choices=["shap","lime","shapr"],default="shap")
    parser.add_argument("--replacement_sample_size",type=int,default=1000)
    parser.add_argument("--random_state",type=int,default=2026)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f"results/auto_mpg/{args.method_name}/outcome/seed_{args.random_state}"
    fold_dir = Path(args.prepared_dir) / f"fold_{args.fold_idx}"
    if args.train_path is None:
        args.train_path = fold_dir / "train.csv"
    if args.evaluation_path is None:
        args.evaluation_path = fold_dir / "outcome_reference.csv"
    if args.model_path is None:
        args.model_path = fold_dir / "predictive_model.json"
    run_attribution_experiment(train_path=args.train_path,evaluation_path=args.evaluation_path,model_path=args.model_path,output_dir=args.output_dir,target_name=args.target_name,fold_idx=args.fold_idx,feature_types_path=args.feature_types_path,method_name=args.method_name,replacement_sample_size=args.replacement_sample_size,random_state=args.random_state)

if __name__ == "__main__":
    main()
