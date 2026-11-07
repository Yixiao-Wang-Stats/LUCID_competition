"""Run each method once, in sequence, for one feature and one prepared fold."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SAMPLERS = ["permutation", "drbart", "cmr", "gcmr", "arf", "cspfi", "clique", "loco"]
ATTRIBUTIONS = ["shap", "lime", "shapr"]


def anonymous(text):
    text = text.replace(str(ROOT), "<supplementary>")
    # Tracebacks and dependency diagnostics can contain machine-specific paths.
    return re.sub(r"/(?:home|usr/xtmp|tmp)/[^\s\"'<>]+", "<local-path>", text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["dgp", "bike_sharing"])
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prepared-dir", type=Path, default=None,
                        help="Override the dataset, including a DGP sensitivity variant")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--evaluation-limit", type=int, default=0,
                        help="0 uses the full held-out fold; positive values make a smoke run")
    parser.add_argument("--replacement-samples", type=int, default=1000)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.evaluation_limit < 0 or args.replacement_samples < 1:
        parser.error("evaluation-limit must be nonnegative; replacement-samples must be positive")
    os.chdir(ROOT)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["LUCID_EXCHANGE_DIR"] = "results/tmp"
    os.environ["MPLCONFIGDIR"] = "results/cache/matplotlib"
    os.environ["NUMBA_CACHE_DIR"] = "results/cache/numba"
    os.environ["PYTENSOR_FLAGS"] = "base_compiledir=results/cache/pytensor"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    dgp = args.case == "dgp"
    prepared = args.prepared_dir or Path("datasets/dgp/regime_dependent/prepared" if dgp else "datasets/bike_sharing/prepared")
    subprocess.run([sys.executable, "-B", "-m", "scripts.data_preparation.prepare_submission",
                    "--prepared-dir", str(prepared), "--fold", str(args.fold)], check=True)
    for directory in ["results/tmp", "results/cache/matplotlib", "results/cache/numba", "results/cache/pytensor"]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    feature, target = ("X1", "target") if dgp else ("hum", "y")
    fold = prepared / f"fold_{args.fold}"
    with (fold / "train.csv").open() as stream:
        features = [name for name in next(csv.reader(stream)) if name != target]
    feature_idx = features.index(feature)
    methods = args.methods or SAMPLERS + (["optimal_lucid", "optimal_loco"] if dgp else []) + ATTRIBUTIONS
    available = SAMPLERS + (["optimal_lucid", "optimal_loco"] if dgp else []) + ATTRIBUTIONS
    if any(method not in available for method in methods):
        parser.error("Available methods: " + ", ".join(available))
    output_case = prepared.parent.name if args.prepared_dir is not None else args.case
    output = Path("results") / output_case / feature / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    evaluation = fold / ("prediction_reference.csv" if dgp else "outcome_reference.csv")
    if args.evaluation_limit:
        destination = output / f"evaluation_fold_{args.fold}_first_{args.evaluation_limit}.csv"
        with evaluation.open() as source, destination.open("w", newline="") as sink:
            reader, writer = csv.reader(source), csv.writer(sink)
            writer.writerow(next(reader))
            for i, row in enumerate(reader):
                if i >= args.evaluation_limit:
                    break
                writer.writerow(row)
        evaluation = destination
    records = []
    for i, method in enumerate(methods, 1):
        print(f"[{i}/{len(methods)}] START {args.case}: {feature}, fold {args.fold}, {method}", flush=True)
        attribution = method in ATTRIBUTIONS
        module = "scripts.experiments_lucid." + ("experiments_attribution" if attribution else "experiments_lucid")
        method_output = output / method
        command = [sys.executable, "-B", "-u", "-m", module,
                   "--prepared_dir", str(prepared), "--train_path", str(fold / "train.csv"),
                   "--evaluation_path", str(evaluation), "--target_name", target,
                   "--fold_idx", str(args.fold), "--feature_types_path", str(prepared / "feature_types.csv"),
                   "--output_dir", str(method_output), "--random_state", str(args.seed),
                   "--replacement_sample_size", str(args.replacement_samples),
                   "--method_name" if attribution else "--sampler_name", method]
        if not attribution:
            command += ["--feature_idx", str(feature_idx), "--evaluation_target_name", "prediction" if dgp else target]
        started = time.monotonic()
        log = output / f"{method}_fold_{args.fold}.log"
        with log.open("w") as sink:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                line = anonymous(line)
                print(line, end="", flush=True)
                sink.write(line)
            code = process.wait()
        # Attribution scripts return all columns; retain the selected feature separately.
        if code == 0 and attribution:
            import numpy as np
            result_dir = method_output / f"fold_{args.fold}"
            values = np.load(result_dir / "attributions.npy")
            np.save(result_dir / f"feature_{feature_idx}.npy", values[:, feature_idx])
        records.append({"method": method, "fold": args.fold, "feature": feature,
                        "exit_code": code, "seconds": round(time.monotonic() - started, 2),
                        "log": str(log), "evaluation_limit": args.evaluation_limit,
                        "replacement_samples": args.replacement_samples})
        (output / f"status_fold_{args.fold}.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"[{i}/{len(methods)}] {'DONE' if code == 0 else 'FAILED'} {method}", flush=True)
        if code and not args.continue_on_error:
            sys.exit(code)
    sys.exit(1 if any(record["exit_code"] for record in records) else 0)


if __name__ == "__main__":
    main()
