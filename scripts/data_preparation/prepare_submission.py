"""Restore the packaged sensitivity CSVs using their recorded fold assignments."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SENSITIVITY = ROOT / "datasets/dgp/irrelevent_feature_sensitivity"


def _write_csv(path, header, columns, indices, expected_hash):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows([column[i] for column in columns] for i in indices)
    payload = stream.getvalue().encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError("Restored CSV failed its recorded checksum.")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(prepared_dir, fold_idx=None):
    prepared = Path(prepared_dir)
    if not prepared.is_absolute():
        prepared = ROOT / prepared
    prepared = prepared.resolve()
    relative = prepared.relative_to(ROOT)
    folds = list(range(5)) if fold_idx is None else [fold_idx]
    manifest_path = SENSITIVITY / "data_manifest.json"
    sensitivity = prepared.parent.parent == SENSITIVITY
    specification = None
    if sensitivity:
        manifest = json.loads(manifest_path.read_text())
        if manifest["version"] != 1:
            raise ValueError("Unsupported sensitivity data format.")
        specification = manifest["datasets"][prepared.parent.name]

    filenames = ["train.csv", "outcome_reference.csv"]
    if "dgp" in relative.parts:
        filenames.append("prediction_reference.csv")
    required = [prepared / "feature_types.csv"]
    model_name = "optimal_model.json" if "dgp" in relative.parts else "predictive_model.json"
    required.extend(prepared / f"fold_{k}" / model_name for k in folds)
    if sensitivity:
        required.append(prepared / "fold_indices.csv")
        required.extend(prepared / f"fold_{k}" / f"optimal_reduced_model_{j}.json"
                        for k in folds for j in range(len(specification["header"]) - 1))
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing packaged file: {path.relative_to(ROOT)}")

    pending = []
    for k in folds:
        for name in filenames:
            path = prepared / f"fold_{k}" / name
            expected = specification["sha256"][f"fold_{k}/{name}"] if sensitivity else None
            if not path.is_file() or (expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected):
                pending.append((k, name, path, expected))
    if pending and not sensitivity:
        raise FileNotFoundError(f"Missing prepared data: {pending[0][2].relative_to(ROOT)}")

    if pending:
        import numpy as np

        with (prepared / "fold_indices.csv").open(newline="") as handle:
            assignments = {int(row["observation_idx"]): int(row["fold_idx"])
                           for row in csv.DictReader(handle)}
        if set(assignments) != set(range(specification["n_rows"])) or set(assignments.values()) != set(range(5)):
            raise ValueError("Invalid recorded fold assignments.")
        with np.load(SENSITIVITY / "data_columns.npz", allow_pickle=False) as data:
            columns = [data[key].tobytes().decode("ascii").splitlines()
                       for key in specification["columns"]]
            prediction_columns = [data[key].tobytes().decode("ascii").splitlines()
                                  for key in specification["prediction_columns"]]
        if any(len(column) != specification["n_rows"] for column in columns + prediction_columns):
            raise ValueError("Packaged column has an incorrect row count.")
        for k, name, path, expected in pending:
            training = name == "train.csv"
            indices = sorted(i for i, fold in assignments.items() if (fold != k if training else fold == k))
            prediction = name == "prediction_reference.csv"
            _write_csv(path, specification["prediction_header"] if prediction else specification["header"],
                       prediction_columns if prediction else columns, indices, expected)
    print(f"[prepare] {relative}: ready ({len(pending)} CSVs restored)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=None,
                        help="Check/prepare one dataset; the default restores all sensitivity datasets")
    parser.add_argument("--fold", type=int, choices=range(5), default=None,
                        help="Prepare one fold; the default prepares all five")
    args = parser.parse_args()
    if args.prepared_dir is not None:
        prepare(args.prepared_dir, args.fold)
    else:
        manifest = json.loads((SENSITIVITY / "data_manifest.json").read_text())
        for name in sorted(manifest["datasets"]):
            prepare(SENSITIVITY / name / "prepared", args.fold)


if __name__ == "__main__":
    main()
