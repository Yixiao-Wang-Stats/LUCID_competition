# LUCID competition
During the cleaning process our code env follows:

```
cd /home/users/yw676/projects/VI/LUCID_competition

source /usr/xtmp/$USER/envs/lucid/bin/activate

export R_LIBS_USER=/home/users/yw676/Rlibs
export R_PROFILE_USER=/dev/null
export R_ENVIRON_USER=/dev/null
export PYTHONDONTWRITEBYTECODE=1
```


```
source /usr/xtmp/$USER/envs/lucid_install_test/bin/activate
```












# LUCID supplementary code











This directory contains the calculation package, datasets, experiment
scripts, and examples for reproducing one feature on one held-out fold.
All commands below run from this directory. All experiment outputs go to `results/`.

## Code structure

```text
package/src/
  LUCID_calculation/         losses and single-feature/group calculations
  predictive_model/         predictive-model interface and XGBoost wrapper
  sampler/                  permutation, DR-BART, CMR, GCMR, ARF, CSPFI,
                            clique, LOCO, and analytic oracle samplers
scripts/experiments_lucid/   experiment, attribution, uncertainty, and aggregation entry points
scripts/data_preparation/   DGP generation, known models, and data preparation helpers
datasets/dgp/               five prepared DGPs, compressed sensitivity data, fold/model specifications
datasets/bike_sharing/      bike data, prepared folds, frozen XGBoost models
examples/                   sequential single-fold examples and submission comments
env/                        Python requirements and R installation instructions/script
results/                    generated outputs (created at runtime)
```


## Create a new Python environment

Use Python 3.14 and R 4.5.2 for the documented setup.
R must be installed before installing `pyshapr`
because its Python bridge uses `rpy2`.

```bash
python3.14 -m venv env/.venv
source env/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r env/requirements.txt
```

`env/requirements.txt` pins the direct dependencies. For the complete Python
dependency list, install `env/requirements-lock.txt` instead.
The calculation scripts import PyMC and PyMC-BART even when another sampler is
selected. `pymc-bart==0.13.1` provides the `PosteriorSampler` import used here.

## Create an independent R library

Use a new library inside this directory; no personal R library is needed.
On Linux, building the R dependencies requires C/C++ and Fortran compilers,
GNU make, R development headers, and development libraries for curl, OpenSSL,
XML, and ICU.

```bash
mkdir -p env/R/library results/tmp
export R_LIBS_USER="$PWD/env/R/library"
export TMPDIR="$PWD/results/tmp"
Rscript --vanilla env/install_r.R
```

The installation script installs `arf`, `mlt`, `partykit`, `trtf`, and `shapr`
from CRAN, and `drbart` from its upstream repository. It also installs their
dependencies and writes the installed versions to `results/r_versions.csv`.
The reference R versions are listed in `env/r-packages.csv`; the DR-BART source
commit is pinned by the installer. Other R packages are installed from CRAN.
DR-BART follows the [upstream installation instructions](https://github.com/vittorioorlandi/drbart).
ARF is available on [CRAN](https://cran.r-project.org/package=arf), as is
[trtf](https://cran.r-project.org/package=trtf).

After installation, use a clean R startup so that local profile files cannot
inject a different library. Set the same library each time you activate the
Python environment:

```bash
source env/.venv/bin/activate
export R_LIBS_USER="$PWD/env/R/library"
export R_PROFILE_USER=/dev/null
export R_ENVIRON_USER=/dev/null
export PYTHONDONTWRITEBYTECODE=1
mkdir -p results/tmp
export TMPDIR="$PWD/results/tmp"
```

## Smallest running units

Prepare all sensitivity folds with `python -B -m scripts.data_preparation.prepare_submission`.
The submission stores their data columns once in a compressed archive; preparation
restores the original CSVs using the recorded fold assignments and verifies every
file's SHA-256 checksum. It does not generate new observations or train models.
The DGP sensitivity experiments use the supplied analytic models. Their unused
XGBoost models are omitted. The five baseline DGPs and bike sharing remain prepared.
Both examples automatically run preparation for the selected fold before any method;
already prepared files are reused.

Each command below runs one method on one feature and one fold. A positive
evaluation limit uses the first held-out rows; training still uses the full
training fold. These reduced runs are smoke checks, not full experimental results.

```bash
# Synthetic regime_dependent DGP: x_1 is named X1 in the supplied CSV.
bash examples/dgp_x1.sh --fold 0 --methods permutation \
  --evaluation-limit 8 --replacement-samples 100

# Bike sharing: hum is feature index 3; the outcome is y.
bash examples/bike_hum.sh --fold 0 --methods permutation \
  --evaluation-limit 8 --replacement-samples 100

# Sensitivity variant: same example, with 20 added irrelevant features.
bash examples/dgp_x1.sh \
  --prepared-dir datasets/dgp/irrelevent_feature_sensitivity/regime_dependent_20/prepared \
  --fold 0 --methods permutation --evaluation-limit 8 --replacement-samples 100
```

To run every method sequentially once on the complete fold:

```bash
bash examples/dgp_x1.sh --fold 0
bash examples/bike_hum.sh --fold 0
```

Both examples run permutation, DR-BART, CMR, GCMR, ARF, CSPFI, clique, LOCO,
SHAP, LIME, and SHAPr. The DGP example also runs analytic `optimal_lucid`
and `optimal_loco`. Analytic oracles are unavailable for bike sharing.
SHAP/LIME/SHAPr explain the model prediction and are attribution baselines;
their values are not loss differences and should not be equated to LUCID.
Attribution scripts compute all features together; the example additionally
saves the requested feature column separately.

The DGP uses its known optimal predictive model and `prediction_reference.csv`
with evaluation target `prediction`, following the original experiment entry
point. Bike sharing uses the frozen fold-specific XGBoost model and
`outcome_reference.csv` with target `y`. Preparation restores missing packaged data;
predictive models are not refitted. Conditional samplers and LOCO are fitted as usual.
DR-BART retains the original 2,000 burn-in and 2,000 sampling iterations.
It can take substantially longer than the other methods.

Progress prints `[i/N] START` followed by `[i/N] DONE` or `FAILED` and streams
the method output. The default stops on the first failure; use
`--continue-on-error` to collect statuses for every method. The command returns
a nonzero exit code if any method failed.

## Outputs and other submission options

For each case, the output structure is:

```text
results/<case>/<feature>/seed_2026/
  <method>/fold_0/feature_<index>.npy
  <method>/fold_0/sampler_fit/         methods that save a fit
  <method>/fold_0/attributions.npy    attribution methods: all feature columns
  <method>_fold_0.log
  status_fold_0.json
```

`feature_0.npy` corresponds to DGP X1; `feature_3.npy` corresponds to bike hum.
Status records contain exit codes and elapsed times. Logs redact the current
supplementary root and home/temporary paths. Shared exchange/cache directories
are under `results/`. Multiple folds can be run using `--fold 0` through `--fold 4`.
The example shell scripts include commented Slurm, separate-method, and
background submission commands. Activate the environment and export the R
library before submission. Scheduler time/memory limits are examples and may
need adjustment to your cluster.

The original entry points can also be called with `python -B -m
scripts.experiments_lucid.experiments_lucid --help`. Always pass prepared-data,
target, feature-type and output paths explicitly; the inherited defaults describe
a different dataset. Aggregation scripts expect all features and folds, so they
are not needed for these one-feature, one-fold examples.
