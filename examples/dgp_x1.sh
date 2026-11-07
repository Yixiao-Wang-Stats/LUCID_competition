#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python -B -u examples/run_one_fold.py dgp "$@"

# Full single fold, all methods in sequence:
# bash examples/dgp_x1.sh --fold 0
# Smoke run (full training fold, first eight held-out observations):
# bash examples/dgp_x1.sh --evaluation-limit 8 --replacement-samples 100
# Submit to Slurm from the supplementary root after activating the environment:
# sbatch --cpus-per-task=1 --mem=16G --time=08:00:00 \
#   --output=results/dgp_slurm_%j.log --wrap='bash examples/dgp_x1.sh --fold 0'
# Submit separate methods (each job fits its own sampler):
# for method in permutation drbart cmr gcmr arf cspfi clique loco optimal_lucid optimal_loco shap lime shapr; do
#   sbatch --output="results/dgp_${method}_%j.log" \
#     --wrap="bash examples/dgp_x1.sh --methods $method --fold 0"
# done
# Run another fold: bash examples/dgp_x1.sh --fold 1
