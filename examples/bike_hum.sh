#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python -B -u examples/run_one_fold.py bike_sharing "$@"

# Full single fold, all methods in sequence:
# bash examples/bike_hum.sh --fold 0
# Smoke run:
# bash examples/bike_hum.sh --evaluation-limit 8 --replacement-samples 100
# Submit to Slurm after activating the environment:
# sbatch --cpus-per-task=1 --mem=16G --time=08:00:00 \
#   --output=results/bike_slurm_%j.log --wrap='bash examples/bike_hum.sh --fold 0'
# Submit just one method:
# sbatch --output=results/bike_drbart_%j.log \
#   --wrap='bash examples/bike_hum.sh --methods drbart --fold 0'
# Run in the background:
# nohup bash examples/bike_hum.sh > results/bike_background.log 2>&1 &
