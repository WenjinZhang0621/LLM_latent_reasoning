#!/usr/bin/env bash
#SBATCH --job-name=sae_microbench
#SBATCH --output=results/benchmark_small/logs/slurm_%j.out
#SBATCH --error=results/benchmark_small/logs/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/benchmark_small/logs

# Adjust these lines for the institutional cluster environment.
# module purge
# module load cuda
# module load anaconda
# source ~/.bashrc
# conda activate sae_latent_reasoning

nvidia-smi
python scripts/system_check.py --config configs/benchmark_small.yaml
bash scripts/run_microbenchmark.sh --config configs/benchmark_small.yaml
