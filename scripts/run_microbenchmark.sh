#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/benchmark_small.yaml"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON:-python3}"

EXTRA=()
if [[ "$DRY_RUN" == "1" ]]; then
  EXTRA=(--dry-run)
fi

mkdir -p results/benchmark_small/logs

"$PYTHON_BIN" scripts/system_check.py --config "$CONFIG" | tee results/benchmark_small/logs/system_check.json
"$PYTHON_BIN" scripts/generate_cot.py --config "$CONFIG" "${EXTRA[@]}" 2>&1 | tee results/benchmark_small/logs/01_generate_cot.log
"$PYTHON_BIN" scripts/extract_activations.py --config "$CONFIG" --max-samples 10 "${EXTRA[@]}" 2>&1 | tee results/benchmark_small/logs/02_extract_activations.log
"$PYTHON_BIN" scripts/encode_sae.py --config "$CONFIG" "${EXTRA[@]}" 2>&1 | tee results/benchmark_small/logs/03_encode_sae.log
"$PYTHON_BIN" scripts/fit_latent_model.py --config "$CONFIG" "${EXTRA[@]}" 2>&1 | tee results/benchmark_small/logs/04_fit_latent_model.log
"$PYTHON_BIN" scripts/run_steering.py --config "$CONFIG" --method residual_caa "${EXTRA[@]}" 2>&1 | tee results/benchmark_small/logs/05_run_steering.log

echo "Microbenchmark driver completed. Inspect results/benchmark_small/logs."
