# SAE Latent Reasoning

Cloud-ready workflow for discovering and steering latent reasoning modes in
generative AI systems using sparse autoencoder (SAE) measurements and
discrete causal representation learning (DCRL).

This repository is being prepared for NVIDIA cloud execution. The current
pipeline has been validated locally on a 600-trace Qwen3-8B pilot. It is not
yet a production H100 benchmark package; a small institutional GPU benchmark
will be used to calibrate final throughput, memory, and storage estimates
before the grant submission.

## Project Summary

The workflow starts from public math reasoning prompts, generates
chain-of-thought traces, extracts hidden states from a Qwen reasoning model,
encodes those activations with the official Qwen layer-18 TopK SAE, fits a
vector-valued DCRL measurement model over selected SAE features, and validates
learned latent variables through activation steering.

Preliminary local results show output-effective steering directions:

- K6 `Z5`: polynomial / quotient-remainder / Remainder-Theorem steering basin.
- K7 `Z1`: fraction / decimal / numerator-denominator steering basin.
- Random-vector controls do not form comparable target-specific basins.
- SAE-space contrastive baselines produce lower-level procedural steering
  basins, useful as a noisier baseline.

## Cloud Readiness

The workflow is organized into modular scripts:

1. `scripts/generate_cot.py` - batch CoT generation.
2. `scripts/extract_activations.py` - hidden-state extraction.
3. `scripts/encode_sae.py` - official Qwen SAE TopK encoding.
4. `scripts/fit_latent_model.py` - feature selection and DCRL fitting.
5. `scripts/run_steering.py` - residual or SAE-basis steering evaluation.
6. `scripts/run_microbenchmark.sh` - small benchmark driver for throughput,
   memory, and storage calibration.
7. `slurm/ginsburg_microbenchmark.sbatch` - template Slurm entry point for
   institutional GPU benchmarking.

For NVIDIA cloud execution, the intended environment starts from an NVIDIA NGC
PyTorch container and installs pinned Python dependencies from
`requirements.txt`. The scripts write resumable JSONL/NPZ/PT artifacts under
`results/`.

Data are generated from public reasoning benchmarks and contain no personal
information. Large generated traces, model weights, SAE checkpoints, and
derived activation files are not committed to this repository.

## Quick Start

Build the container:

```bash
docker build -t sae-latent-reasoning:latest .
```

Run a local dry run:

```bash
bash scripts/run_microbenchmark.sh --config configs/benchmark_small.yaml --dry-run
```

Run the benchmark on a CUDA machine:

```bash
bash scripts/run_microbenchmark.sh --config configs/benchmark_small.yaml
```

Submit the benchmark on a Slurm-managed GPU cluster after adjusting account,
partition, and module lines for the local environment:

```bash
sbatch slurm/ginsburg_microbenchmark.sbatch
```

## Expected GPU Workflow

```bash
python scripts/generate_cot.py --config configs/qwen3_8b.yaml
python scripts/extract_activations.py --config configs/qwen3_8b.yaml
python scripts/encode_sae.py --config configs/qwen3_8b.yaml
python scripts/fit_latent_model.py --config configs/qwen3_8b.yaml
python scripts/run_steering.py --config configs/qwen3_8b.yaml
```

Each command supports `--dry-run` to print the concrete command plan without
running the heavy model step.

The wrapper scripts currently delegate to the validated local pilot
implementations when this repository is kept adjacent to the original
`project/scripts` directory. For a clean standalone GitHub clone, the dry-run
planner still works, but non-dry GPU execution requires copying or porting the
heavy implementations into this repository. This is intentional for the grant
stage: the repo demonstrates workflow readiness without claiming a fully
benchmarked production H100 package.

## Repository Layout

```text
sae_latent_reasoning/
  README.md
  Dockerfile
  requirements.txt
  scripts/
    generate_cot.py
    extract_activations.py
    encode_sae.py
    fit_latent_model.py
    run_steering.py
    run_microbenchmark.sh
  slurm/
    ginsburg_microbenchmark.sbatch
  configs/
    qwen3_8b.yaml
    benchmark_small.yaml
  data/
    README.md
  results/
    README.md
  docs/
    cloud_readiness.md
```

## Honest Status

Validated locally:

- 600 generated reasoning traces.
- Qwen3-8B / Qwen3-8B-Base model family.
- Official Qwen SAE repository:
  `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`.
- Layer 18 SAE TopK-50 activations.
- 65,536 raw SAE features filtered to 100 selected DCRL features.
- K6/K7 DCRL fits and steering validation.

Pending before final proposal submission:

- CUDA/H100 microbenchmark.
- Final throughput estimate in tokens/sec and seconds/trace.
- Peak VRAM and storage-per-trace estimate.
- Final H100-hour budget table.
