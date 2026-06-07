# Cloud Readiness

The current pipeline has been validated locally on a 600-trace Qwen3-8B pilot.
The workflow consists of five modular stages: batch CoT generation, activation
extraction, official SAE encoding, latent-variable fitting, and activation
steering. For NVIDIA cloud execution, we use an NVIDIA NGC PyTorch container
with pinned Python dependencies and scripted batch jobs.

Data are generated from public reasoning benchmarks and contain no personal
information. Generated traces can be staged to cloud storage from
institution-managed research storage. GPU utilization, memory, throughput, and
storage will be logged for each benchmark job.

We do not yet claim production-ready H100 benchmarking. A small institutional
GPU benchmark will be used to calibrate:

- prompts/sec for CoT generation;
- tokens/sec for activation extraction;
- SAE encoding throughput;
- steering generations/sec;
- peak VRAM;
- storage per trace.
