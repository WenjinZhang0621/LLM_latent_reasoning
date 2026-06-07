# Proposal Cloud-Readiness Paragraph

Cloud readiness. The current pipeline has been validated locally on a
600-trace Qwen3-8B pilot. The workflow is modularized into scripts for
chain-of-thought generation, activation extraction, official Qwen SAE
encoding, latent-model fitting, and activation steering. For NVIDIA cloud
execution, we will use an NVIDIA NGC PyTorch container with pinned
dependencies and scripted batch jobs. Data are generated from public reasoning
benchmarks and contain no personal information. We are currently requesting
Columbia Ginsburg GPU access to run a small benchmark for throughput, memory,
and storage calibration before final submission.
