#!/usr/bin/env python
"""Hidden-state extraction stage.

The validated pilot extracts layer-18 Qwen residual activations while encoding
TopK SAE features. This wrapper keeps activation extraction as its own cloud
stage for scheduling and benchmarking.
"""

from __future__ import annotations

import argparse
import sys

from _common import load_config, project_script, run_command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_8b.yaml")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]
    model = cfg["model"]
    command = [
        sys.executable,
        str(project_script("03_extract_sae_features_qwen.py")),
        "--config",
        args.config,
        "--input",
        args.input or data["cots_path"],
        "--output",
        args.output or data["token_features_path"],
        "--layer",
        str(args.layer if args.layer is not None else model["layer"]),
    ]
    if args.max_samples is not None:
        command += ["--max-samples", str(args.max_samples)]
    run_command("extract_activations", command, args.dry_run)


if __name__ == "__main__":
    main()
