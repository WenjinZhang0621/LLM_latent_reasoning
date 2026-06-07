#!/usr/bin/env python
"""Batch CoT generation stage.

This wrapper is cloud-facing and resumable. In the current local pilot
workspace it delegates to the validated LM Studio/OpenAI-compatible generator.
For NVIDIA cloud use, replace the endpoint/model with an NVIDIA-hosted or
container-local inference service while keeping the JSONL interface unchanged.
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]
    command = [
        sys.executable,
        str(project_script("02_generate_cots_lmstudio.py")),
        "--config",
        args.config,
        "--input",
        args.input or data["raw_path"],
        "--output",
        args.output or data["cots_path"],
    ]
    run_command("generate_cot", command, args.dry_run)


if __name__ == "__main__":
    main()
