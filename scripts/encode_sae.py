#!/usr/bin/env python
"""SAE TopK encoding and sample-matrix construction stage."""

from __future__ import annotations

import argparse
import sys

from _common import load_config, project_script, run_command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_8b.yaml")
    parser.add_argument("--token-features")
    parser.add_argument("--cots")
    parser.add_argument("--output-npz")
    parser.add_argument("--metadata-output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]
    command = [
        sys.executable,
        str(project_script("04_build_X_matrix.py")),
        "--config",
        args.config,
        "--input",
        args.token_features or data["token_features_path"],
        "--cots",
        args.cots or data["cots_path"],
        "--output-npz",
        args.output_npz or data["sample_features_path"],
        "--metadata-output",
        args.metadata_output or data["sample_metadata_path"],
    ]
    run_command("encode_sae", command, args.dry_run)


if __name__ == "__main__":
    main()
