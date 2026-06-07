#!/usr/bin/env python
"""Feature selection and DCRL latent-model fitting stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import load_config, project_script, run_command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_8b.yaml")
    parser.add_argument("--sample-features")
    parser.add_argument("--sample-metadata")
    parser.add_argument("--selected-X")
    parser.add_argument("--selected-features")
    parser.add_argument("--fits-dir")
    parser.add_argument("--model-selection")
    parser.add_argument("--J", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]
    dcrl = cfg["dcrl"]
    out_dir = Path(cfg["project"]["output_dir"])
    selected_x = args.selected_X or data["selected_X_path"]
    selected_features = args.selected_features or data["selected_features_path"]

    select_command = [
        sys.executable,
        str(project_script("05_select_sae_features_for_dcrl.py")),
        "--config",
        args.config,
        "--input",
        args.sample_features or data["sample_features_path"],
        "--metadata",
        args.sample_metadata or data["sample_metadata_path"],
        "--output",
        selected_x,
        "--features-output",
        selected_features,
        "--J",
        str(args.J if args.J is not None else dcrl["J"]),
    ]
    run_command("select_sae_features", select_command, args.dry_run)

    fit_command = [
        sys.executable,
        str(project_script("06_fit_dcrl_hurdle_em.py")),
        "--config",
        args.config,
        "--input",
        selected_x,
        "--selected-features",
        selected_features,
        "--fits-dir",
        args.fits_dir or str(out_dir / "fits"),
        "--summary-output",
        args.model_selection or str(out_dir / "model_selection.csv"),
    ]
    run_command("fit_dcrl", fit_command, args.dry_run)


if __name__ == "__main__":
    main()
