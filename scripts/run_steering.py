#!/usr/bin/env python
"""Build and evaluate activation-steering vectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import load_config, project_script, run_command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_8b.yaml")
    parser.add_argument("--fit", required=False)
    parser.add_argument("--selected-X")
    parser.add_argument("--metadata")
    parser.add_argument("--cots")
    parser.add_argument(
        "--method",
        choices=["residual_caa", "sae_contrastive_all", "sae_contrastive_qmasked"],
        default="residual_caa",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]
    steer = cfg["steering"]
    out_dir = Path(args.output_dir or Path(cfg["project"]["output_dir"]) / f"steering_{args.method}")
    fit_path = args.fit or str(Path(cfg["project"]["output_dir"]) / "fits" / "BEST_FIT_PLACEHOLDER.npz")
    vectors_pt = out_dir / "latent_steering_vectors.pt"
    report = out_dir / "steering_vector_metadata.json"

    if args.method == "residual_caa":
        build_command = [
            sys.executable,
            str(project_script("17_build_contrastive_dcrl_steering_vectors.py")),
            "--config",
            args.config,
            "--fit",
            fit_path,
            "--metadata",
            args.metadata or data["sample_metadata_path"],
            "--cots",
            args.cots or data["cots_path"],
            "--output-pt",
            str(vectors_pt),
            "--report",
            str(report),
            "--examples-per-side",
            str(steer.get("examples_per_side", 8)),
            "--device",
            cfg["model"].get("device", "cuda"),
            "--dtype",
            cfg["model"].get("dtype", "bfloat16"),
        ]
    else:
        scope = "q_linked" if args.method == "sae_contrastive_qmasked" else "all"
        build_command = [
            sys.executable,
            str(project_script("17c_build_sae_contrastive_dcrl_steering_vectors.py")),
            "--config",
            args.config,
            "--fit",
            fit_path,
            "--selected-X",
            args.selected_X or data["selected_X_path"],
            "--metadata",
            args.metadata or data["sample_metadata_path"],
            "--output-pt",
            str(vectors_pt),
            "--report",
            str(report),
            "--feature-scope",
            scope,
            "--examples-per-side",
            str(steer.get("examples_per_side", 8)),
        ]
    run_command("build_steering_vectors", build_command, args.dry_run)

    probe_command = [
        sys.executable,
        str(project_script("15_run_semantic_steering_probe_qwen.py")),
        "--config",
        args.config,
        "--input",
        data["raw_path"],
        "--steering-pt",
        str(vectors_pt),
        "--output",
        str(out_dir / "steering_outputs.jsonl"),
        "--model-name",
        cfg["sae"]["model_name"],
        "--latents",
        "all",
        "--coefficients",
        *[str(x) for x in steer.get("coefficients", [16, 32, 64, 96])],
        "--max-new-tokens",
        str(steer.get("max_new_tokens", 128)),
        "--device",
        cfg["model"].get("device", "cuda"),
        "--dtype",
        cfg["model"].get("dtype", "bfloat16"),
    ]
    run_command("run_steering_probe", probe_command, args.dry_run)


if __name__ == "__main__":
    main()
