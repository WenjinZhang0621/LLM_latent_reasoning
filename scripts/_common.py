#!/usr/bin/env python
"""Shared helpers for the portable workflow scripts."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def print_plan(name: str, command: list[str], dry_run: bool) -> None:
    print(f"[{name}] command:")
    print(" ".join(command))
    if dry_run:
        print(f"[{name}] dry-run only; command was not executed.")


def run_command(name: str, command: list[str], dry_run: bool) -> None:
    print_plan(name, command, dry_run)
    if dry_run:
        return
    start = time.time()
    subprocess.run(command, check=True)
    elapsed = time.time() - start
    print(f"[{name}] completed in {elapsed:.2f} seconds")


def environment_snapshot() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_count"] = int(torch.cuda.device_count())
            info["cuda_device_name_0"] = torch.cuda.get_device_name(0)
            info["cuda_capability_0"] = torch.cuda.get_device_capability(0)
    except Exception as exc:
        info["torch_error"] = repr(exc)
    return info


def project_script(script_name: str) -> Path:
    """Resolve the validated local pilot script when this repo is nested.

    The grant repo is intentionally lightweight. During local development in
    the original workspace, these wrappers can call the validated pilot scripts
    in ../project/scripts. In a clean GitHub clone, this function raises a clear
    error and the script still supports --dry-run for command planning.
    """

    candidate = repo_root().parent / "project" / "scripts" / script_name
    if not candidate.exists():
        print(
            f"[warning] validated pilot script is not present at {candidate}. "
            "Dry-run command planning can continue, but non-dry execution "
            "requires porting/copying the heavy implementation or keeping the "
            "original project/scripts directory adjacent to this repo.",
            file=sys.stderr,
        )
    return candidate
