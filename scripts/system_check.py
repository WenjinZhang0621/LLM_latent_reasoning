#!/usr/bin/env python
"""Print environment information for cloud readiness and benchmarks."""

from __future__ import annotations

import argparse
import json

from _common import environment_snapshot, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_small.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    info = environment_snapshot()
    info["config"] = args.config
    info["project"] = cfg.get("project", {})
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
