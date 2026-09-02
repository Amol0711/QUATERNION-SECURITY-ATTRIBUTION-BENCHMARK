#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def invoke(root: Path, *arguments: str) -> None:
    command = [sys.executable, "-m", "qsa_benchmark.protocol.targeted_validation", *arguments]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(command, cwd=root, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute two independent targeted-validation runs")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    results = (args.results_root or root / "results/targeted_validation").resolve()
    for label in ("primary_execution", "independent_execution"):
        invoke(
            root,
            "run",
            "--repo-root", str(root),
            "--run-root", str(results / label),
            "--run-label", label,
        )
    invoke(root, "reconcile", "--repo-root", str(root), "--results-root", str(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
