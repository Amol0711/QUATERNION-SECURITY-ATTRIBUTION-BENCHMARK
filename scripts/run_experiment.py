#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

def _bootstrap(root: Path) -> None:
    source = root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def invoke(root: Path, *arguments: str) -> None:
    command = [sys.executable, "-m", "qsa_benchmark.protocol.experiment", *arguments]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(command, cwd=root, env=env, check=True)


def run_one(root: Path, run_root: Path, label: str, include_timing: bool) -> None:
    from qsa_benchmark.protocol.config import load_protocol_config
    config = load_protocol_config(root / "configs/protocol/experiment.json", root / "configs/protocol/experiment.schema.json")
    common = ["--repo-root", str(root), "--run-root", str(run_root), "--run-label", label]
    invoke(root, "init", *common)
    for method_id in config.methods:
        invoke(root, "differential-shard", *common, "--method", method_id)
    invoke(root, "merge-differential", *common)
    invoke(root, "leakage", *common)
    invoke(root, "attacks", *common)
    if include_timing:
        for method_id in config.methods:
            invoke(root, "timing-shard", *common, "--method", method_id)
        invoke(root, "merge-timing", *common)
        invoke(root, "cost-model", *common)


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute two independent full protocol runs")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    _bootstrap(root)
    results = (args.results_root or root / "results/experiment").resolve()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    primary_root = results / "primary_execution"
    independent_root = results / "independent_execution"
    run_one(root, primary_root, "primary_execution", not args.skip_timing)
    run_one(root, independent_root, "independent_execution", not args.skip_timing)
    if args.skip_timing:
        from qsa_benchmark.validation.deterministic import (
            compare_execution_trees,
            verify_execution_tree,
        )
        primary = verify_execution_tree(root, primary_root)
        independent = verify_execution_tree(root, independent_root)
        cross = compare_execution_trees(root, primary_root, independent_root)
        summary = {
            "primary": {key: value for key, value in primary.items() if key != "checks"},
            "independent": {key: value for key, value in independent.items() if key != "checks"},
            "cross_execution": {key: value for key, value in cross.items() if key != "checks"},
        }
        summary["status"] = "PASS" if all(
            item["status"] == "PASS" for item in (primary, independent, cross)
        ) else "FAIL"
        results.mkdir(parents=True, exist_ok=True)
        (results / "deterministic_reconciliation.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if summary["status"] != "PASS":
            raise RuntimeError("deterministic execution trees do not match the frozen reference")
    else:
        invoke(root, "reconcile", "--repo-root", str(root), "--results-root", str(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
