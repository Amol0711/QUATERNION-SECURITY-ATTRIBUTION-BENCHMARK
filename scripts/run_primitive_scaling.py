#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap(root: Path) -> None:
    source = root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute the registered primitive-scaling protocol")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--run-label", default="primary_execution")
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    _bootstrap(root)
    from qsa_benchmark.protocol.primitive_scaling import execute_primitive_scaling
    output = (args.results_root or root / "results/primitive_scaling" / args.run_label).resolve()
    result = execute_primitive_scaling(root, output, run_label=args.run_label, timing=bool(args.timing))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
