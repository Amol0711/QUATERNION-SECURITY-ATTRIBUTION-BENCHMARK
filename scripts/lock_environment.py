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
    parser = argparse.ArgumentParser(description="Verify or explicitly replace the frozen software-environment lock")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--update", action="store_true")
    parser.add_argument("--policy", choices=("exact", "packages", "report"), default="exact")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    _bootstrap(root)
    from qsa_benchmark.validation.environment import compare_environment, write_environment_lock
    if args.update:
        write_environment_lock(root)
        from qsa_benchmark.protocol.registry_export import write_sha256sums
        write_sha256sums(root)
    summary = compare_environment(root, args.policy)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
