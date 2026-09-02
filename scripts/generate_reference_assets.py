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


def _mode(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="verify committed assets without changing files")
    group.add_argument("--update", action="store_true", help="explicitly replace committed reference assets")


def main() -> int:
    parser = argparse.ArgumentParser(description="Explicitly update or verify frozen reference assets")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="asset", required=True)
    known = subparsers.add_parser("known-answers", description="manage the 24 fixed known-answer cases")
    _mode(known)
    malformed = subparsers.add_parser("malformed-objects", description="manage canonical-parser rejection vectors")
    _mode(malformed)
    deterministic = subparsers.add_parser("deterministic-artifacts", description="manage the 37 frozen deterministic artifacts")
    _mode(deterministic)
    deterministic.add_argument("--run-root", type=Path, help="one completed execution tree to verify")
    deterministic.add_argument("--results-root", type=Path, help="directory containing primary_execution and independent_execution")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    _bootstrap(root)

    from qsa_benchmark.protocol.registry_export import check_registry_files, verify_sha256sums, write_sha256sums
    errors: list[str] = []
    fields: dict[str, object] = {}
    try:
        if args.asset == "known-answers":
            from qsa_benchmark.validation.known_answers import known_answer_summary, verify_known_answer_assets, write_known_answer_files
            if args.update:
                errors.extend(check_registry_files(root))
                if not errors:
                    write_known_answer_files(root)
                    write_sha256sums(root)
            errors.extend(verify_known_answer_assets(root))
            if not errors:
                fields = known_answer_summary(root)
        elif args.asset == "malformed-objects":
            from qsa_benchmark.validation.known_answers import verify_known_answer_assets
            from qsa_benchmark.validation.malformed import malformed_object_summary, verify_malformed_object_assets, write_malformed_object_files
            if args.update:
                errors.extend(check_registry_files(root))
                errors.extend(verify_known_answer_assets(root))
                if not errors:
                    write_malformed_object_files(root)
                    write_sha256sums(root)
            errors.extend(verify_malformed_object_assets(root))
            if not errors:
                fields = malformed_object_summary(root)
        else:
            from qsa_benchmark.validation.deterministic import (
                compare_execution_trees,
                deterministic_reference_summary,
                verify_execution_tree,
                verify_static_artifact_contract,
                write_artifact_manifest_from_results,
            )
            if args.update:
                if args.results_root is None:
                    raise ValueError("--update requires --results-root with two completed execution trees")
                write_artifact_manifest_from_results(root, args.results_root)
                write_sha256sums(root)
            errors.extend(verify_static_artifact_contract(root))
            if args.run_root is not None:
                result = verify_execution_tree(root, args.run_root)
                if result["status"] != "PASS":
                    errors.append("completed execution tree differs from the frozen deterministic reference")
                fields["run_verification"] = {key:value for key,value in result.items() if key != "checks"}
            if args.results_root is not None:
                primary = Path(args.results_root).resolve() / "primary_execution"
                independent = Path(args.results_root).resolve() / "independent_execution"
                cross = compare_execution_trees(root, primary, independent)
                first = verify_execution_tree(root, primary)
                second = verify_execution_tree(root, independent)
                if any(item["status"] != "PASS" for item in (cross, first, second)):
                    errors.append("two-tree deterministic reference verification failed")
                fields["two_tree_verification"] = {
                    "primary": {key:value for key,value in first.items() if key != "checks"},
                    "independent": {key:value for key,value in second.items() if key != "checks"},
                    "cross_execution": {key:value for key,value in cross.items() if key != "checks"},
                }
            fields.update(deterministic_reference_summary(root))
        errors.extend(verify_sha256sums(root))
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    summary = {"asset": args.asset, "mode": "update" if args.update else "check", "status": "PASS" if not errors else "FAIL", "errors": errors, **fields}
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
