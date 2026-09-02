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
    parser = argparse.ArgumentParser(
        description="Verify fixed-answer, malformed-object, and active-modification reference layers"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="optional machine-readable output file",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    _bootstrap(root)

    from qsa_benchmark.validation.active_modification import (
        active_modification_counts,
        build_active_modification_rows,
    )
    from qsa_benchmark.validation.known_answers import (
        known_answer_summary,
        verify_known_answer_assets,
    )
    from qsa_benchmark.validation.malformed import (
        malformed_object_summary,
        verify_malformed_object_assets,
    )

    failures: list[str] = []

    known_errors = verify_known_answer_assets(root)
    known = known_answer_summary(root) if not known_errors else {}
    known_passed = not known_errors and int(known.get("case_count", 0)) == 24
    if not known_passed:
        failures.append("fixed known-answer verification failed")

    malformed_errors = verify_malformed_object_assets(root)
    malformed = malformed_object_summary(root) if not malformed_errors else {}
    malformed_passed = not malformed_errors and int(malformed.get("case_count", 0)) == 33
    if not malformed_passed:
        failures.append("malformed-object verification failed")

    try:
        active_rows = build_active_modification_rows(root)
        active_counts = active_modification_counts(active_rows)
        active_passed = (
            len(active_rows) == 180
            and int(active_counts.get("authenticated_cases", 0)) == 73
            and int(active_counts.get("authenticated_rejections", 0)) == 73
        )
    except Exception as exc:  # pragma: no cover - exercised on repository corruption
        active_rows = []
        active_counts = {"error": f"{type(exc).__name__}: {exc}"}
        active_passed = False
    if not active_passed:
        failures.append("active-modification verification failed")

    summary = {
        "verification_id": "QSA-REFERENCE-LAYERS-VERIFICATION-V1",
        "status": "PASS" if not failures else "FAIL",
        "exit_code": 0 if not failures else 2,
        "failures": failures,
        "known_answers": {
            "status": "PASS" if known_passed else "FAIL",
            "errors": known_errors,
            "summary": known,
        },
        "malformed_objects": {
            "status": "PASS" if malformed_passed else "FAIL",
            "errors": malformed_errors,
            "summary": malformed,
        },
        "active_modifications": {
            "status": "PASS" if active_passed else "FAIL",
            "row_count": len(active_rows),
            "counts": active_counts,
        },
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output_json is not None:
        target = args.output_json.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return int(summary["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
