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
        description="Generate or verify the public machine-readable registries"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root containing configs, src, and registries",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and reject any committed difference",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    _bootstrap(root)

    from qsa_benchmark.protocol.registry_export import (  # pylint: disable=import-outside-toplevel
        check_registry_files,
        registry_summary,
        verify_sha256sums,
        write_registry_files,
        write_sha256sums,
    )

    errors: list[str] = []
    summary_fields: dict[str, object] = {}
    try:
        if args.check:
            errors.extend(check_registry_files(root))
            errors.extend(verify_sha256sums(root))
        else:
            write_registry_files(root)
            write_sha256sums(root)
        if not errors:
            summary_fields = registry_summary(root)
    except Exception as exc:  # The command must fail closed with machine-readable output.
        errors.append(f"{type(exc).__name__}: {exc}")

    summary = {
        "mode": "check" if args.check else "write",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        **summary_fields,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
