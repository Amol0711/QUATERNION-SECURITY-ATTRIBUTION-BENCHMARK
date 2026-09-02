#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap(root: Path) -> None:
    source = root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic benchmark corpora")
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 96, 192, 256, 512])
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    _bootstrap(root)
    from qsa_benchmark.benchmark.datasets import build_corpora, write_manifests
    for size in args.sizes:
        if size < 16:
            parser.error("every size must be at least 16")
        corpus_root = root / "data/generated" / f"{size}x{size}"
        manifest_dir = root / "data/manifests" / f"{size}x{size}"
        records = build_corpora(corpus_root, size=(size, size))
        write_manifests(records, manifest_dir, size=(size, size))
        print(manifest_dir / "dataset_manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
