#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the optional quaternion-Feistel accelerator")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cc", default="gcc")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    source = root / "src/qsa_benchmark/benchmark/qfeistel_fast.c"
    output = root / "src/qsa_benchmark/benchmark/_qfeistel_fast.so"
    command = [args.cc, "-O3", "-fPIC", "-shared", "-fopenmp", str(source), "-o", str(output), "-lcrypto"]
    subprocess.run(command, check=True)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
