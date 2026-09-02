from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .config import load_config
from .registry import method_registry
from .runner import prepare_data, run_benchmark


def repository_root() -> Path:
    # Installed CLI commands that need the repository accept explicit config paths;
    # source-tree execution resolves the repository from this module.
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qsa-benchmark", description="Quaternion security-attribution benchmark runner")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare-data", "run", "reproduce", "validate-config"):
        item = sub.add_parser(command)
        item.add_argument("--config", required=True)
    sub.add_parser("registry")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "registry":
        print(json.dumps(method_registry(), indent=2, sort_keys=True))
        return 0
    config = load_config(args.config)
    root = config.path.parents[2]
    if args.command == "validate-config":
        print(json.dumps({"valid": True, "benchmark_id": config.benchmark_id, "method_count": len(config.methods)}, indent=2))
        return 0
    if args.command == "prepare-data":
        print(prepare_data(config, root))
        return 0
    if args.command == "reproduce" and config.output_root.exists():
        shutil.rmtree(config.output_root)
    if args.command == "reproduce":
        prepare_data(config, root)
    paths = run_benchmark(config, root)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    print(json.dumps({"outputs": {k: str(v) for k, v in paths.items()}, "validation": summary["validation"]}, indent=2))
    return 0 if summary["validation"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
