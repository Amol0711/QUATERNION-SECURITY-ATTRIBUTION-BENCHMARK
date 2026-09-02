from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jsonschema
import yaml

CONFIG_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["benchmark_id", "master_seed", "master_key_hex", "methods", "dataset", "execution", "outputs"],
    "properties": {
        "benchmark_id": {"type": "string", "minLength": 1},
        "master_seed": {"type": "integer", "minimum": 0},
        "master_key_hex": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$"},
        "methods": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string"}},
        "dataset": {
            "type": "object", "additionalProperties": False,
            "required": ["image_size", "splits", "corpora", "max_records_per_corpus"],
            "properties": {
                "image_size": {"type": "array", "items": {"type": "integer", "minimum": 16}, "minItems": 2, "maxItems": 2},
                "splits": {"type": "array", "items": {"enum": ["development", "validation", "test"]}},
                "corpora": {"type": "array", "items": {"enum": ["synthetic", "natural"]}},
                "max_records_per_corpus": {"type": "integer", "minimum": 1},
            },
        },
        "execution": {
            "type": "object", "additionalProperties": False,
            "required": ["repetitions", "tamper_probe", "measure_systems", "strict_expected_ordering"],
            "properties": {
                "repetitions": {"type": "integer", "minimum": 1, "maximum": 20},
                "tamper_probe": {"type": "boolean"},
                "measure_systems": {"type": "boolean"},
                "strict_expected_ordering": {"type": "boolean"},
            },
        },
        "outputs": {
            "type": "object", "additionalProperties": False,
            "required": ["root", "write_objects", "write_reconstructions"],
            "properties": {
                "root": {"type": "string", "minLength": 1},
                "write_objects": {"type": "boolean"},
                "write_reconstructions": {"type": "boolean"},
            },
        },
    },
}


@dataclass(frozen=True)
class BenchmarkConfig:
    path: Path
    payload: Mapping[str, Any]

    @property
    def benchmark_id(self) -> str:
        return str(self.payload["benchmark_id"])

    @property
    def master_seed(self) -> int:
        return int(self.payload["master_seed"])

    @property
    def master_key(self) -> bytes:
        return bytes.fromhex(str(self.payload["master_key_hex"]))

    @property
    def methods(self) -> list[str]:
        return list(self.payload["methods"])

    @property
    def output_root(self) -> Path:
        configured = Path(str(self.payload["outputs"]["root"]))
        return configured if configured.is_absolute() else (self.path.parents[2] / configured).resolve()


def load_config(path: str | Path) -> BenchmarkConfig:
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    elif source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        raise ValueError("configuration must be YAML or JSON")
    jsonschema.Draft202012Validator(CONFIG_SCHEMA).validate(payload)
    return BenchmarkConfig(source, payload)


def write_schema(path: str | Path) -> None:
    Path(path).write_text(json.dumps(CONFIG_SCHEMA, indent=2, sort_keys=True) + "\n", encoding="utf-8")
