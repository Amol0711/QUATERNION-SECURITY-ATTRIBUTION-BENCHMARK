from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema

from qsa_benchmark.benchmark.registry import EXTENDED_METHOD_FACTORIES
from qsa_benchmark.benchmark.utils import canonical_json_bytes, sha256_file
from qsa_benchmark.protocol.config import load_protocol_config
from qsa_benchmark.protocol.differential import PERTURBATIONS
from qsa_benchmark.protocol.registry_export import (
    REGISTRY_PATHS,
    check_registry_files,
    verify_sha256sums,
)
from qsa_benchmark.protocol.schedule import schedule_registry

ROOT = Path(__file__).resolve().parents[1]


def test_all_committed_registries_match_executable_authorities(
    regenerated_registry_files: dict[str, bytes],
) -> None:
    assert set(regenerated_registry_files) == set(REGISTRY_PATHS)
    for relative, expected in regenerated_registry_files.items():
        assert (ROOT / relative).read_bytes() == expected
    assert check_registry_files(ROOT) == []


def test_registry_cross_links_and_embedded_schemas() -> None:
    config = load_protocol_config()
    construction = json.loads((ROOT / "registries/construction_registry.json").read_text())
    leakage = json.loads((ROOT / "registries/leakage_registry.json").read_text())
    protocol = json.loads((ROOT / "registries/protocol_registry.json").read_text())
    manifest = json.loads((ROOT / "registries/registry_manifest.json").read_text())

    expected_methods = list(EXTENDED_METHOD_FACTORIES)
    assert construction["method_order"] == expected_methods
    assert [row["method_id"] for row in construction["constructions"]] == expected_methods
    assert [row["method_id"] for row in leakage["methods"]] == expected_methods
    assert protocol["method_order"] == expected_methods
    assert construction["construction_count"] == 24
    assert construction["authenticated_count"] == 9
    assert protocol["deterministic_context_schedule"] == schedule_registry()
    assert protocol["derived_counts"] == {
        "authenticated_constructions": 9,
        "corpus_images": 24,
        "corpus_registry_rows": 72,
        "differential_encryptions": 73_536,
        "differential_pairs": 36_768,
        "differential_plan_rows": 138,
        "independent_executions": 2,
        "total_cases": 180,
        "authenticated_cases": 73,
        "authenticated_rejections": 73,
        "unauthenticated_cases": 107,
        "unauthenticated_acceptances": 77,
        "unauthenticated_rejections": 30,
        "unauthenticated_plaintext_changed": 37,
        "unauthenticated_plaintext_unchanged": 40,
        "parser_length_violations": 48,
        "primary_perturbations": 12,
        "registered_constructions": 24,
        "registered_perturbations": 24,
        "secondary_perturbations": 12,
        "standard_corpus_rows": 48,
        "timing_configurations": 576,
        "timing_corpus_rows": 72,
    }
    assert manifest["protocol_sha256"] == config.sha256
    assert manifest["method_order_sha256"] == hashlib.sha256(
        canonical_json_bytes(expected_methods)
    ).hexdigest()
    assert manifest["registry_file_count"] == 7
    assert manifest["listed_registry_count"] == 6
    assert len(manifest["registries"]) == 6
    assert set(manifest["schema_catalog"]) == {
        "active_modification_schedule",
        "construction_registry",
        "corpus_registry",
        "leakage_registry",
        "perturbation_schedule",
        "protocol_registry",
        "registry_manifest",
    }

    payloads = {
        "construction_registry": construction,
        "leakage_registry": leakage,
        "protocol_registry": protocol,
        "registry_manifest": manifest,
    }
    for name, payload in payloads.items():
        schema_entry = manifest["schema_catalog"][name]
        schema = schema_entry["schema"]
        assert schema_entry["schema_sha256"] == hashlib.sha256(
            canonical_json_bytes(schema)
        ).hexdigest()
        jsonschema.Draft202012Validator(schema).validate(payload)


def test_targeted_validation_linkage_is_exact() -> None:
    protocol = json.loads((ROOT / "registries/protocol_registry.json").read_text())
    linkage = protocol["targeted_validation"]
    target_path = ROOT / linkage["source_path"]
    target_schema_path = ROOT / linkage["source_schema_path"]
    target = json.loads(target_path.read_text(encoding="utf-8"))

    assert linkage["protocol_id"] == target["protocol_id"]
    assert linkage["protocol_sha256"] == hashlib.sha256(
        canonical_json_bytes(target)
    ).hexdigest()
    assert linkage["parent_protocol_id"] == protocol["protocol_id"]
    assert linkage["parent_protocol_sha256"] == protocol["protocol_sha256"]
    assert linkage["method_ids"] == [
        "B13_permutation_only",
        "B23_secure_fixed_header",
    ]
    assert linkage["source_file_sha256"] == sha256_file(target_path)
    assert linkage["source_schema_sha256"] == sha256_file(target_schema_path)


def test_registry_manifest_digests_and_authorities() -> None:
    manifest = json.loads((ROOT / "registries/registry_manifest.json").read_text())
    entries = {entry["path"]: entry for entry in manifest["registries"]}
    assert set(entries) == set(REGISTRY_PATHS) - {"registries/registry_manifest.json"}
    for relative, entry in entries.items():
        assert entry["file_sha256"] == sha256_file(ROOT / relative)
        assert len(entry["canonical_sha256"]) == 64
    source_paths = [source["path"] for source in manifest["source_authorities"]]
    assert len(source_paths) == len(set(source_paths)) == 25
    assert {
        "configs/reference/active_modification_inputs.json",
        "configs/reference/active_modification_inputs.schema.json",
        "src/qsa_benchmark/benchmark/constructions.py",
        "src/qsa_benchmark/benchmark/envelope.py",
        "src/qsa_benchmark/benchmark/controls.py",
        "src/qsa_benchmark/benchmark/plugins.py",
        "src/qsa_benchmark/benchmark/transforms.py",
        "src/qsa_benchmark/protocol/experiment.py",
        "src/qsa_benchmark/validation/active_modification.py",
    } <= set(source_paths)
    for source in manifest["source_authorities"]:
        assert source["sha256"] == sha256_file(ROOT / source["path"])
    assert verify_sha256sums(ROOT) == []


def test_perturbation_registry_order_is_exact() -> None:
    import csv

    with (ROOT / "registries/perturbation_schedule.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["perturbation_id"] for row in rows] == list(PERTURBATIONS)
    assert len(rows) == 24
    assert [int(row["ordinal"]) for row in rows] == list(range(1, 25))
