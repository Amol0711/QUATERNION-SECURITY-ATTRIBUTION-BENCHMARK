from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema
import numpy as np
from PIL import Image

from qsa_benchmark.benchmark.datasets import build_corpora
from qsa_benchmark.benchmark.registry import EXTENDED_METHOD_FACTORIES, method_registry
from qsa_benchmark.benchmark.utils import canonical_json_bytes, sha256_file
from qsa_benchmark.protocol.config import ProtocolConfig, load_protocol_config
from qsa_benchmark.protocol.differential import PERTURBATIONS, apply_perturbation, planned_pair_counts
from qsa_benchmark.protocol.registry import POLICIES, operation_regime
from qsa_benchmark.protocol.schedule import schedule_registry
from qsa_benchmark.protocol.timing import FORWARD_STAGES, INVERSE_STAGES, applicable_stages
from qsa_benchmark.validation.active_modification import (
    active_modification_counts,
    build_active_modification_rows,
    load_active_modification_config,
)

REGISTRY_SCHEMA_VERSION = 1
REGISTRY_SET_VERSION = "1.0.0"
CANONICAL_FORMAT_PROFILE_ID = "QSA-CANONICAL-FORMATS-V1"

REGISTRY_PATHS = (
    "registries/active_modification_schedule.csv",
    "registries/construction_registry.json",
    "registries/corpus_registry.csv",
    "registries/leakage_registry.json",
    "registries/perturbation_schedule.csv",
    "registries/protocol_registry.json",
    "registries/registry_manifest.json",
)

CORPUS_FIELDS = (
    "registry_row_id",
    "image_id",
    "image_size",
    "width",
    "height",
    "channels",
    "dtype",
    "color_space",
    "corpus",
    "split",
    "semantic_label",
    "source",
    "source_provider",
    "preprocessing",
    "license_note",
    "generated_relative_path",
    "standard_experiment_size",
    "timing_size",
    "primary_panel",
    "ensemble_panel",
    "secondary_panel",
    "timing_panel",
    "pixel_sha256",
    "file_sha256",
    "file_size_bytes",
)

PERTURBATION_FIELDS = (
    "ordinal",
    "perturbation_id",
    "panel",
    "inferential_role",
    "family",
    "location",
    "channel_indices",
    "channel_names",
    "magnitude",
    "expected_pixel_count_rule",
    "expected_pixels_by_size",
    "expected_coordinates_by_size",
    "description",
    "applicable_tiers",
    "tier_schedule",
    "update_rule",
    "gradient_tie_break",
    "validation_probe_id",
    "validation_pixel_counts",
    "validation_coordinate_counts",
)

ACTIVE_MODIFICATION_FIELDS = (
    "case_id",
    "case_ordinal",
    "method_ordinal",
    "method_mutation_ordinal",
    "method_id",
    "authenticated",
    "mutation_id",
    "target_component",
    "mutation_class",
    "parser_level",
    "expected_outcome",
    "expected_accepted",
    "expected_plaintext_changed",
    "failure_stage",
    "expected_exception_class",
    "expected_parser_code",
    "base_input_sha256",
    "donor_input_sha256",
    "context_sha256",
    "original_object_sha256",
    "mutated_object_sha256",
    "original_object_bytes",
    "mutated_object_bytes",
    "byte_difference_count",
    "nonce_bytes",
    "public_payload_bytes",
    "protected_payload_bytes",
    "tag_bytes",
    "byte_distinct",
)

_SOURCE_AUTHORITY_PATHS = (
    "pyproject.toml",
    "configs/protocol/experiment.json",
    "configs/protocol/experiment.schema.json",
    "configs/protocol/targeted_validation.json",
    "configs/protocol/targeted_validation.schema.json",
    "configs/reference/active_modification_inputs.json",
    "configs/reference/active_modification_inputs.schema.json",
    "src/qsa_benchmark/benchmark/constructions.py",
    "src/qsa_benchmark/benchmark/envelope.py",
    "src/qsa_benchmark/benchmark/controls.py",
    "src/qsa_benchmark/benchmark/datasets.py",
    "src/qsa_benchmark/benchmark/plugins.py",
    "src/qsa_benchmark/benchmark/registry.py",
    "src/qsa_benchmark/benchmark/transforms.py",
    "src/qsa_benchmark/benchmark/utils.py",
    "src/qsa_benchmark/protocol/config.py",
    "src/qsa_benchmark/protocol/experiment.py",
    "src/qsa_benchmark/protocol/differential.py",
    "src/qsa_benchmark/protocol/models.py",
    "src/qsa_benchmark/protocol/registry.py",
    "src/qsa_benchmark/protocol/schedule.py",
    "src/qsa_benchmark/protocol/registry_export.py",
    "src/qsa_benchmark/protocol/timing.py",
    "src/qsa_benchmark/validation/active_modification.py",
    "scripts/export_registries.py",
)


def _human_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _csv_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return canonical_json_bytes(value).decode("ascii")
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, np.generic):
        return _csv_scalar(value.item())
    return str(value)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fields),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        missing = [field for field in fields if field not in row]
        if missing:
            raise ValueError(f"missing CSV fields: {missing}")
        writer.writerow({field: _csv_scalar(row[field]) for field in fields})
    payload = buffer.getvalue().encode("utf-8")
    if b"\r" in payload:
        raise RuntimeError("registry CSV contains a carriage return")
    return payload


def _read_csv_bytes(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8")
    return list(csv.DictReader(io.StringIO(text, newline="")))


def _semantic_csv_sha256(payload: bytes) -> str:
    rows = _read_csv_bytes(payload)
    normalized = [{key: row[key] for key in sorted(row)} for row in rows]
    normalized.sort(key=canonical_json_bytes)
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _registered_sizes(config: ProtocolConfig) -> list[int]:
    corpus_sizes = {int(value) for value in config.payload["corpus"]["sizes"]}
    timing_sizes = {int(value) for value in config.payload["timing"]["sizes"]}
    sizes = sorted(corpus_sizes | timing_sizes)
    if sizes != [96, 256, 512]:
        raise ValueError(f"unexpected registered corpus sizes: {sizes}")
    return sizes


def _build_corpus_rows(
    config: ProtocolConfig,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    payload = config.payload
    corpus_config = payload["corpus"]
    image_order = list(corpus_config["image_ids"])
    standard_sizes = {int(value) for value in corpus_config["sizes"]}
    timing_sizes = {int(value) for value in payload["timing"]["sizes"]}
    sizes = _registered_sizes(config)
    panels = {
        name: set(corpus_config[name])
        for name in ("primary_panel", "ensemble_panel", "secondary_panel", "timing_panel")
    }
    rows: list[dict[str, Any]] = []
    probes: dict[int, np.ndarray] = {}

    with tempfile.TemporaryDirectory(prefix="qsa-corpus-registry-") as temporary:
        temporary_root = Path(temporary)
        for size in sizes:
            generated_root = temporary_root / f"{size}x{size}"
            records = build_corpora(generated_root, size=(size, size))
            by_id = {record.image_id: record for record in records}
            if len(by_id) != len(records):
                raise RuntimeError(f"duplicate corpus identifier at size {size}")
            if list(by_id) != image_order:
                raise RuntimeError(
                    f"generated corpus order at size {size} does not match the protocol order"
                )
            for image_id in image_order:
                record = by_id[image_id]
                generated_path = Path(record.path)
                file_digest = sha256_file(generated_path)
                if file_digest != record.sha256:
                    raise RuntimeError(f"generated-file digest disagreement for {image_id} {size}")
                image = np.asarray(Image.open(generated_path).convert("RGB"), dtype=np.uint8)
                if image.shape != (size, size, 3):
                    raise RuntimeError(
                        f"generated corpus shape disagreement for {image_id} {size}: {image.shape}"
                    )
                pixel_digest = hashlib.sha256(image.tobytes(order="C")).hexdigest()
                if image_id == "syn_lowrank":
                    probes[size] = image.copy()
                preprocessing = (
                    corpus_config["synthetic_generation"]
                    if record.corpus == "synthetic"
                    else corpus_config["natural_generation"]
                )
                source_provider = "qsa_benchmark" if record.corpus == "synthetic" else "scikit-image"
                generated_relative_path = (
                    f"data/generated/{size}x{size}/{record.corpus}/"
                    f"{image_id}_{size}x{size}.png"
                )
                rows.append(
                    {
                        "registry_row_id": f"{image_id}@{size}x{size}",
                        "image_id": image_id,
                        "image_size": size,
                        "width": size,
                        "height": size,
                        "channels": 3,
                        "dtype": corpus_config["dtype"],
                        "color_space": corpus_config["color_space"],
                        "corpus": record.corpus,
                        "split": record.split,
                        "semantic_label": record.semantic_label,
                        "source": record.source,
                        "source_provider": source_provider,
                        "preprocessing": preprocessing,
                        "license_note": record.license,
                        "generated_relative_path": generated_relative_path,
                        "standard_experiment_size": size in standard_sizes,
                        "timing_size": size in timing_sizes,
                        "primary_panel": image_id in panels["primary_panel"],
                        "ensemble_panel": image_id in panels["ensemble_panel"],
                        "secondary_panel": image_id in panels["secondary_panel"],
                        "timing_panel": image_id in panels["timing_panel"],
                        "pixel_sha256": pixel_digest,
                        "file_sha256": file_digest,
                        "file_size_bytes": generated_path.stat().st_size,
                    }
                )

    expected_rows = len(image_order) * len(sizes)
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} corpus rows, generated {len(rows)}")
    image_position = {image_id: index for index, image_id in enumerate(image_order)}
    rows.sort(key=lambda row: (image_position[str(row["image_id"])], int(row["image_size"])))
    if set(probes) != set(sizes):
        raise RuntimeError("the canonical perturbation probe is incomplete")
    return rows, probes


def _expected_pixel_count(perturbation_id: str, size: int) -> tuple[int, str]:
    spec = PERTURBATIONS[perturbation_id]
    if spec.expected_pixel_count is not None:
        return min(int(spec.expected_pixel_count), size * size), f"min({spec.expected_pixel_count},H*W)"
    if perturbation_id == "CHECKER_LSB_ALL":
        return (size * size + 1) // 2, "ceil(H*W/2)"
    if perturbation_id == "ROW_CENTER_ALL_UNIT":
        return size, "W"
    raise RuntimeError(f"missing expected-pixel rule for {perturbation_id}")


def _tier_perturbations(config: ProtocolConfig, tier: Mapping[str, Any]) -> list[str]:
    selected = tier["perturbations"]
    if isinstance(selected, str):
        return list(config.payload["perturbations"][selected])
    return list(selected)


def _build_perturbation_rows(
    config: ProtocolConfig,
    probes: Mapping[int, np.ndarray],
) -> list[dict[str, Any]]:
    payload = config.payload
    primary = list(payload["perturbations"]["primary"])
    secondary = list(payload["perturbations"]["secondary"])
    ordered = primary + secondary
    if list(PERTURBATIONS) != ordered:
        raise RuntimeError("executable perturbation order does not match the protocol configuration")

    channel_names = {0: "R", 1: "G", 2: "B"}
    rows: list[dict[str, Any]] = []
    sizes = _registered_sizes(config)
    for ordinal, perturbation_id in enumerate(ordered, start=1):
        spec = PERTURBATIONS[perturbation_id]
        panel = "primary" if perturbation_id in primary else "secondary"
        if spec.inferential_role != panel:
            raise RuntimeError(f"inferential-role disagreement for {perturbation_id}")

        expected_pixels: dict[str, int] = {}
        expected_coordinates: dict[str, int] = {}
        validated_pixels: dict[str, int] = {}
        validated_coordinates: dict[str, int] = {}
        pixel_count_rule: str | None = None
        for size in sizes:
            expected, rule = _expected_pixel_count(perturbation_id, size)
            pixel_count_rule = rule if pixel_count_rule is None else pixel_count_rule
            if rule != pixel_count_rule:
                raise RuntimeError(f"size-dependent rule description for {perturbation_id}")
            source = probes[size]
            changed = apply_perturbation(source, perturbation_id)
            delta = changed != source
            actual_pixels = int(np.count_nonzero(np.any(delta, axis=2)))
            actual_coordinates = int(np.count_nonzero(delta))
            expected_coordinate_count = expected * len(spec.channels)
            if actual_pixels != expected or actual_coordinates != expected_coordinate_count:
                raise RuntimeError(
                    f"perturbation validation mismatch for {perturbation_id} at {size}: "
                    f"pixels={actual_pixels}/{expected}, coordinates={actual_coordinates}/{expected_coordinate_count}"
                )
            key = str(size)
            expected_pixels[key] = expected
            expected_coordinates[key] = expected_coordinate_count
            validated_pixels[key] = actual_pixels
            validated_coordinates[key] = actual_coordinates

        tier_schedule: dict[str, Any] = {}
        for tier_id, tier_raw in payload["execution_tiers"].items():
            tier = dict(tier_raw)
            if perturbation_id not in _tier_perturbations(config, tier):
                continue
            methods = payload["methods"] if tier["methods"] == "all" else tier["methods"]
            image_panel = str(tier["images"])
            tier_schedule[tier_id] = {
                "methods": list(methods),
                "method_count": len(methods),
                "image_panel": image_panel,
                "image_count": len(payload["corpus"][image_panel]),
                "sizes": [int(value) for value in tier["sizes"]],
                "key_repetitions": int(tier["key_repetitions"]),
                "state_repetitions_by_size": {
                    str(key): int(value)
                    for key, value in tier["state_repetitions_by_size"].items()
                },
                "protocols": list(tier["protocols"]),
            }
        if not tier_schedule:
            raise RuntimeError(f"perturbation {perturbation_id} is not scheduled in any execution tier")

        rows.append(
            {
                "ordinal": ordinal,
                "perturbation_id": perturbation_id,
                "panel": panel,
                "inferential_role": spec.inferential_role,
                "family": spec.family,
                "location": spec.location,
                "channel_indices": list(spec.channels),
                "channel_names": [channel_names[index] for index in spec.channels],
                "magnitude": int(spec.magnitude),
                "expected_pixel_count_rule": pixel_count_rule,
                "expected_pixels_by_size": expected_pixels,
                "expected_coordinates_by_size": expected_coordinates,
                "description": spec.description,
                "applicable_tiers": list(tier_schedule),
                "tier_schedule": tier_schedule,
                "update_rule": payload["perturbations"]["update_rule"],
                "gradient_tie_break": payload["perturbations"]["gradient_tie_break"],
                "validation_probe_id": "syn_lowrank",
                "validation_pixel_counts": validated_pixels,
                "validation_coordinate_counts": validated_coordinates,
            }
        )
    return rows


def _build_construction_registry(config: ProtocolConfig) -> dict[str, Any]:
    runtime_rows = {row["method_id"]: row for row in method_registry("extended")}
    methods: list[dict[str, Any]] = []
    for ordinal, method_id in enumerate(config.methods, start=1):
        runtime = runtime_rows[method_id]
        policy = POLICIES[method_id]
        methods.append(
            {
                "ordinal": ordinal,
                "method_id": method_id,
                "display_name": runtime["display_name"],
                "family": runtime["family"],
                "benchmark_role": runtime["benchmark_role"],
                "profile": runtime["profile"],
                "authenticated": bool(runtime["authenticated"]),
                "secure_control": bool(runtime["secure_control"]),
                "exact": bool(runtime["exact"]),
                "publicly_invertible": bool(policy.publicly_invertible),
                "component_ids": list(runtime["component_ids"]),
                "metric_body_source": policy.metric_body_source,
                "body_metric_domain": policy.body_metric_domain,
                "nonce_length": int(policy.nonce_length),
                "common_map_class": policy.common_map_class,
                "timing_path": policy.timing_path,
                "p1_semantics": policy.p1_semantics,
                "p1_correct_use": bool(policy.p1_correct_use),
                "p1_operation_regime": operation_regime(method_id, "P1_common_context"),
                "p2_semantics": policy.p2_semantics,
                "p2_applicable": bool(policy.p2_applicable),
                "p2_operation_regime": (
                    operation_regime(method_id, "P2_fresh_randomness")
                    if policy.p2_applicable
                    else None
                ),
                "notes": policy.notes,
            }
        )
    authenticated_count = sum(bool(item["authenticated"]) for item in methods)
    if authenticated_count != 9:
        raise RuntimeError(f"expected 9 authenticated constructions, found {authenticated_count}")
    return {
        "registry_id": "QSA-CONSTRUCTION-REGISTRY-V1",
        "schema_id": "QSA-CONSTRUCTION-REGISTRY-SCHEMA-V1",
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "protocol_id": config.protocol_id,
        "protocol_sha256": config.sha256,
        "construction_count": len(methods),
        "authenticated_count": authenticated_count,
        "method_order": config.methods,
        "constructions": methods,
    }


def _build_leakage_registry(config: ProtocolConfig) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ordinal, method_id in enumerate(config.methods, start=1):
        policy = POLICIES[method_id]
        rows.append(
            {
                "ordinal": ordinal,
                "method_id": method_id,
                "metric_body_source": policy.metric_body_source,
                "body_metric_domain": policy.body_metric_domain,
                "deterministic_plaintext_leakage": list(policy.deterministic_plaintext_leakage),
                "public_randomness": list(policy.public_randomness),
                "public_recovery_material": list(policy.public_recovery_material),
                "authenticated_coverage": list(policy.authenticated_coverage),
                "permitted_functionality": list(policy.permitted_functionality),
                "leakage_equivalence_rule": policy.leakage_equivalence_rule,
                "prechallenge_entropy_rule": policy.prechallenge_entropy_rule,
                "descriptor_entropy_rule": policy.descriptor_entropy_rule,
                "post_object_recovery": policy.post_object_recovery,
            }
        )
    object_model = config.payload["object_model"]
    return {
        "registry_id": "QSA-LEAKAGE-REGISTRY-V1",
        "schema_id": "QSA-LEAKAGE-REGISTRY-SCHEMA-V1",
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "protocol_id": config.protocol_id,
        "protocol_sha256": config.sha256,
        "method_count": len(rows),
        "object_model": {
            "metadata_profile": object_model["metadata_profile"],
            "evaluation_identifiers_in_object": bool(object_model["evaluation_identifiers_in_object"]),
            "external_log_fields": list(object_model["external_log_fields"]),
            "complete_public_view": list(object_model["complete_public_view"]),
        },
        "global_leakage_policy": config.payload["leakage"],
        "methods": rows,
    }


def _summarize_plan(config: ProtocolConfig) -> dict[str, Any]:
    rows = planned_pair_counts(config.payload)
    by_tier: dict[str, dict[str, int]] = {}
    for row in rows:
        tier_id = str(row["tier_id"])
        summary = by_tier.setdefault(
            tier_id,
            {"plan_rows": 0, "planned_pairs": 0, "planned_encryptions": 0},
        )
        summary["plan_rows"] += 1
        summary["planned_pairs"] += int(row["planned_pairs"])
        summary["planned_encryptions"] += int(row["planned_encryptions"])
    return {
        "plan_row_count": len(rows),
        "planned_pairs": sum(int(row["planned_pairs"]) for row in rows),
        "planned_encryptions": sum(int(row["planned_encryptions"]) for row in rows),
        "by_tier": by_tier,
    }


def _targeted_validation_linkage(config: ProtocolConfig) -> dict[str, Any]:
    """Return the validated linkage to the targeted-validation protocol."""

    repo_root = config.path.parents[2]
    source = repo_root / "configs/protocol/targeted_validation.json"
    schema_source = repo_root / "configs/protocol/targeted_validation.schema.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    schema = json.loads(schema_source.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(payload)
    protocol_sha256 = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if payload["parent_protocol_id"] != config.protocol_id:
        raise RuntimeError("targeted validation parent protocol identifier mismatch")
    if payload["parent_protocol_sha256"] != config.sha256:
        raise RuntimeError("targeted validation parent protocol digest mismatch")
    method_ids = [
        payload["b13_query_tightness"]["method_id"],
        payload["b23_npcr_power"]["method_id"],
    ]
    if method_ids != ["B13_permutation_only", "B23_secure_fixed_header"]:
        raise RuntimeError("targeted validation method linkage is incorrect")
    if any(method_id not in config.methods for method_id in method_ids):
        raise RuntimeError("targeted validation references an unregistered construction")
    return {
        "protocol_id": payload["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "parent_protocol_id": payload["parent_protocol_id"],
        "parent_protocol_sha256": payload["parent_protocol_sha256"],
        "scope": payload["scope"],
        "method_ids": method_ids,
        "replication_policy": payload["replication_policy"],
        "source_path": "configs/protocol/targeted_validation.json",
        "source_file_sha256": sha256_file(source),
        "source_schema_path": "configs/protocol/targeted_validation.schema.json",
        "source_schema_sha256": sha256_file(schema_source),
    }


def _timing_stage_registry(config: ProtocolConfig) -> dict[str, Any]:
    return {
        "forward_stages": list(FORWARD_STAGES),
        "inverse_stages": list(INVERSE_STAGES),
        "method_paths": [
            {
                "method_id": method_id,
                "encrypt": list(applicable_stages(method_id, "encrypt")),
                "decrypt": list(applicable_stages(method_id, "decrypt")),
            }
            for method_id in config.methods
        ],
    }


def _build_protocol_registry(
    config: ProtocolConfig,
    corpus_rows: Sequence[Mapping[str, Any]],
    perturbation_rows: Sequence[Mapping[str, Any]],
    active_rows: Sequence[Mapping[str, Any]],
    active_config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = config.payload
    differential_plan = _summarize_plan(config)
    timing_configuration_count = (
        len(config.methods)
        * len(payload["timing"]["sizes"])
        * len(payload["corpus"]["timing_panel"])
        * 2
    )
    if timing_configuration_count != 576:
        raise RuntimeError(
            f"expected 576 timing configurations, found {timing_configuration_count}"
        )
    derived_counts = {
        "registered_constructions": len(config.methods),
        "authenticated_constructions": sum(
            bool(POLICIES[method_id].authenticated_coverage) for method_id in config.methods
        ),
        "corpus_images": len(payload["corpus"]["image_ids"]),
        "corpus_registry_rows": len(corpus_rows),
        "standard_corpus_rows": sum(
            bool(row["standard_experiment_size"]) for row in corpus_rows
        ),
        "timing_corpus_rows": sum(bool(row["timing_size"]) for row in corpus_rows),
        "registered_perturbations": len(perturbation_rows),
        "primary_perturbations": len(payload["perturbations"]["primary"]),
        "secondary_perturbations": len(payload["perturbations"]["secondary"]),
        "differential_plan_rows": differential_plan["plan_row_count"],
        "differential_pairs": differential_plan["planned_pairs"],
        "differential_encryptions": differential_plan["planned_encryptions"],
        "timing_configurations": timing_configuration_count,
        "independent_executions": int(payload["replication_policy"]["independent_runs"]),
        **active_modification_counts(list(active_rows)),
    }
    if derived_counts["registered_constructions"] != 24:
        raise RuntimeError("the protocol registry must contain 24 constructions")
    if derived_counts["authenticated_constructions"] != 9:
        raise RuntimeError("the protocol registry must contain 9 authenticated constructions")
    if derived_counts["corpus_registry_rows"] != 72:
        raise RuntimeError("the corpus registry must contain 72 rows")
    if derived_counts["standard_corpus_rows"] != 48:
        raise RuntimeError("the standard corpus must contain 48 rows")
    if derived_counts["registered_perturbations"] != 24:
        raise RuntimeError("the perturbation registry must contain 24 perturbations")
    if derived_counts["total_cases"] != 180 or derived_counts["authenticated_cases"] != 73:
        raise RuntimeError("the active-modification schedule has incorrect primary counts")
    return {
        "registry_id": "QSA-PROTOCOL-REGISTRY-V1",
        "schema_id": "QSA-PROTOCOL-REGISTRY-SCHEMA-V1",
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "protocol_id": config.protocol_id,
        "protocol_sha256": config.sha256,
        "benchmark_profile": payload["benchmark_profile"],
        "deterministic_context": {
            "master_seed": int(payload["master_seed"]),
            "master_key_hex": payload["master_key_hex"],
            "notice": "The configured key and seed are public deterministic benchmark values and are not operational secrets.",
        },
        "deterministic_context_schedule": schedule_registry(),
        "method_order": config.methods,
        "extended_controls": list(payload["extended_controls"]),
        "object_model": payload["object_model"],
        "corpus": {
            **payload["corpus"],
            "timing_sizes": [int(value) for value in payload["timing"]["sizes"]],
            "registry_sizes": _registered_sizes(config),
        },
        "differential_protocols": payload["protocols"],
        "perturbation_policy": {
            **payload["perturbations"],
            "registered_order": [str(row["perturbation_id"]) for row in perturbation_rows],
        },
        "active_modification_policy": {
            "schedule_id": active_config["schedule_id"],
            "registry_path": "registries/active_modification_schedule.csv",
            "method_profile": active_config["method_profile"],
            "base_input_sha256": active_config["input_pair"]["base_sha256"],
            "donor_input_sha256": active_config["input_pair"]["donor_sha256"],
            "mutation_order": list(dict.fromkeys(str(row["mutation_id"]) for row in active_rows)),
            "parser_level_mutations": ["trailing_byte_append", "one_byte_truncation"],
            "outcome_classes": [
                "parser_reject",
                "integrity_reject",
                "accepted_plaintext_changed",
                "accepted_plaintext_unchanged",
            ],
            "public_test_values": bool(active_config["public_test_values"]),
            "operational_use_prohibited": bool(active_config["operational_use_prohibited"]),
        },
        "execution_tiers": payload["execution_tiers"],
        "statistics": payload["statistics"],
        "endpoint_policy": payload["endpoint_policy"],
        "leakage_policy": payload["leakage"],
        "timing": payload["timing"],
        "timing_stage_registry": _timing_stage_registry(config),
        "replication_policy": payload["replication_policy"],
        "outputs": payload["outputs"],
        "targeted_validation": _targeted_validation_linkage(config),
        "derived_counts": derived_counts,
        "differential_plan_summary": differential_plan,
    }


def _json_registry_schemas() -> dict[str, dict[str, Any]]:
    method_pattern = r"^B(?:0[1-9]|1[0-9]|2[0-4])_[a-z0-9_]+$"
    construction_item = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "ordinal", "method_id", "display_name", "family", "benchmark_role", "profile",
            "authenticated", "secure_control", "exact", "publicly_invertible", "component_ids",
            "metric_body_source", "body_metric_domain", "nonce_length", "common_map_class",
            "timing_path", "p1_semantics", "p1_correct_use", "p1_operation_regime",
            "p2_semantics", "p2_applicable", "p2_operation_regime", "notes",
        ],
        "properties": {
            "ordinal": {"type": "integer", "minimum": 1},
            "method_id": {"type": "string", "pattern": method_pattern},
            "display_name": {"type": "string", "minLength": 1},
            "family": {"type": "string", "minLength": 1},
            "benchmark_role": {"type": "string", "minLength": 1},
            "profile": {"const": "extended"},
            "authenticated": {"type": "boolean"},
            "secure_control": {"type": "boolean"},
            "exact": {"type": "boolean"},
            "publicly_invertible": {"type": "boolean"},
            "component_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "metric_body_source": {"enum": ["protected_payload", "public_payload"]},
            "body_metric_domain": {"enum": ["rgb_u8", "opaque_u8", "int32_serialized", "fixed_prefix_rgb_suffix"]},
            "nonce_length": {"enum": [12, 16]},
            "common_map_class": {"type": "string", "minLength": 1},
            "timing_path": {"type": "string", "minLength": 1},
            "p1_semantics": {"type": "string", "minLength": 1},
            "p1_correct_use": {"type": "boolean"},
            "p1_operation_regime": {"type": "string", "minLength": 1},
            "p2_semantics": {"type": "string", "minLength": 1},
            "p2_applicable": {"type": "boolean"},
            "p2_operation_regime": {"type": ["string", "null"]},
            "notes": {"type": "string"},
        },
    }
    construction_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:qsa:schema:construction-registry:v1",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "registry_id", "schema_id", "schema_version", "protocol_id", "protocol_sha256",
            "construction_count", "authenticated_count", "method_order", "constructions",
        ],
        "properties": {
            "registry_id": {"const": "QSA-CONSTRUCTION-REGISTRY-V1"},
            "schema_id": {"const": "QSA-CONSTRUCTION-REGISTRY-SCHEMA-V1"},
            "schema_version": {"const": 1},
            "protocol_id": {"const": "QSA-PROTOCOL-V1"},
            "protocol_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "construction_count": {"const": 24},
            "authenticated_count": {"const": 9},
            "method_order": {"type": "array", "minItems": 24, "maxItems": 24, "uniqueItems": True, "items": {"type": "string", "pattern": method_pattern}},
            "constructions": {"type": "array", "minItems": 24, "maxItems": 24, "items": construction_item},
        },
    }
    leakage_item = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "ordinal", "method_id", "metric_body_source", "body_metric_domain",
            "deterministic_plaintext_leakage", "public_randomness", "public_recovery_material",
            "authenticated_coverage", "permitted_functionality", "leakage_equivalence_rule",
            "prechallenge_entropy_rule", "descriptor_entropy_rule", "post_object_recovery",
        ],
        "properties": {
            "ordinal": {"type": "integer", "minimum": 1},
            "method_id": {"type": "string", "pattern": method_pattern},
            "metric_body_source": {"enum": ["protected_payload", "public_payload"]},
            "body_metric_domain": {"enum": ["rgb_u8", "opaque_u8", "int32_serialized", "fixed_prefix_rgb_suffix"]},
            "deterministic_plaintext_leakage": {"type": "array", "items": {"type": "string"}},
            "public_randomness": {"type": "array", "items": {"type": "string"}},
            "public_recovery_material": {"type": "array", "items": {"type": "string"}},
            "authenticated_coverage": {"type": "array", "items": {"type": "string"}},
            "permitted_functionality": {"type": "array", "items": {"type": "string"}},
            "leakage_equivalence_rule": {"type": "string"},
            "prechallenge_entropy_rule": {"type": "string"},
            "descriptor_entropy_rule": {"type": "string"},
            "post_object_recovery": {"type": "string"},
        },
    }
    leakage_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:qsa:schema:leakage-registry:v1",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "registry_id", "schema_id", "schema_version", "protocol_id", "protocol_sha256",
            "method_count", "object_model", "global_leakage_policy", "methods",
        ],
        "properties": {
            "registry_id": {"const": "QSA-LEAKAGE-REGISTRY-V1"},
            "schema_id": {"const": "QSA-LEAKAGE-REGISTRY-SCHEMA-V1"},
            "schema_version": {"const": 1},
            "protocol_id": {"const": "QSA-PROTOCOL-V1"},
            "protocol_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "method_count": {"const": 24},
            "object_model": {"type": "object"},
            "global_leakage_policy": {"type": "object"},
            "methods": {"type": "array", "minItems": 24, "maxItems": 24, "items": leakage_item},
        },
    }
    protocol_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:qsa:schema:protocol-registry:v1",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "registry_id", "schema_id", "schema_version", "protocol_id", "protocol_sha256",
            "benchmark_profile", "deterministic_context", "deterministic_context_schedule",
            "method_order", "extended_controls", "object_model", "corpus",
            "differential_protocols", "perturbation_policy", "active_modification_policy", "execution_tiers",
            "statistics", "endpoint_policy", "leakage_policy", "timing",
            "timing_stage_registry", "replication_policy", "outputs",
            "targeted_validation", "derived_counts", "differential_plan_summary",
        ],
        "properties": {
            "registry_id": {"const": "QSA-PROTOCOL-REGISTRY-V1"},
            "schema_id": {"const": "QSA-PROTOCOL-REGISTRY-SCHEMA-V1"},
            "schema_version": {"const": 1},
            "protocol_id": {"const": "QSA-PROTOCOL-V1"},
            "protocol_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "benchmark_profile": {"const": "extended"},
            "deterministic_context": {"type": "object"},
            "deterministic_context_schedule": {"type": "object"},
            "method_order": {"type": "array", "minItems": 24, "maxItems": 24, "uniqueItems": True, "items": {"type": "string", "pattern": method_pattern}},
            "extended_controls": {"type": "array", "minItems": 4, "maxItems": 4, "uniqueItems": True, "items": {"type": "string"}},
            "object_model": {"type": "object"},
            "corpus": {"type": "object"},
            "differential_protocols": {"type": "object"},
            "perturbation_policy": {"type": "object"},
            "active_modification_policy": {"type": "object"},
            "execution_tiers": {"type": "object"},
            "statistics": {"type": "object"},
            "endpoint_policy": {"type": "object"},
            "leakage_policy": {"type": "object"},
            "timing": {"type": "object"},
            "timing_stage_registry": {"type": "object"},
            "replication_policy": {"type": "object"},
            "outputs": {"type": "object"},
            "targeted_validation": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "protocol_id", "protocol_sha256", "parent_protocol_id",
                    "parent_protocol_sha256", "scope", "method_ids",
                    "replication_policy", "source_path", "source_file_sha256",
                    "source_schema_path", "source_schema_sha256",
                ],
                "properties": {
                    "protocol_id": {"const": "QSA-TARGETED-VALIDATION-V1"},
                    "protocol_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "parent_protocol_id": {"const": "QSA-PROTOCOL-V1"},
                    "parent_protocol_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "scope": {"type": "string", "minLength": 1},
                    "method_ids": {
                        "const": ["B13_permutation_only", "B23_secure_fixed_header"]
                    },
                    "replication_policy": {"type": "object"},
                    "source_path": {"const": "configs/protocol/targeted_validation.json"},
                    "source_file_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "source_schema_path": {"const": "configs/protocol/targeted_validation.schema.json"},
                    "source_schema_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
            },
            "derived_counts": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "registered_constructions", "authenticated_constructions",
                    "corpus_images", "corpus_registry_rows", "standard_corpus_rows",
                    "timing_corpus_rows", "registered_perturbations",
                    "primary_perturbations", "secondary_perturbations",
                    "differential_plan_rows", "differential_pairs",
                    "differential_encryptions", "timing_configurations",
                    "independent_executions", "total_cases", "authenticated_cases",
                    "authenticated_rejections", "unauthenticated_cases",
                    "unauthenticated_acceptances", "unauthenticated_rejections",
                    "unauthenticated_plaintext_changed",
                    "unauthenticated_plaintext_unchanged", "parser_length_violations",
                ],
                "properties": {
                    "registered_constructions": {"const": 24},
                    "authenticated_constructions": {"const": 9},
                    "corpus_images": {"const": 24},
                    "corpus_registry_rows": {"const": 72},
                    "standard_corpus_rows": {"const": 48},
                    "timing_corpus_rows": {"const": 72},
                    "registered_perturbations": {"const": 24},
                    "primary_perturbations": {"const": 12},
                    "secondary_perturbations": {"const": 12},
                    "differential_plan_rows": {"const": 138},
                    "differential_pairs": {"const": 36768},
                    "differential_encryptions": {"const": 73536},
                    "timing_configurations": {"const": 576},
                    "independent_executions": {"const": 2},
                    "total_cases": {"const": 180},
                    "authenticated_cases": {"const": 73},
                    "authenticated_rejections": {"const": 73},
                    "unauthenticated_cases": {"const": 107},
                    "unauthenticated_acceptances": {"const": 77},
                    "unauthenticated_rejections": {"const": 30},
                    "unauthenticated_plaintext_changed": {"const": 37},
                    "unauthenticated_plaintext_unchanged": {"const": 40},
                    "parser_length_violations": {"const": 48},
                },
            },
            "differential_plan_summary": {
                "type": "object",
                "additionalProperties": False,
                "required": ["plan_row_count", "planned_pairs", "planned_encryptions", "by_tier"],
                "properties": {
                    "plan_row_count": {"const": 138},
                    "planned_pairs": {"const": 36768},
                    "planned_encryptions": {"const": 73536},
                    "by_tier": {"type": "object"},
                },
            },
        },
    }
    return {
        "construction_registry": construction_schema,
        "leakage_registry": leakage_schema,
        "protocol_registry": protocol_schema,
    }


def _csv_registry_schemas() -> dict[str, dict[str, Any]]:
    return {
        "active_modification_schedule": {
            "schema_id": "QSA-ACTIVE-MODIFICATION-SCHEDULE-SCHEMA-V1",
            "schema_version": 1,
            "format": "csv",
            "encoding": "UTF-8 without BOM",
            "line_ending": "LF",
            "primary_key": ["case_id"],
            "row_order": "registered method order, then registered mutation order",
            "fields": [
                {"name": name, "type": (
                    "integer" if name in {
                        "case_ordinal", "method_ordinal", "method_mutation_ordinal",
                        "original_object_bytes", "mutated_object_bytes",
                        "byte_difference_count", "nonce_bytes", "public_payload_bytes",
                        "protected_payload_bytes", "tag_bytes",
                    }
                    else "boolean" if name in {
                        "authenticated", "parser_level", "expected_accepted", "byte_distinct",
                    }
                    else "sha256" if name.endswith("sha256")
                    else "string"
                )}
                for name in ACTIVE_MODIFICATION_FIELDS
            ],
        },
        "corpus_registry": {
            "schema_id": "QSA-CORPUS-REGISTRY-SCHEMA-V1",
            "schema_version": 1,
            "format": "csv",
            "encoding": "UTF-8 without BOM",
            "line_ending": "LF",
            "primary_key": ["registry_row_id"],
            "row_order": "protocol image order, then ascending image_size",
            "fields": [
                {"name": name, "type": (
                    "integer" if name in {"image_size", "width", "height", "channels", "file_size_bytes"}
                    else "boolean" if name.endswith("_panel") or name in {"standard_experiment_size", "timing_size"}
                    else "sha256" if name.endswith("sha256")
                    else "string"
                )}
                for name in CORPUS_FIELDS
            ],
        },
        "perturbation_schedule": {
            "schema_id": "QSA-PERTURBATION-SCHEDULE-SCHEMA-V1",
            "schema_version": 1,
            "format": "csv",
            "encoding": "UTF-8 without BOM",
            "line_ending": "LF",
            "primary_key": ["perturbation_id"],
            "row_order": "protocol primary panel order followed by secondary panel order",
            "fields": [
                {"name": name, "type": (
                    "integer" if name in {"ordinal", "magnitude"}
                    else "canonical_json" if name in {
                        "channel_indices", "channel_names", "expected_pixels_by_size",
                        "expected_coordinates_by_size", "applicable_tiers", "tier_schedule",
                        "validation_pixel_counts", "validation_coordinate_counts",
                    }
                    else "string"
                )}
                for name in PERTURBATION_FIELDS
            ],
        },
    }


def _registry_manifest_schema() -> dict[str, Any]:
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    schema_entry = {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "schema_sha256"],
        "properties": {
            "schema": {"type": "object"},
            "schema_sha256": digest,
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:qsa:schema:registry-manifest:v1",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "manifest_id", "schema_id", "schema_version", "registry_set_version",
            "canonical_format_profile_id", "protocol_id", "protocol_sha256",
            "method_order_sha256", "registry_file_count", "listed_registry_count",
            "registries",
            "source_authorities", "schema_catalog", "invariants", "self_hash_rule",
        ],
        "properties": {
            "manifest_id": {"const": "QSA-REGISTRY-MANIFEST-V1"},
            "schema_id": {"const": "QSA-REGISTRY-MANIFEST-SCHEMA-V1"},
            "schema_version": {"const": 1},
            "registry_set_version": {"const": "1.0.0"},
            "canonical_format_profile_id": {"const": "QSA-CANONICAL-FORMATS-V1"},
            "protocol_id": {"const": "QSA-PROTOCOL-V1"},
            "protocol_sha256": digest,
            "method_order_sha256": digest,
            "registry_file_count": {"const": 7},
            "listed_registry_count": {"const": 6},
            "registries": {
                "type": "array",
                "minItems": 6,
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "path", "registry_id", "schema_id", "schema_version",
                        "format", "record_count", "primary_key",
                        "canonical_sha256", "file_sha256",
                    ],
                    "properties": {
                        "path": {
                            "enum": [
                                "registries/active_modification_schedule.csv",
                                "registries/construction_registry.json",
                                "registries/corpus_registry.csv",
                                "registries/leakage_registry.json",
                                "registries/perturbation_schedule.csv",
                                "registries/protocol_registry.json",
                            ]
                        },
                        "registry_id": {"type": "string", "minLength": 1},
                        "schema_id": {"type": "string", "minLength": 1},
                        "schema_version": {"const": 1},
                        "format": {"enum": ["json", "csv"]},
                        "record_count": {"type": "integer", "minimum": 1},
                        "primary_key": {
                            "type": "array", "minItems": 1,
                            "items": {"type": "string", "minLength": 1},
                        },
                        "canonical_sha256": digest,
                        "file_sha256": digest,
                    },
                },
            },
            "source_authorities": {
                "type": "array",
                "minItems": len(_SOURCE_AUTHORITY_PATHS),
                "maxItems": len(_SOURCE_AUTHORITY_PATHS),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path", "sha256"],
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "sha256": digest,
                    },
                },
            },
            "schema_catalog": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "active_modification_schedule", "construction_registry",
                    "corpus_registry", "leakage_registry", "perturbation_schedule",
                    "protocol_registry", "registry_manifest",
                ],
                "properties": {
                    "active_modification_schedule": schema_entry,
                    "construction_registry": schema_entry,
                    "corpus_registry": schema_entry,
                    "leakage_registry": schema_entry,
                    "perturbation_schedule": schema_entry,
                    "protocol_registry": schema_entry,
                    "registry_manifest": schema_entry,
                },
            },
            "invariants": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "registered_constructions", "authenticated_constructions",
                    "corpus_registry_rows", "standard_corpus_rows",
                    "registered_perturbations", "differential_pairs",
                    "timing_configurations", "active_modification_cases",
                    "authenticated_active_modifications",
                    "authenticated_active_rejections",
                    "unauthenticated_active_modifications",
                    "unauthenticated_active_acceptances",
                    "unauthenticated_plaintext_changed",
                    "unauthenticated_plaintext_unchanged",
                    "parser_length_violations",
                ],
                "properties": {
                    "registered_constructions": {"const": 24},
                    "authenticated_constructions": {"const": 9},
                    "corpus_registry_rows": {"const": 72},
                    "standard_corpus_rows": {"const": 48},
                    "registered_perturbations": {"const": 24},
                    "differential_pairs": {"const": 36768},
                    "timing_configurations": {"const": 576},
                    "active_modification_cases": {"const": 180},
                    "authenticated_active_modifications": {"const": 73},
                    "authenticated_active_rejections": {"const": 73},
                    "unauthenticated_active_modifications": {"const": 107},
                    "unauthenticated_active_acceptances": {"const": 77},
                    "unauthenticated_plaintext_changed": {"const": 37},
                    "unauthenticated_plaintext_unchanged": {"const": 40},
                    "parser_length_violations": {"const": 48},
                },
            },
            "self_hash_rule": {
                "const": "The manifest does not hash itself. SHA256SUMS.txt records its release-file digest."
            },
        },
    }


def _all_registry_schemas() -> dict[str, dict[str, Any]]:
    return {
        **_json_registry_schemas(),
        **_csv_registry_schemas(),
        "registry_manifest": _registry_manifest_schema(),
    }


def _validate_csv_schema(
    payload: bytes,
    schema: Mapping[str, Any],
    expected_count: int,
) -> list[dict[str, str]]:
    text = payload.decode("utf-8")
    if "\r" in text:
        raise ValueError(f"{schema['schema_id']} requires LF line endings")
    if not text.endswith("\n"):
        raise ValueError(f"{schema['schema_id']} requires a final newline")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    expected_fields = [field["name"] for field in schema["fields"]]
    if reader.fieldnames != expected_fields:
        raise ValueError(
            f"{schema['schema_id']} header mismatch: {reader.fieldnames} != {expected_fields}"
        )
    rows = list(reader)
    if len(rows) != expected_count:
        raise ValueError(
            f"{schema['schema_id']} row-count mismatch: {len(rows)} != {expected_count}"
        )
    primary_key = list(schema["primary_key"])
    keys = [tuple(row[name] for name in primary_key) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{schema['schema_id']} contains duplicate primary keys")
    for row in rows:
        for field in schema["fields"]:
            name = field["name"]
            value = row[name]
            kind = field["type"]
            if kind == "integer":
                int(value)
            elif kind == "boolean" and value not in {"true", "false"}:
                raise ValueError(f"invalid boolean in {schema['schema_id']}: {name}={value}")
            elif kind == "sha256" and (len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)):
                raise ValueError(f"invalid SHA-256 in {schema['schema_id']}: {name}")
            elif kind == "canonical_json":
                decoded = json.loads(value)
                if canonical_json_bytes(decoded).decode("ascii") != value:
                    raise ValueError(f"noncanonical JSON in {schema['schema_id']}: {name}")
    return rows


def _validate_registry_payloads(
    files: Mapping[str, bytes],
    schemas: Mapping[str, Mapping[str, Any]],
    config: ProtocolConfig,
) -> None:
    construction = json.loads(files["registries/construction_registry.json"])
    leakage = json.loads(files["registries/leakage_registry.json"])
    protocol = json.loads(files["registries/protocol_registry.json"])
    jsonschema.Draft202012Validator(schemas["construction_registry"]).validate(construction)
    jsonschema.Draft202012Validator(schemas["leakage_registry"]).validate(leakage)
    jsonschema.Draft202012Validator(schemas["protocol_registry"]).validate(protocol)
    corpus_rows = _validate_csv_schema(files["registries/corpus_registry.csv"], schemas["corpus_registry"], 72)
    perturbation_rows = _validate_csv_schema(files["registries/perturbation_schedule.csv"], schemas["perturbation_schedule"], 24)
    active_rows = _validate_csv_schema(files["registries/active_modification_schedule.csv"], schemas["active_modification_schedule"], 180)

    expected_methods = list(EXTENDED_METHOD_FACTORIES)
    for registry_methods in (
        construction["method_order"],
        [row["method_id"] for row in construction["constructions"]],
        [row["method_id"] for row in leakage["methods"]],
        protocol["method_order"],
    ):
        if registry_methods != expected_methods:
            raise ValueError("exported construction order disagrees with the executable registry")
    for item in (construction, leakage, protocol):
        if item["protocol_sha256"] != config.sha256:
            raise ValueError(f"{item['registry_id']} has an incorrect protocol digest")
    if [row["perturbation_id"] for row in perturbation_rows] != list(PERTURBATIONS):
        raise ValueError("exported perturbation order disagrees with the executable registry")
    image_order = config.payload["corpus"]["image_ids"]
    sizes = _registered_sizes(config)
    expected_corpus_order = [f"{image_id}@{size}x{size}" for image_id in image_order for size in sizes]
    actual_corpus_order = [row["registry_row_id"] for row in corpus_rows]
    if actual_corpus_order != expected_corpus_order:
        raise ValueError("corpus registry row order disagrees with the registry schema")
    if protocol["derived_counts"]["differential_pairs"] != 36768:
        raise ValueError("the registered differential-pair count is not 36,768")
    if [int(row["case_ordinal"]) for row in active_rows] != list(range(1, 181)):
        raise ValueError("active-modification case order is incorrect")
    active_counts = {
        "authenticated": sum(row["authenticated"] == "true" for row in active_rows),
        "authenticated_rejections": sum(row["authenticated"] == "true" and row["expected_accepted"] == "false" for row in active_rows),
        "unauthenticated": sum(row["authenticated"] == "false" for row in active_rows),
        "unauthenticated_acceptances": sum(row["authenticated"] == "false" and row["expected_accepted"] == "true" for row in active_rows),
        "changed": sum(row["authenticated"] == "false" and row["expected_plaintext_changed"] == "true" for row in active_rows),
        "unchanged": sum(row["authenticated"] == "false" and row["expected_plaintext_changed"] == "false" for row in active_rows),
        "parser": sum(row["parser_level"] == "true" for row in active_rows),
    }
    if active_counts != {
        "authenticated": 73, "authenticated_rejections": 73,
        "unauthenticated": 107, "unauthenticated_acceptances": 77,
        "changed": 37, "unchanged": 40, "parser": 48,
    }:
        raise ValueError(f"active-modification count partition is incorrect: {active_counts}")


def _registry_record_count(path: str, payload: bytes) -> int:
    if path.endswith(".csv"):
        return len(_read_csv_bytes(payload))
    decoded = json.loads(payload)
    if path.endswith("construction_registry.json"):
        return len(decoded["constructions"])
    if path.endswith("leakage_registry.json"):
        return len(decoded["methods"])
    if path.endswith("protocol_registry.json"):
        return 1
    raise ValueError(path)


def _schema_for_path(path: str) -> str:
    return {
        "registries/active_modification_schedule.csv": "active_modification_schedule",
        "registries/construction_registry.json": "construction_registry",
        "registries/corpus_registry.csv": "corpus_registry",
        "registries/leakage_registry.json": "leakage_registry",
        "registries/perturbation_schedule.csv": "perturbation_schedule",
        "registries/protocol_registry.json": "protocol_registry",
    }[path]


def _primary_key_for_schema(schema: Mapping[str, Any]) -> list[str]:
    if schema.get("format") == "csv":
        return list(schema["primary_key"])
    if schema.get("$id", "").endswith("construction-registry:v1"):
        return ["constructions[].method_id"]
    if schema.get("$id", "").endswith("leakage-registry:v1"):
        return ["methods[].method_id"]
    return ["protocol_id"]


def _build_registry_manifest(
    repo_root: Path,
    files: Mapping[str, bytes],
    schemas: Mapping[str, Mapping[str, Any]],
    config: ProtocolConfig,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(files):
        schema_name = _schema_for_path(path)
        schema = schemas[schema_name]
        if path.endswith(".csv"):
            canonical_digest = _semantic_csv_sha256(files[path])
            format_name = "csv"
        else:
            canonical_digest = hashlib.sha256(
                canonical_json_bytes(json.loads(files[path]))
            ).hexdigest()
            format_name = "json"
        entries.append(
            {
                "path": path,
                "registry_id": (
                    json.loads(files[path])["registry_id"]
                    if path.endswith(".json")
                    else (
                        "QSA-CORPUS-REGISTRY-V1"
                        if path.endswith("corpus_registry.csv")
                        else (
                            "QSA-PERTURBATION-SCHEDULE-V1"
                            if path.endswith("perturbation_schedule.csv")
                            else "QSA-ACTIVE-MODIFICATION-SCHEDULE-V1"
                        )
                    )
                ),
                "schema_id": schema["$id"] if "$id" in schema else schema["schema_id"],
                "schema_version": int(schema.get("schema_version", REGISTRY_SCHEMA_VERSION)),
                "format": format_name,
                "record_count": _registry_record_count(path, files[path]),
                "primary_key": _primary_key_for_schema(schema),
                "canonical_sha256": canonical_digest,
                "file_sha256": _sha256_bytes(files[path]),
            }
        )

    source_authorities: list[dict[str, str]] = []
    for relative in _SOURCE_AUTHORITY_PATHS:
        source = repo_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        source_authorities.append({"path": relative, "sha256": sha256_file(source)})

    schema_catalog: dict[str, Any] = {}
    for name, schema in sorted(schemas.items()):
        schema_catalog[name] = {
            "schema": schema,
            "schema_sha256": hashlib.sha256(canonical_json_bytes(schema)).hexdigest(),
        }

    construction = json.loads(files["registries/construction_registry.json"])
    protocol = json.loads(files["registries/protocol_registry.json"])
    return {
        "manifest_id": "QSA-REGISTRY-MANIFEST-V1",
        "schema_id": "QSA-REGISTRY-MANIFEST-SCHEMA-V1",
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "registry_set_version": REGISTRY_SET_VERSION,
        "canonical_format_profile_id": CANONICAL_FORMAT_PROFILE_ID,
        "protocol_id": config.protocol_id,
        "protocol_sha256": config.sha256,
        "method_order_sha256": hashlib.sha256(
            canonical_json_bytes(config.methods)
        ).hexdigest(),
        "registry_file_count": len(entries) + 1,
        "listed_registry_count": len(entries),
        "registries": entries,
        "source_authorities": source_authorities,
        "schema_catalog": schema_catalog,
        "invariants": {
            "registered_constructions": construction["construction_count"],
            "authenticated_constructions": construction["authenticated_count"],
            "corpus_registry_rows": protocol["derived_counts"]["corpus_registry_rows"],
            "standard_corpus_rows": protocol["derived_counts"]["standard_corpus_rows"],
            "registered_perturbations": protocol["derived_counts"]["registered_perturbations"],
            "differential_pairs": protocol["derived_counts"]["differential_pairs"],
            "timing_configurations": protocol["derived_counts"]["timing_configurations"],
            "active_modification_cases": protocol["derived_counts"]["total_cases"],
            "authenticated_active_modifications": protocol["derived_counts"]["authenticated_cases"],
            "authenticated_active_rejections": protocol["derived_counts"]["authenticated_rejections"],
            "unauthenticated_active_modifications": protocol["derived_counts"]["unauthenticated_cases"],
            "unauthenticated_active_acceptances": protocol["derived_counts"]["unauthenticated_acceptances"],
            "unauthenticated_plaintext_changed": protocol["derived_counts"]["unauthenticated_plaintext_changed"],
            "unauthenticated_plaintext_unchanged": protocol["derived_counts"]["unauthenticated_plaintext_unchanged"],
            "parser_length_violations": protocol["derived_counts"]["parser_length_violations"],
        },
        "self_hash_rule": "The manifest does not hash itself. SHA256SUMS.txt records its release-file digest.",
    }


def _validate_manifest(
    repo_root: Path,
    files: Mapping[str, bytes],
    schemas: Mapping[str, Mapping[str, Any]],
) -> None:
    manifest = json.loads(files["registries/registry_manifest.json"])
    jsonschema.Draft202012Validator(schemas["registry_manifest"]).validate(manifest)

    expected_paths = sorted(
        path for path in files if path != "registries/registry_manifest.json"
    )
    actual_paths = [entry["path"] for entry in manifest["registries"]]
    if actual_paths != expected_paths:
        raise ValueError("registry manifest path inventory is incorrect")
    if manifest["listed_registry_count"] != len(expected_paths):
        raise ValueError("listed registry count is incorrect")
    if manifest["registry_file_count"] != len(expected_paths) + 1:
        raise ValueError("registry file count is incorrect")

    for entry in manifest["registries"]:
        path = entry["path"]
        payload = files[path]
        if entry["file_sha256"] != _sha256_bytes(payload):
            raise ValueError(f"file digest mismatch in registry manifest for {path}")
        canonical = (
            _semantic_csv_sha256(payload)
            if path.endswith(".csv")
            else hashlib.sha256(canonical_json_bytes(json.loads(payload))).hexdigest()
        )
        if entry["canonical_sha256"] != canonical:
            raise ValueError(f"canonical digest mismatch in registry manifest for {path}")

    source_paths = [entry["path"] for entry in manifest["source_authorities"]]
    if source_paths != list(_SOURCE_AUTHORITY_PATHS):
        raise ValueError("registry manifest source-authority inventory is incorrect")
    if len(source_paths) != len(set(source_paths)):
        raise ValueError("registry manifest contains duplicate source authorities")
    for source_entry in manifest["source_authorities"]:
        source = repo_root / source_entry["path"]
        if not source.is_file():
            raise FileNotFoundError(source)
        if source_entry["sha256"] != sha256_file(source):
            raise ValueError(
                f"source-authority digest mismatch for {source_entry['path']}"
            )

    if set(manifest["schema_catalog"]) != set(schemas):
        raise ValueError("registry manifest schema catalog is incomplete")
    for name, schema in schemas.items():
        entry = manifest["schema_catalog"][name]
        if entry["schema"] != schema:
            raise ValueError(f"embedded schema differs from the exporter for {name}")
        expected_digest = hashlib.sha256(canonical_json_bytes(schema)).hexdigest()
        if entry["schema_sha256"] != expected_digest:
            raise ValueError(f"schema digest mismatch for {name}")

    config = load_protocol_config(
        repo_root / "configs/protocol/experiment.json",
        repo_root / "configs/protocol/experiment.schema.json",
    )
    expected_method_digest = hashlib.sha256(
        canonical_json_bytes(config.methods)
    ).hexdigest()
    if manifest["method_order_sha256"] != expected_method_digest:
        raise ValueError("method-order digest mismatch in registry manifest")



def build_registry_files(repo_root: str | Path) -> dict[str, bytes]:
    root = Path(repo_root).resolve()
    config = load_protocol_config(
        root / "configs/protocol/experiment.json",
        root / "configs/protocol/experiment.schema.json",
    )
    corpus_rows, probes = _build_corpus_rows(config)
    perturbation_rows = _build_perturbation_rows(config, probes)
    active_config = load_active_modification_config(root)
    active_rows = build_active_modification_rows(root)
    construction = _build_construction_registry(config)
    leakage = _build_leakage_registry(config)
    protocol = _build_protocol_registry(
        config, corpus_rows, perturbation_rows, active_rows, active_config
    )
    schemas = _all_registry_schemas()

    files: dict[str, bytes] = {
        "registries/active_modification_schedule.csv": _csv_bytes(
            active_rows, ACTIVE_MODIFICATION_FIELDS
        ),
        "registries/construction_registry.json": _human_json_bytes(construction),
        "registries/corpus_registry.csv": _csv_bytes(corpus_rows, CORPUS_FIELDS),
        "registries/leakage_registry.json": _human_json_bytes(leakage),
        "registries/perturbation_schedule.csv": _csv_bytes(
            perturbation_rows, PERTURBATION_FIELDS
        ),
        "registries/protocol_registry.json": _human_json_bytes(protocol),
    }
    _validate_registry_payloads(files, schemas, config)
    manifest = _build_registry_manifest(root, files, schemas, config)
    files["registries/registry_manifest.json"] = _human_json_bytes(manifest)
    _validate_manifest(root, files, schemas)
    return files


def write_registry_files(repo_root: str | Path) -> dict[str, bytes]:
    root = Path(repo_root).resolve()
    files = build_registry_files(root)
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return files


def check_registry_files(repo_root: str | Path) -> list[str]:
    root = Path(repo_root).resolve()
    expected = build_registry_files(root)
    errors: list[str] = []
    for relative, payload in expected.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing registry file: {relative}")
            continue
        actual = path.read_bytes()
        if actual != payload:
            errors.append(f"registry file differs from executable authorities: {relative}")

    registry_root = root / "registries"
    if registry_root.is_dir():
        expected_paths = {Path(relative) for relative in expected}
        actual_paths = {
            path.relative_to(root)
            for path in registry_root.rglob("*")
            if path.is_file()
        }
        for unexpected in sorted(actual_paths - expected_paths, key=lambda item: item.as_posix()):
            errors.append(f"unexpected registry file: {unexpected.as_posix()}")
    return errors


def release_file_paths(repo_root: str | Path) -> list[Path]:
    root = Path(repo_root).resolve()
    excluded_parts = {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "data",
        "results",
        "build",
        "dist",
    }
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        relative = path.relative_to(root)
        if any(
            part in excluded_parts or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path.suffix.lower() in {".pyc", ".pyo", ".so", ".dll", ".dylib", ".whl", ".o", ".a"}:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def write_sha256sums(repo_root: str | Path) -> bytes:
    root = Path(repo_root).resolve()
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in release_file_paths(root)
    ]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    (root / "SHA256SUMS.txt").write_bytes(payload)
    return payload


def verify_sha256sums(repo_root: str | Path) -> list[str]:
    root = Path(repo_root).resolve()
    path = root / "SHA256SUMS.txt"
    if not path.is_file():
        return ["missing SHA256SUMS.txt"]
    errors: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    expected_paths = [item.relative_to(root).as_posix() for item in release_file_paths(root)]
    actual_paths: list[str] = []
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            errors.append(f"invalid checksum line: {line!r}")
            continue
        digest, relative = line[:64], line[66:]
        actual_paths.append(relative)
        target = root / relative
        if not target.is_file():
            errors.append(f"checksum target is missing: {relative}")
            continue
        if sha256_file(target) != digest:
            errors.append(f"checksum mismatch: {relative}")
    if actual_paths != expected_paths:
        errors.append("SHA256SUMS.txt path inventory or ordering is incorrect")
    return errors


def registry_summary(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_path = root / "registries/registry_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "manifest_id": manifest["manifest_id"],
        "registry_set_version": manifest["registry_set_version"],
        "protocol_id": manifest["protocol_id"],
        "protocol_sha256": manifest["protocol_sha256"],
        "registry_file_count": manifest["registry_file_count"],
        "listed_registry_count": manifest["listed_registry_count"],
        "schema_count": len(manifest["schema_catalog"]),
        "invariants": manifest["invariants"],
        "registries": [
            {
                "path": entry["path"],
                "record_count": entry["record_count"],
                "canonical_sha256": entry["canonical_sha256"],
            }
            for entry in manifest["registries"]
        ],
    }
