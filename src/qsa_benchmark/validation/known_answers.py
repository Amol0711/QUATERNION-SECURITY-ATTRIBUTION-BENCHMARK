from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import jsonschema
import numpy as np

from qsa_benchmark.benchmark.envelope import MAGIC, VERSION, encode_envelope, parse_envelope
from qsa_benchmark.benchmark.models import EnvelopeParts, RunContext
from qsa_benchmark.benchmark.registry import EXTENDED_METHOD_FACTORIES, make_method
from qsa_benchmark.benchmark.utils import canonical_json_bytes, sha256_file
from qsa_benchmark.protocol.registry import method_policy
from qsa_benchmark.protocol.schedule import (
    derive_fixed_reference_bytes,
    public_material_length,
)

CONFIG_RELATIVE_PATH = "configs/reference/known_answer_inputs.json"
CONFIG_SCHEMA_RELATIVE_PATH = "configs/reference/known_answer_inputs.schema.json"
INPUT_RELATIVE_PATH = "reference/known_answers/input.rgb"
MANIFEST_RELATIVE_PATH = "reference/known_answers/manifest.json"
OBJECT_DIRECTORY_RELATIVE_PATH = "reference/known_answers/objects"
MANIFEST_ID = "QSA-KNOWN-ANSWER-MANIFEST-V1"
REFERENCE_SCHEMA_VERSION = 1
SHA256_PATTERN = "^[0-9a-f]{64}$"
HEX_PATTERN = "^[0-9a-f]*$"

_SOURCE_AUTHORITY_PATHS = (
    "pyproject.toml",
    CONFIG_RELATIVE_PATH,
    CONFIG_SCHEMA_RELATIVE_PATH,
    "registries/construction_registry.json",
    "registries/registry_manifest.json",
    "registries/protocol_registry.json",
    "src/qsa_benchmark/benchmark/components.py",
    "src/qsa_benchmark/benchmark/constructions.py",
    "src/qsa_benchmark/benchmark/controls.py",
    "src/qsa_benchmark/benchmark/crypto.py",
    "src/qsa_benchmark/benchmark/envelope.py",
    "src/qsa_benchmark/benchmark/external.py",
    "src/qsa_benchmark/benchmark/finite.py",
    "src/qsa_benchmark/benchmark/models.py",
    "src/qsa_benchmark/benchmark/quaternion.py",
    "src/qsa_benchmark/benchmark/registry.py",
    "src/qsa_benchmark/benchmark/serialization.py",
    "src/qsa_benchmark/benchmark/transforms.py",
    "src/qsa_benchmark/benchmark/utils.py",
    "src/qsa_benchmark/protocol/registry.py",
    "src/qsa_benchmark/protocol/schedule.py",
    "src/qsa_benchmark/validation/known_answers.py",
    "scripts/generate_reference_assets.py",
)


def _human_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_repo_path(repo_root: str | Path, relative: str) -> Path:
    root = Path(repo_root).resolve()
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise ValueError(f"unsafe repository-relative path: {relative}")
    target = (root / Path(*pure.parts)).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes repository root: {relative}")
    return target


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _manifest_schema(method_order: list[str]) -> dict[str, Any]:
    digest = {"pattern": SHA256_PATTERN, "type": "string"}
    component_names = (
        "ciphertext_view",
        "descriptor",
        "header",
        "metadata",
        "nonce",
        "object",
        "protected_payload",
        "public_payload",
        "recovered_image",
        "tag",
    )
    lengths = {
        "additionalProperties": False,
        "properties": {
            "ciphertext_view": {"minimum": 0, "type": "integer"},
            "header": {"minimum": 1, "type": "integer"},
            "nonce": {"enum": [12, 16], "type": "integer"},
            "object": {"minimum": 1, "type": "integer"},
            "protected_payload": {"minimum": 0, "type": "integer"},
            "public_payload": {"minimum": 0, "type": "integer"},
            "tag": {"enum": [0, 16, 32], "type": "integer"},
        },
        "required": [
            "ciphertext_view",
            "header",
            "nonce",
            "object",
            "protected_payload",
            "public_payload",
            "tag",
        ],
        "type": "object",
    }
    digests = {
        "additionalProperties": False,
        "properties": {name: digest for name in component_names},
        "required": list(component_names),
        "type": "object",
    }
    context = {
        "additionalProperties": False,
        "properties": {
            "canonical_sha256": digest,
            "image_id": {"type": "string", "minLength": 1},
            "master_key_hex": {"pattern": "^[0-9a-f]{64}$", "type": "string"},
            "nonce_hex": {"pattern": HEX_PATTERN, "type": "string"},
            "protocol_id": {"type": "string", "minLength": 1},
            "public_material_hex": {"pattern": HEX_PATTERN, "type": "string"},
            "public_metadata": {"type": "object"},
            "run_id": {"type": "string", "minLength": 1},
            "seed": {"minimum": 0, "maximum": 18446744073709551615, "type": "integer"},
        },
        "required": [
            "canonical_sha256",
            "image_id",
            "master_key_hex",
            "nonce_hex",
            "protocol_id",
            "public_material_hex",
            "public_metadata",
            "run_id",
            "seed",
        ],
        "type": "object",
    }
    header_consistency = {
        "additionalProperties": False,
        "properties": {
            "image_shape": {
                "items": False,
                "maxItems": 3,
                "minItems": 3,
                "prefixItems": [
                    {"type": "integer", "minimum": 1},
                    {"type": "integer", "minimum": 1},
                    {"const": 3},
                ],
                "type": "array",
            },
            "method_id": {"enum": method_order, "type": "string"},
            "nonce_length": {"minimum": 0, "type": "integer"},
            "protected_length": {"minimum": 0, "type": "integer"},
            "public_length": {"minimum": 0, "type": "integer"},
            "tag_length": {"minimum": 0, "type": "integer"},
        },
        "required": [
            "image_shape",
            "method_id",
            "nonce_length",
            "protected_length",
            "public_length",
            "tag_length",
        ],
        "type": "object",
    }
    case = {
        "additionalProperties": False,
        "properties": {
            "authenticated": {"type": "boolean"},
            "context": context,
            "header_consistency": header_consistency,
            "input_sha256": digest,
            "lengths": lengths,
            "method_id": {"enum": method_order, "type": "string"},
            "object_path": {
                "pattern": "^reference/known_answers/objects/B[0-9]{2}_[a-z0-9_]+\\.qsb$",
                "type": "string",
            },
            "ordinal": {"maximum": 24, "minimum": 1, "type": "integer"},
            "sha256": digests,
        },
        "required": [
            "authenticated",
            "context",
            "header_consistency",
            "input_sha256",
            "lengths",
            "method_id",
            "object_path",
            "ordinal",
            "sha256",
        ],
        "type": "object",
    }
    source_authority = {
        "additionalProperties": False,
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "sha256": digest,
        },
        "required": ["path", "sha256"],
        "type": "object",
    }
    return {
        "$id": "urn:qsa:schema:known-answer-manifest:v1",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": {
            "authenticated_case_count": {"const": 9},
            "case_count": {"const": 24},
            "case_order_sha256": digest,
            "cases": {
                "items": case,
                "maxItems": 24,
                "minItems": 24,
                "type": "array",
            },
            "config_file_sha256": digest,
            "config_schema_sha256": digest,
            "input": {
                "additionalProperties": False,
                "properties": {
                    "byte_length": {"const": 768},
                    "channel_maxima": {
                        "items": {"minimum": 0, "maximum": 255, "type": "integer"},
                        "maxItems": 3,
                        "minItems": 3,
                        "type": "array",
                    },
                    "channel_minima": {
                        "items": {"minimum": 0, "maximum": 255, "type": "integer"},
                        "maxItems": 3,
                        "minItems": 3,
                        "type": "array",
                    },
                    "distinct_byte_values": {"minimum": 1, "maximum": 256, "type": "integer"},
                    "dtype": {"const": "uint8"},
                    "generator_id": {"const": "coordinate-mix-v1"},
                    "layout": {"const": "C-order RGB"},
                    "path": {"const": INPUT_RELATIVE_PATH},
                    "sha256": digest,
                    "shape": {
                        "items": False,
                        "maxItems": 3,
                        "minItems": 3,
                        "prefixItems": [{"const": 16}, {"const": 16}, {"const": 3}],
                        "type": "array",
                    },
                },
                "required": [
                    "byte_length",
                    "channel_maxima",
                    "channel_minima",
                    "distinct_byte_values",
                    "dtype",
                    "generator_id",
                    "layout",
                    "path",
                    "sha256",
                    "shape",
                ],
                "type": "object",
            },
            "manifest_id": {"const": MANIFEST_ID},
            "method_order": {
                "items": False,
                "maxItems": 24,
                "minItems": 24,
                "prefixItems": [{"const": method_id} for method_id in method_order],
                "type": "array",
            },
            "method_order_sha256": digest,
            "method_profile": {"const": "extended"},
            "object_format": {
                "additionalProperties": False,
                "properties": {
                    "component_order": {
                        "const": [
                            "header",
                            "nonce",
                            "public_payload",
                            "protected_payload",
                            "tag",
                        ]
                    },
                    "magic_ascii": {"const": "QSB1"},
                    "version": {"const": 1},
                },
                "required": ["component_order", "magic_ascii", "version"],
                "type": "object",
            },
            "object_set_sha256": digest,
            "operational_use_prohibited": {"const": True},
            "protocol_id": {"const": "QSA-KNOWN-ANSWER-V1"},
            "protocol_registry_sha256": digest,
            "public_test_values": {"const": True},
            "reference_semantics": {
                "additionalProperties": False,
                "properties": {
                    "fixed_known_answer": {
                        "const": (
                            "Each case compares the complete serialized object and "
                            "component digests against frozen expected values."
                        )
                    },
                    "round_trip": {
                        "const": (
                            "Exact recovery is an additional functional check and does "
                            "not replace the frozen expected-output comparison."
                        )
                    },
                },
                "required": ["fixed_known_answer", "round_trip"],
                "type": "object",
            },
            "reference_set_id": {"const": "QSA-KNOWN-ANSWER-SET-V1"},
            "registry_manifest_sha256": digest,
            "schema_version": {"const": REFERENCE_SCHEMA_VERSION},
            "source_authorities": {
                "items": source_authority,
                "minItems": len(_SOURCE_AUTHORITY_PATHS),
                "maxItems": len(_SOURCE_AUTHORITY_PATHS),
                "type": "array",
            },
            "test_value_notice": {"type": "string", "minLength": 1},
            "total_object_bytes": {"minimum": 1, "type": "integer"},
        },
        "required": [
            "authenticated_case_count",
            "case_count",
            "case_order_sha256",
            "cases",
            "config_file_sha256",
            "config_schema_sha256",
            "input",
            "manifest_id",
            "method_order",
            "method_order_sha256",
            "method_profile",
            "object_format",
            "object_set_sha256",
            "operational_use_prohibited",
            "protocol_id",
            "protocol_registry_sha256",
            "public_test_values",
            "reference_semantics",
            "reference_set_id",
            "registry_manifest_sha256",
            "schema_version",
            "source_authorities",
            "test_value_notice",
            "total_object_bytes",
        ],
        "type": "object",
    }


def generate_input_image(config: Mapping[str, Any]) -> np.ndarray:
    input_config = config["input"]
    shape = tuple(int(value) for value in input_config["shape"])
    if shape != (16, 16, 3) or input_config["generator_id"] != "coordinate-mix-v1":
        raise ValueError("unsupported fixed known-answer input definition")
    row, column = np.indices(shape[:2], dtype=np.int64)
    red = (11 + 17 * row + 29 * column + 3 * row * column + 5 * row * row) % 256
    green = (
        53
        + 31 * row
        + 7 * column
        + 5 * np.bitwise_xor(row, column)
        + 2 * column * column
    ) % 256
    blue = (
        101
        + 13 * row
        + 19 * column
        + 9 * ((row + column) % 17)
        + 7 * row * column
    ) % 256
    image = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)
    return image


def load_known_answer_config(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config_path = _safe_repo_path(root, CONFIG_RELATIVE_PATH)
    schema_path = _safe_repo_path(root, CONFIG_SCHEMA_RELATIVE_PATH)
    config = _load_json_object(config_path)
    schema = _load_json_object(schema_path)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(config)

    method_order = list(EXTENDED_METHOD_FACTORIES)
    if config["method_order"] != method_order:
        raise ValueError("known-answer method order differs from the executable registry")
    if config["case_count"] != len(method_order):
        raise ValueError("known-answer case count differs from the executable registry")
    image = generate_input_image(config)
    input_bytes = image.tobytes(order="C")
    if len(input_bytes) != config["input"]["byte_length"]:
        raise ValueError("known-answer input byte length mismatch")
    if _sha256_bytes(input_bytes) != config["input"]["sha256"]:
        raise ValueError("known-answer input digest mismatch")
    if config["input"]["relative_path"] != INPUT_RELATIVE_PATH:
        raise ValueError("known-answer input path differs from the public contract")
    if bytes.fromhex(config["context_profile"]["master_key_hex"]) != bytes(range(32)):
        raise ValueError("known-answer master key differs from the fixed public test key")
    forbidden = {"image_id", "run_id", "pair_id", "key_index", "state_index", "perturbation_id"}
    overlap = forbidden.intersection(config["context_profile"]["public_metadata"])
    if overlap:
        raise ValueError(f"known-answer public metadata contains prohibited identifiers: {sorted(overlap)}")
    return config


def derive_known_answer_context(
    config: Mapping[str, Any],
    *,
    method_id: str,
    ordinal: int,
    input_sha256: str,
    payload_length: int,
) -> RunContext:
    if ordinal < 1 or ordinal > len(config["method_order"]):
        raise ValueError("known-answer method ordinal is outside the registered range")
    if config["method_order"][ordinal - 1] != method_id:
        raise ValueError("known-answer method ordinal mismatch")
    context_profile = config["context_profile"]
    policy = method_policy(method_id)
    nonce = derive_fixed_reference_bytes(
        domain_ascii=context_profile["nonce_domain_ascii"],
        method_id=method_id,
        ordinal=ordinal,
        input_sha256=input_sha256,
        length=policy.nonce_length,
    )
    seed = int.from_bytes(
        derive_fixed_reference_bytes(
            domain_ascii=context_profile["seed_domain_ascii"],
            method_id=method_id,
            ordinal=ordinal,
            input_sha256=input_sha256,
            length=8,
        ),
        "big",
    )
    material_length = public_material_length(method_id, payload_length)
    public_material = derive_fixed_reference_bytes(
        domain_ascii=context_profile["public_material_domain_ascii"],
        method_id=method_id,
        ordinal=ordinal,
        input_sha256=input_sha256,
        length=material_length,
    )
    return RunContext(
        master_key=bytes.fromhex(context_profile["master_key_hex"]),
        nonce=nonce,
        seed=seed,
        image_id=context_profile["image_id"],
        method_id=method_id,
        run_id=context_profile["run_id_template"].format(method_id=method_id),
        public_metadata=dict(context_profile["public_metadata"]),
        public_material=public_material,
        protocol_id=context_profile["protocol_id"],
    )


def _context_record(context: RunContext) -> dict[str, Any]:
    core = {
        "image_id": context.image_id,
        "master_key_hex": context.master_key.hex(),
        "nonce_hex": context.nonce.hex(),
        "protocol_id": context.protocol_id,
        "public_material_hex": context.public_material.hex(),
        "public_metadata": dict(context.public_metadata or {}),
        "run_id": context.run_id,
        "seed": int(context.seed),
    }
    return {"canonical_sha256": _sha256_bytes(canonical_json_bytes(core)), **core}


def _component_record(payload: bytes) -> tuple[int, str]:
    return len(payload), _sha256_bytes(payload)


def _case_record(
    *,
    method_id: str,
    ordinal: int,
    input_image: np.ndarray,
    input_sha256: str,
    context: RunContext,
    object_bytes: bytes,
    ciphertext_view: bytes,
) -> dict[str, Any]:
    method = make_method(method_id, profile="extended")
    parsed = parse_envelope(object_bytes)
    recovered = method.decrypt(object_bytes, context)
    if parsed.header["method_id"] != method_id:
        raise RuntimeError(f"known-answer method mismatch for {method_id}")
    if tuple(parsed.header["image_shape"]) != tuple(input_image.shape):
        raise RuntimeError(f"known-answer image-shape mismatch for {method_id}")
    if parsed.nonce != context.nonce:
        raise RuntimeError(f"known-answer nonce mismatch for {method_id}")
    if parsed.header["metadata"] != dict(context.public_metadata or {}):
        raise RuntimeError(f"known-answer metadata mismatch for {method_id}")
    if not np.array_equal(recovered, input_image):
        raise RuntimeError(f"known-answer exact recovery failed for {method_id}")

    component_payloads = {
        "ciphertext_view": ciphertext_view,
        "descriptor": canonical_json_bytes(parsed.header["descriptor"]),
        "header": parsed.header_bytes,
        "metadata": canonical_json_bytes(parsed.header["metadata"]),
        "nonce": parsed.nonce,
        "object": object_bytes,
        "protected_payload": parsed.protected_payload,
        "public_payload": parsed.public_payload,
        "recovered_image": recovered.tobytes(order="C"),
        "tag": parsed.tag,
    }
    lengths = {
        name: _component_record(payload)[0]
        for name, payload in component_payloads.items()
        if name in {
            "ciphertext_view",
            "header",
            "nonce",
            "object",
            "protected_payload",
            "public_payload",
            "tag",
        }
    }
    digests = {name: _sha256_bytes(payload) for name, payload in component_payloads.items()}
    return {
        "authenticated": bool(method.authenticated),
        "context": _context_record(context),
        "header_consistency": {
            "image_shape": list(parsed.header["image_shape"]),
            "method_id": parsed.header["method_id"],
            "nonce_length": int(parsed.header["nonce_length"]),
            "protected_length": int(parsed.header["protected_length"]),
            "public_length": int(parsed.header["public_length"]),
            "tag_length": int(parsed.header["tag_length"]),
        },
        "input_sha256": input_sha256,
        "lengths": lengths,
        "method_id": method_id,
        "object_path": f"{OBJECT_DIRECTORY_RELATIVE_PATH}/{method_id}.qsb",
        "ordinal": ordinal,
        "sha256": digests,
    }


def _source_authorities(repo_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relative in _SOURCE_AUTHORITY_PATHS:
        path = _safe_repo_path(repo_root, relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"path": relative, "sha256": sha256_file(path)})
    return rows


def build_known_answer_files(repo_root: str | Path) -> dict[str, bytes]:
    root = Path(repo_root).resolve()
    config = load_known_answer_config(root)
    input_image = generate_input_image(config)
    input_bytes = input_image.tobytes(order="C")
    input_sha256 = _sha256_bytes(input_bytes)
    method_order = list(config["method_order"])

    files: dict[str, bytes] = {INPUT_RELATIVE_PATH: input_bytes}
    cases: list[dict[str, Any]] = []
    for ordinal, method_id in enumerate(method_order, start=1):
        context = derive_known_answer_context(
            config,
            method_id=method_id,
            ordinal=ordinal,
            input_sha256=input_sha256,
            payload_length=len(input_bytes),
        )
        method = make_method(method_id, profile="extended")
        output = method.encrypt(input_image, context)
        case = _case_record(
            method_id=method_id,
            ordinal=ordinal,
            input_image=input_image,
            input_sha256=input_sha256,
            context=context,
            object_bytes=output.object_bytes,
            ciphertext_view=output.ciphertext_view,
        )
        files[case["object_path"]] = output.object_bytes
        cases.append(case)

    authenticated_count = sum(bool(case["authenticated"]) for case in cases)
    if len(cases) != 24 or authenticated_count != 9:
        raise RuntimeError("known-answer cardinality differs from the registered method set")

    object_rows = [
        {
            "method_id": case["method_id"],
            "object_length_bytes": case["lengths"]["object"],
            "object_path": case["object_path"],
            "object_sha256": case["sha256"]["object"],
        }
        for case in cases
    ]
    case_order_rows = [
        {
            "context_sha256": case["context"]["canonical_sha256"],
            "method_id": case["method_id"],
            "object_sha256": case["sha256"]["object"],
            "ordinal": case["ordinal"],
        }
        for case in cases
    ]
    manifest = {
        "authenticated_case_count": authenticated_count,
        "case_count": len(cases),
        "case_order_sha256": _sha256_bytes(canonical_json_bytes(case_order_rows)),
        "cases": cases,
        "config_file_sha256": sha256_file(root / CONFIG_RELATIVE_PATH),
        "config_schema_sha256": sha256_file(root / CONFIG_SCHEMA_RELATIVE_PATH),
        "input": {
            "byte_length": len(input_bytes),
            "channel_maxima": [int(input_image[..., index].max()) for index in range(3)],
            "channel_minima": [int(input_image[..., index].min()) for index in range(3)],
            "distinct_byte_values": int(len(np.unique(input_image))),
            "dtype": str(input_image.dtype),
            "generator_id": config["input"]["generator_id"],
            "layout": config["input"]["layout"],
            "path": config["input"]["relative_path"],
            "sha256": input_sha256,
            "shape": list(input_image.shape),
        },
        "manifest_id": MANIFEST_ID,
        "method_order": method_order,
        "method_order_sha256": _sha256_bytes(canonical_json_bytes(method_order)),
        "method_profile": config["method_profile"],
        "object_format": {
            "component_order": [
                "header",
                "nonce",
                "public_payload",
                "protected_payload",
                "tag",
            ],
            "magic_ascii": MAGIC.decode("ascii"),
            "version": VERSION,
        },
        "object_set_sha256": _sha256_bytes(canonical_json_bytes(object_rows)),
        "operational_use_prohibited": bool(config["operational_use_prohibited"]),
        "protocol_id": config["context_profile"]["protocol_id"],
        "protocol_registry_sha256": sha256_file(root / "registries/protocol_registry.json"),
        "public_test_values": bool(config["public_test_values"]),
        "reference_semantics": {
            "fixed_known_answer": (
                "Each case compares the complete serialized object and component "
                "digests against frozen expected values."
            ),
            "round_trip": (
                "Exact recovery is an additional functional check and does not replace "
                "the frozen expected-output comparison."
            ),
        },
        "reference_set_id": config["reference_set_id"],
        "registry_manifest_sha256": sha256_file(root / "registries/registry_manifest.json"),
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "source_authorities": _source_authorities(root),
        "test_value_notice": (
            "All keys, nonces, seeds, and public materials in this reference set are "
            "public test values. They must not be reused in operational systems."
        ),
        "total_object_bytes": sum(int(case["lengths"]["object"]) for case in cases),
    }
    schema = _manifest_schema(method_order)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(manifest)
    files[MANIFEST_RELATIVE_PATH] = _human_json_bytes(manifest)
    return dict(sorted(files.items()))


def write_known_answer_files(repo_root: str | Path) -> dict[str, bytes]:
    root = Path(repo_root).resolve()
    files = build_known_answer_files(root)
    reference_root = _safe_repo_path(root, "reference/known_answers")
    if reference_root.is_dir():
        expected_paths = {Path(relative) for relative in files}
        for path in sorted(reference_root.rglob("*"), reverse=True):
            if path.is_file() and path.relative_to(root) not in expected_paths:
                path.unlink()
        for path in sorted(reference_root.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
    for relative, payload in files.items():
        target = _safe_repo_path(root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return files


def load_known_answer_manifest(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest = _load_json_object(_safe_repo_path(root, MANIFEST_RELATIVE_PATH))
    method_order = list(EXTENDED_METHOD_FACTORIES)
    schema = _manifest_schema(method_order)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(manifest)
    return manifest


def check_known_answer_files(repo_root: str | Path) -> list[str]:
    root = Path(repo_root).resolve()
    expected = build_known_answer_files(root)
    errors: list[str] = []
    for relative, payload in expected.items():
        path = _safe_repo_path(root, relative)
        if not path.is_file():
            errors.append(f"missing known-answer file: {relative}")
        elif path.read_bytes() != payload:
            errors.append(f"known-answer file differs from frozen executable reference: {relative}")

    reference_root = _safe_repo_path(root, "reference/known_answers")
    actual = {
        path.relative_to(root).as_posix()
        for path in reference_root.rglob("*")
        if path.is_file()
    } if reference_root.is_dir() else set()
    for unexpected in sorted(actual - set(expected)):
        errors.append(f"unexpected known-answer file: {unexpected}")
    return errors


def verify_known_answer_assets(repo_root: str | Path) -> list[str]:
    root = Path(repo_root).resolve()
    errors = check_known_answer_files(root)
    try:
        config = load_known_answer_config(root)
        manifest = load_known_answer_manifest(root)
        method_order = list(EXTENDED_METHOD_FACTORIES)
        if manifest["method_order"] != method_order:
            errors.append("known-answer manifest method ordering differs from the registry")
        if [case["method_id"] for case in manifest["cases"]] != method_order:
            errors.append("known-answer case ordering differs from the registry")

        for source in manifest["source_authorities"]:
            if sha256_file(root / source["path"]) != source["sha256"]:
                errors.append(f"known-answer source authority changed: {source['path']}")

        input_path = _safe_repo_path(root, manifest["input"]["path"])
        input_bytes = input_path.read_bytes()
        if _sha256_bytes(input_bytes) != manifest["input"]["sha256"]:
            errors.append("known-answer input digest mismatch")
        image = np.frombuffer(input_bytes, dtype=np.uint8).reshape(manifest["input"]["shape"]).copy()

        for case in manifest["cases"]:
            method_id = case["method_id"]
            try:
                object_path = _safe_repo_path(root, case["object_path"])
                object_bytes = object_path.read_bytes()
                if len(object_bytes) != case["lengths"]["object"]:
                    raise AssertionError("object length mismatch")
                if _sha256_bytes(object_bytes) != case["sha256"]["object"]:
                    raise AssertionError("object digest mismatch")
                ordinal = int(case["ordinal"])
                context = derive_known_answer_context(
                    config,
                    method_id=method_id,
                    ordinal=ordinal,
                    input_sha256=manifest["input"]["sha256"],
                    payload_length=len(input_bytes),
                )
                if _context_record(context) != case["context"]:
                    raise AssertionError("context mismatch")

                parsed = parse_envelope(object_bytes)
                component_payloads = {
                    "descriptor": canonical_json_bytes(parsed.header["descriptor"]),
                    "header": parsed.header_bytes,
                    "metadata": canonical_json_bytes(parsed.header["metadata"]),
                    "nonce": parsed.nonce,
                    "object": object_bytes,
                    "protected_payload": parsed.protected_payload,
                    "public_payload": parsed.public_payload,
                    "tag": parsed.tag,
                }
                for name, payload in component_payloads.items():
                    if _sha256_bytes(payload) != case["sha256"][name]:
                        raise AssertionError(f"{name} digest mismatch")
                reconstructed = encode_envelope(
                    EnvelopeParts(
                        method_id=parsed.header["method_id"],
                        image_shape=tuple(parsed.header["image_shape"]),
                        descriptor=parsed.header["descriptor"],
                        metadata=parsed.header["metadata"],
                        nonce=parsed.nonce,
                        public_payload=parsed.public_payload,
                        protected_payload=parsed.protected_payload,
                        tag=parsed.tag,
                    )
                )
                if reconstructed != object_bytes:
                    raise AssertionError("canonical envelope reconstruction mismatch")

                method = make_method(method_id, profile="extended")
                recovered = method.decrypt(object_bytes, context)
                if not np.array_equal(recovered, image):
                    raise AssertionError("exact recovery mismatch")
                if _sha256_bytes(recovered.tobytes(order="C")) != case["sha256"]["recovered_image"]:
                    raise AssertionError("recovered-image digest mismatch")
                regenerated = method.encrypt(image, context)
                if regenerated.object_bytes != object_bytes:
                    raise AssertionError("regenerated object differs from the frozen answer")
                if _sha256_bytes(regenerated.ciphertext_view) != case["sha256"]["ciphertext_view"]:
                    raise AssertionError("ciphertext-view digest mismatch")
                if len(regenerated.ciphertext_view) != case["lengths"]["ciphertext_view"]:
                    raise AssertionError("ciphertext-view length mismatch")
            except Exception as exc:
                errors.append(f"{method_id}: {type(exc).__name__}: {exc}")
    except Exception as exc:
        errors.append(f"known-answer verification failed: {type(exc).__name__}: {exc}")
    return errors


def known_answer_summary(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest = load_known_answer_manifest(root)
    return {
        "authenticated_case_count": manifest["authenticated_case_count"],
        "case_count": manifest["case_count"],
        "input_sha256": manifest["input"]["sha256"],
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": sha256_file(root / MANIFEST_RELATIVE_PATH),
        "object_set_sha256": manifest["object_set_sha256"],
        "reference_set_id": manifest["reference_set_id"],
        "source_authority_count": len(manifest["source_authorities"]),
        "total_object_bytes": manifest["total_object_bytes"],
    }
