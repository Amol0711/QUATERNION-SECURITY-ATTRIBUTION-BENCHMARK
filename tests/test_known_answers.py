from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from qsa_benchmark.benchmark.envelope import parse_envelope
from qsa_benchmark.benchmark.registry import EXTENDED_METHOD_FACTORIES, make_method
from qsa_benchmark.benchmark.utils import canonical_json_bytes, sha256_file
from qsa_benchmark.protocol.registry import method_policy
from qsa_benchmark.protocol.schedule import (
    derive_fixed_reference_bytes,
    public_material_length,
)
from qsa_benchmark.validation.known_answers import (
    CONFIG_RELATIVE_PATH,
    CONFIG_SCHEMA_RELATIVE_PATH,
    INPUT_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    OBJECT_DIRECTORY_RELATIVE_PATH,
    build_known_answer_files,
    check_known_answer_files,
    derive_known_answer_context,
    generate_input_image,
    load_known_answer_config,
    load_known_answer_manifest,
    verify_known_answer_assets,
)

ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = list(EXTENDED_METHOD_FACTORIES)


@pytest.fixture(scope="session")
def known_answer_config() -> dict[str, object]:
    return load_known_answer_config(ROOT)


@pytest.fixture(scope="session")
def known_answer_manifest() -> dict[str, object]:
    return load_known_answer_manifest(ROOT)


def test_known_answer_input_configuration_is_valid_and_nondegenerate(
    known_answer_config: dict[str, object],
    known_answer_manifest: dict[str, object],
) -> None:
    schema = json.loads(
        (ROOT / CONFIG_SCHEMA_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(known_answer_config)

    image = generate_input_image(known_answer_config)
    payload = image.tobytes(order="C")
    pixels = image.reshape(-1, 3)
    eigenvalues = np.linalg.eigvalsh(
        np.cov(pixels.astype(np.float64), rowvar=False)
    )

    assert image.shape == (16, 16, 3)
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    assert len(payload) == 768
    assert payload == (ROOT / INPUT_RELATIVE_PATH).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == known_answer_config["input"][
        "sha256"
    ]
    assert hashlib.sha256(payload).hexdigest() == known_answer_manifest["input"][
        "sha256"
    ]
    assert len(np.unique(pixels, axis=0)) == 256
    assert all(len(np.unique(image[..., channel])) >= 90 for channel in range(3))
    assert np.all(eigenvalues > 4_000.0)
    assert np.min(np.diff(eigenvalues)) > 500.0


def test_committed_known_answers_match_all_executable_authorities() -> None:
    expected = build_known_answer_files(ROOT)
    assert len(expected) == 26
    assert set(expected) == {
        INPUT_RELATIVE_PATH,
        MANIFEST_RELATIVE_PATH,
        *(
            f"{OBJECT_DIRECTORY_RELATIVE_PATH}/{method_id}.qsb"
            for method_id in METHOD_ORDER
        ),
    }
    for relative, payload in expected.items():
        assert (ROOT / relative).read_bytes() == payload
    assert check_known_answer_files(ROOT) == []
    assert verify_known_answer_assets(ROOT) == []


def test_known_answer_manifest_has_exact_registered_order_and_counts(
    known_answer_config: dict[str, object],
    known_answer_manifest: dict[str, object],
) -> None:
    manifest = known_answer_manifest
    cases = manifest["cases"]
    construction_registry = json.loads(
        (ROOT / "registries/construction_registry.json").read_text(encoding="utf-8")
    )

    assert known_answer_config["method_order"] == METHOD_ORDER
    assert manifest["method_order"] == METHOD_ORDER
    assert construction_registry["method_order"] == METHOD_ORDER
    assert [case["method_id"] for case in cases] == METHOD_ORDER
    assert [case["ordinal"] for case in cases] == list(range(1, 25))
    assert manifest["case_count"] == known_answer_config["case_count"] == 24
    assert manifest["authenticated_case_count"] == 9
    assert sum(bool(case["authenticated"]) for case in cases) == 9
    assert manifest["public_test_values"] is True
    assert manifest["operational_use_prohibited"] is True
    assert manifest["input"]["sha256"] == cases[0]["input_sha256"]
    assert manifest["method_order_sha256"] == hashlib.sha256(
        canonical_json_bytes(METHOD_ORDER)
    ).hexdigest()
    assert len({case["sha256"]["object"] for case in cases}) == 24
    assert len({case["context"]["canonical_sha256"] for case in cases}) == 24
    assert len({case["object_path"] for case in cases}) == 24

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
    assert manifest["object_set_sha256"] == hashlib.sha256(
        canonical_json_bytes(object_rows)
    ).hexdigest()
    assert manifest["case_order_sha256"] == hashlib.sha256(
        canonical_json_bytes(case_order_rows)
    ).hexdigest()
    assert manifest["total_object_bytes"] == sum(
        case["lengths"]["object"] for case in cases
    )
    assert "complete serialized object" in manifest["reference_semantics"][
        "fixed_known_answer"
    ]
    assert "does not replace" in manifest["reference_semantics"]["round_trip"]


def test_known_answer_source_authorities_are_complete_and_current(
    known_answer_manifest: dict[str, object],
) -> None:
    authorities = known_answer_manifest["source_authorities"]
    paths = [entry["path"] for entry in authorities]
    assert len(paths) == len(set(paths)) == 23
    assert {
        CONFIG_RELATIVE_PATH,
        CONFIG_SCHEMA_RELATIVE_PATH,
        "registries/construction_registry.json",
        "registries/protocol_registry.json",
        "registries/registry_manifest.json",
        "src/qsa_benchmark/protocol/schedule.py",
        "src/qsa_benchmark/validation/known_answers.py",
        "scripts/generate_reference_assets.py",
    } <= set(paths)
    for entry in authorities:
        assert entry["sha256"] == sha256_file(ROOT / entry["path"])
    assert known_answer_manifest["config_file_sha256"] == sha256_file(
        ROOT / CONFIG_RELATIVE_PATH
    )
    assert known_answer_manifest["config_schema_sha256"] == sha256_file(
        ROOT / CONFIG_SCHEMA_RELATIVE_PATH
    )
    assert known_answer_manifest["protocol_registry_sha256"] == sha256_file(
        ROOT / "registries/protocol_registry.json"
    )
    assert known_answer_manifest["registry_manifest_sha256"] == sha256_file(
        ROOT / "registries/registry_manifest.json"
    )


@pytest.mark.parametrize(
    ("ordinal", "method_id"),
    list(enumerate(METHOD_ORDER, start=1)),
)
def test_every_known_answer_checks_exact_object_components_and_recovery(
    ordinal: int,
    method_id: str,
    known_answer_config: dict[str, object],
    known_answer_manifest: dict[str, object],
) -> None:
    manifest = known_answer_manifest
    case = manifest["cases"][ordinal - 1]
    input_bytes = (ROOT / INPUT_RELATIVE_PATH).read_bytes()
    image = np.frombuffer(input_bytes, dtype=np.uint8).reshape(16, 16, 3).copy()
    context = derive_known_answer_context(
        known_answer_config,
        method_id=method_id,
        ordinal=ordinal,
        input_sha256=manifest["input"]["sha256"],
        payload_length=len(input_bytes),
    )
    object_path = ROOT / case["object_path"]
    object_bytes = object_path.read_bytes()
    parsed = parse_envelope(object_bytes)
    method = make_method(method_id, profile="extended")
    regenerated = method.encrypt(image, context)
    recovered = method.decrypt(object_bytes, context)

    assert case["method_id"] == method_id
    assert case["ordinal"] == ordinal
    assert object_path.parent == ROOT / OBJECT_DIRECTORY_RELATIVE_PATH
    assert object_path.name == f"{method_id}.qsb"
    assert regenerated.object_bytes == object_bytes
    assert np.array_equal(recovered, image)
    assert hashlib.sha256(object_bytes).hexdigest() == case["sha256"]["object"]
    assert hashlib.sha256(parsed.header_bytes).hexdigest() == case["sha256"][
        "header"
    ]
    assert hashlib.sha256(parsed.nonce).hexdigest() == case["sha256"]["nonce"]
    assert hashlib.sha256(parsed.public_payload).hexdigest() == case["sha256"][
        "public_payload"
    ]
    assert hashlib.sha256(parsed.protected_payload).hexdigest() == case["sha256"][
        "protected_payload"
    ]
    assert hashlib.sha256(parsed.tag).hexdigest() == case["sha256"]["tag"]
    assert hashlib.sha256(
        canonical_json_bytes(parsed.header["descriptor"])
    ).hexdigest() == case["sha256"]["descriptor"]
    assert hashlib.sha256(
        canonical_json_bytes(parsed.header["metadata"])
    ).hexdigest() == case["sha256"]["metadata"]
    assert hashlib.sha256(regenerated.ciphertext_view).hexdigest() == case["sha256"][
        "ciphertext_view"
    ]
    assert hashlib.sha256(recovered.tobytes(order="C")).hexdigest() == case[
        "sha256"
    ]["recovered_image"]
    assert case["input_sha256"] == manifest["input"]["sha256"]

    assert len(object_bytes) == case["lengths"]["object"]
    assert len(parsed.header_bytes) == case["lengths"]["header"]
    assert len(parsed.nonce) == case["lengths"]["nonce"]
    assert len(parsed.public_payload) == case["lengths"]["public_payload"]
    assert len(parsed.protected_payload) == case["lengths"]["protected_payload"]
    assert len(parsed.tag) == case["lengths"]["tag"]
    assert len(regenerated.ciphertext_view) == case["lengths"]["ciphertext_view"]

    assert tuple(parsed.header["image_shape"]) == (16, 16, 3)
    assert parsed.header["method_id"] == method_id
    assert parsed.header["metadata"] == dict(context.public_metadata or {})
    assert parsed.nonce == context.nonce
    assert case["header_consistency"] == {
        "image_shape": [16, 16, 3],
        "method_id": method_id,
        "nonce_length": len(parsed.nonce),
        "protected_length": len(parsed.protected_payload),
        "public_length": len(parsed.public_payload),
        "tag_length": len(parsed.tag),
    }
    assert case["authenticated"] == bool(method.authenticated)
    assert (len(parsed.tag) > 0) == bool(case["authenticated"])
    assert len(context.nonce) == method_policy(method_id).nonce_length
    assert len(context.public_material) == public_material_length(method_id, 768)

    context_core = {
        "image_id": context.image_id,
        "master_key_hex": context.master_key.hex(),
        "nonce_hex": context.nonce.hex(),
        "protocol_id": context.protocol_id,
        "public_material_hex": context.public_material.hex(),
        "public_metadata": dict(context.public_metadata or {}),
        "run_id": context.run_id,
        "seed": int(context.seed),
    }
    assert case["context"] == {
        "canonical_sha256": hashlib.sha256(
            canonical_json_bytes(context_core)
        ).hexdigest(),
        **context_core,
    }


def test_known_answer_context_lengths_follow_registered_policies(
    known_answer_manifest: dict[str, object],
) -> None:
    payload_length = known_answer_manifest["input"]["byte_length"]
    public_material_cases = 0
    for case in known_answer_manifest["cases"]:
        method_id = case["method_id"]
        nonce = bytes.fromhex(case["context"]["nonce_hex"])
        public_material = bytes.fromhex(case["context"]["public_material_hex"])
        assert len(nonce) == method_policy(method_id).nonce_length
        assert len(public_material) == public_material_length(method_id, payload_length)
        public_material_cases += bool(public_material)
    assert public_material_cases == 2


def test_fixed_reference_derivation_rejects_ambiguous_inputs() -> None:
    with pytest.raises(ValueError):
        derive_fixed_reference_bytes(
            domain_ascii="",
            method_id="B01_aes_gcm",
            ordinal=1,
            input_sha256="0" * 64,
            length=12,
        )
    with pytest.raises(ValueError):
        derive_fixed_reference_bytes(
            domain_ascii="QSA-TEST",
            method_id="B01_aes_gcm",
            ordinal=0,
            input_sha256="0" * 64,
            length=12,
        )
    with pytest.raises(ValueError):
        derive_fixed_reference_bytes(
            domain_ascii="QSA-TEST",
            method_id="B01_aes_gcm",
            ordinal=1,
            input_sha256="A" * 64,
            length=12,
        )
    with pytest.raises(ValueError):
        derive_fixed_reference_bytes(
            domain_ascii="QSA-TEST",
            method_id="B01_aes_gcm",
            ordinal=1,
            input_sha256="0" * 64,
            length=-1,
        )


def test_fixed_known_answers_are_stronger_than_round_trip_only(
    known_answer_config: dict[str, object],
    known_answer_manifest: dict[str, object],
) -> None:
    method_id = "B13_permutation_only"
    ordinal = METHOD_ORDER.index(method_id) + 1
    case = known_answer_manifest["cases"][ordinal - 1]
    image = generate_input_image(known_answer_config)
    method = make_method(method_id, profile="extended")
    fixed_context = derive_known_answer_context(
        known_answer_config,
        method_id=method_id,
        ordinal=ordinal,
        input_sha256=known_answer_manifest["input"]["sha256"],
        payload_length=image.nbytes,
    )
    changed_context = replace(
        fixed_context,
        nonce=bytes([fixed_context.nonce[0] ^ 1]) + fixed_context.nonce[1:],
        run_id=fixed_context.run_id + "|alternate",
    )
    alternate = method.encrypt(image, changed_context)
    recovered = method.decrypt(alternate.object_bytes, changed_context)

    assert np.array_equal(recovered, image)
    assert alternate.object_bytes != (ROOT / case["object_path"]).read_bytes()
    assert hashlib.sha256(alternate.object_bytes).hexdigest() != case["sha256"][
        "object"
    ]


def test_committed_object_tampering_is_detected(tmp_path: Path) -> None:
    clone = tmp_path / "repository"
    shutil.copytree(
        ROOT,
        clone,
        ignore=shutil.ignore_patterns(
            ".pytest_cache",
            "__pycache__",
            "*.pyc",
            "results",
            "data",
        ),
    )
    target = clone / "reference/known_answers/objects/B01_aes_gcm.qsb"
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 1
    target.write_bytes(payload)
    errors = verify_known_answer_assets(clone)
    assert errors
    assert any(
        "B01_aes_gcm" in error or "known-answer file differs" in error
        for error in errors
    )


def test_reference_directory_contains_only_registered_files(
    known_answer_config: dict[str, object],
) -> None:
    reference_root = ROOT / "reference/known_answers"
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in reference_root.rglob("*")
        if path.is_file()
    }
    expected = {
        INPUT_RELATIVE_PATH,
        MANIFEST_RELATIVE_PATH,
        *(
            f"{OBJECT_DIRECTORY_RELATIVE_PATH}/{method_id}.qsb"
            for method_id in known_answer_config["method_order"]
        ),
    }
    assert actual == expected
