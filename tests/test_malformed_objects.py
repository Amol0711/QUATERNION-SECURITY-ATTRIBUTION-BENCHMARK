from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qsa_benchmark.benchmark.envelope import EnvelopeFormatError, parse_envelope
from qsa_benchmark.validation.malformed import (
    BASE_OBJECT_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    VECTOR_DIRECTORY_RELATIVE_PATH,
    build_malformed_object_files,
    check_malformed_object_files,
    load_malformed_object_manifest,
    malformed_object_summary,
    verify_malformed_object_assets,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def malformed_manifest() -> dict[str, object]:
    return load_malformed_object_manifest(ROOT)


def test_malformed_reference_set_matches_executable_generator(
    malformed_manifest: dict[str, object],
) -> None:
    expected = build_malformed_object_files(ROOT)
    assert len(expected) == malformed_manifest["case_count"] + 1
    for relative, payload in expected.items():
        assert (ROOT / relative).read_bytes() == payload
    assert check_malformed_object_files(ROOT) == []
    assert verify_malformed_object_assets(ROOT) == []


def test_every_malformed_vector_has_the_registered_parser_outcome(
    malformed_manifest: dict[str, object],
) -> None:
    cases = malformed_manifest["cases"]
    assert malformed_manifest["case_count"] == len(cases) == 33
    assert [case["ordinal"] for case in cases] == list(range(1, 34))
    assert len({case["case_id"] for case in cases}) == 33
    assert len({case["sha256"] for case in cases}) == 33
    for case in cases:
        payload = (ROOT / case["object_path"]).read_bytes()
        assert len(payload) == case["object_bytes"]
        assert hashlib.sha256(payload).hexdigest() == case["sha256"]
        with pytest.raises(EnvelopeFormatError) as caught:
            parse_envelope(payload)
        assert caught.value.code == case["expected_error_code"]


def test_malformed_vectors_cover_prefix_json_semantics_canonicality_and_lengths(
    malformed_manifest: dict[str, object],
) -> None:
    categories = {case["category"] for case in malformed_manifest["cases"]}
    assert categories == {
        "fixed_prefix", "header_framing", "header_json", "header_semantics",
        "canonical_encoding", "component_lengths", "component_framing",
    }
    codes = set(malformed_manifest["parser_contract"]["stable_error_codes"])
    assert {
        "truncated_prefix", "invalid_magic", "unsupported_version",
        "invalid_header_length", "invalid_header_encoding", "invalid_header_json",
        "duplicate_header_field", "invalid_header_fields",
        "noncanonical_header_encoding", "invalid_method_id", "invalid_descriptor",
        "invalid_metadata", "invalid_image_shape", "invalid_component_length",
        "envelope_length_mismatch",
    } == codes


def test_valid_base_object_remains_canonical(malformed_manifest: dict[str, object]) -> None:
    base = (ROOT / BASE_OBJECT_RELATIVE_PATH).read_bytes()
    parsed = parse_envelope(base)
    assert parsed.header["method_id"] == "B20_full_aead_explicit_preview"
    assert malformed_manifest["base_object"]["sha256"] == hashlib.sha256(base).hexdigest()


def test_parser_rejects_nonbytes_with_a_stable_code() -> None:
    with pytest.raises(EnvelopeFormatError) as caught:
        parse_envelope(bytearray(b"QSB1"))  # type: ignore[arg-type]
    assert caught.value.code == "invalid_input_type"


def test_malformed_reference_directory_contains_only_registered_files(
    malformed_manifest: dict[str, object],
) -> None:
    root = ROOT / "reference/malformed_objects"
    actual = {path.relative_to(ROOT).as_posix() for path in root.rglob("*") if path.is_file()}
    expected = {MANIFEST_RELATIVE_PATH, *(case["object_path"] for case in malformed_manifest["cases"])}
    assert actual == expected
    assert all(path.startswith(VECTOR_DIRECTORY_RELATIVE_PATH + "/") for path in expected if path != MANIFEST_RELATIVE_PATH)


def test_malformed_summary_is_complete(malformed_manifest: dict[str, object]) -> None:
    summary = malformed_object_summary(ROOT)
    assert summary["case_count"] == 33
    assert sum(summary["category_counts"].values()) == 33
    assert summary["stable_error_code_count"] == 15
    assert summary["vector_set_sha256"] == malformed_manifest["vector_set_sha256"]
