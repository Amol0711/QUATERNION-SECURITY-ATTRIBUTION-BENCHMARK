from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from qsa_benchmark.benchmark.envelope import EnvelopeFormatError
from qsa_benchmark.benchmark.registry import EXTENDED_METHOD_FACTORIES
from qsa_benchmark.protocol.experiment import _reported_attack_error_type
from qsa_benchmark.protocol.registry_export import ACTIVE_MODIFICATION_FIELDS
from qsa_benchmark.validation.active_modification import (
    active_modification_counts,
    active_modification_summary,
    build_active_modification_rows,
    load_active_modification_config,
    load_active_input_pair,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "registries/active_modification_schedule.csv"


def _rows() -> list[dict[str, str]]:
    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_active_modification_registry_matches_executable_schedule(
    regenerated_registry_files: dict[str, bytes],
) -> None:
    expected = regenerated_registry_files["registries/active_modification_schedule.csv"]
    assert REGISTRY.read_bytes() == expected
    rows = list(csv.DictReader(io.StringIO(expected.decode("utf-8"), newline="")))
    assert tuple(rows[0]) == ACTIVE_MODIFICATION_FIELDS
    assert len(rows) == 180


def test_active_modification_count_partition_is_exact() -> None:
    rows = _rows()
    authenticated = [row for row in rows if row["authenticated"] == "true"]
    unauthenticated = [row for row in rows if row["authenticated"] == "false"]
    assert len(authenticated) == 73
    assert all(row["expected_accepted"] == "false" for row in authenticated)
    assert len(unauthenticated) == 107
    assert sum(row["expected_accepted"] == "true" for row in unauthenticated) == 77
    assert sum(row["expected_accepted"] == "false" for row in unauthenticated) == 30
    assert sum(row["expected_plaintext_changed"] == "true" for row in unauthenticated) == 37
    assert sum(row["expected_plaintext_changed"] == "false" for row in unauthenticated) == 40
    assert sum(row["parser_level"] == "true" for row in rows) == 48


def test_active_modification_method_order_and_denominators_are_exact() -> None:
    rows = _rows()
    methods = list(EXTENDED_METHOD_FACTORIES)
    assert list(dict.fromkeys(row["method_id"] for row in rows)) == methods
    counts = Counter(row["method_id"] for row in rows)
    authenticated = {
        "B01_aes_gcm", "B02_chacha20_poly1305", "B03_shake_hmac",
        "B15_geometry_shake_hmac", "B16_geometry_aes_gcm",
        "B17_tip_r0_emulation", "B20_full_aead_explicit_preview",
        "B23_secure_fixed_header", "B24_aes_gcm_siv",
    }
    for method_id in methods:
        if method_id == "B20_full_aead_explicit_preview":
            assert counts[method_id] == 9
        elif method_id in authenticated:
            assert counts[method_id] == 8
        elif method_id in {"B21_public_fresh_pad", "B22_public_wideblock_prp"}:
            assert counts[method_id] == 8
        else:
            assert counts[method_id] == 7


def test_active_modification_outcome_classes_and_failure_stages_are_separated() -> None:
    rows = _rows()
    for row in rows:
        if row["parser_level"] == "true":
            assert row["expected_outcome"] == "parser_reject"
            assert row["failure_stage"] == "parser"
            assert row["expected_exception_class"] == "EnvelopeFormatError"
            assert row["expected_parser_code"] == "envelope_length_mismatch"
        elif row["authenticated"] == "true":
            assert row["expected_outcome"] == "integrity_reject"
            assert row["failure_stage"] == "integrity_or_release"
            assert row["expected_parser_code"] == ""
        else:
            assert row["expected_outcome"] in {
                "accepted_plaintext_changed", "accepted_plaintext_unchanged"
            }
            assert row["failure_stage"] == "none"
            assert row["expected_exception_class"] == ""


def test_active_modification_objects_are_unique_and_byte_distinct() -> None:
    rows = _rows()
    assert [int(row["case_ordinal"]) for row in rows] == list(range(1, 181))
    assert len({row["case_id"] for row in rows}) == 180
    assert len({row["mutated_object_sha256"] for row in rows}) == 180
    assert all(row["byte_distinct"] == "true" for row in rows)
    assert all(int(row["byte_difference_count"]) > 0 for row in rows)


def test_active_input_pair_and_summary_are_frozen() -> None:
    config = load_active_modification_config(ROOT)
    base, donor = load_active_input_pair(ROOT)
    assert hashlib.sha256(base.tobytes(order="C")).hexdigest() == config["input_pair"]["base_sha256"]
    assert hashlib.sha256(donor.tobytes(order="C")).hexdigest() == config["input_pair"]["donor_sha256"]
    summary = active_modification_summary(ROOT)
    assert summary["counts"] == config["expected_counts"]
    assert summary["method_count"] == 24
    assert summary["ordered_case_digest"] == "ef828d3e9d170581fa1c1183c76169c15d98bc66c05ccc68eecfb54b4462a40b"


def test_protocol_registry_exposes_the_active_modification_contract() -> None:
    protocol = json.loads((ROOT / "registries/protocol_registry.json").read_text(encoding="utf-8"))
    policy = protocol["active_modification_policy"]
    assert policy["schedule_id"] == "QSA-ACTIVE-MODIFICATION-SCHEDULE-V1"
    assert policy["registry_path"] == "registries/active_modification_schedule.csv"
    assert policy["parser_level_mutations"] == ["trailing_byte_append", "one_byte_truncation"]
    assert protocol["derived_counts"]["total_cases"] == 180
    assert protocol["derived_counts"]["authenticated_rejections"] == 73


def test_direct_active_modification_generation_matches_registered_counts() -> None:
    rows = build_active_modification_rows(ROOT)
    assert active_modification_counts(rows) == {
        "total_cases": 180,
        "authenticated_cases": 73,
        "authenticated_rejections": 73,
        "unauthenticated_cases": 107,
        "unauthenticated_acceptances": 77,
        "unauthenticated_rejections": 30,
        "unauthenticated_plaintext_changed": 37,
        "unauthenticated_plaintext_unchanged": 40,
        "parser_length_violations": 48,
    }


def test_public_attack_table_preserves_historical_valueerror_label() -> None:
    assert _reported_attack_error_type(EnvelopeFormatError("invalid_magic", "invalid")) == "ValueError"
    assert _reported_attack_error_type(ValueError("invalid")) == "ValueError"
    assert _reported_attack_error_type(RuntimeError("invalid")) == "RuntimeError"
