from __future__ import annotations

from pathlib import Path

from qsa_benchmark.benchmark.registry import EXTENDED_METHOD_FACTORIES, method_registry
from qsa_benchmark.protocol.config import load_protocol_config
from qsa_benchmark.protocol.registry import protocol_method_registry
from qsa_benchmark.protocol.targeted_validation import load_target_config


def test_registry_cardinality_and_order() -> None:
    assert len(method_registry()) == 20
    assert len(EXTENDED_METHOD_FACTORIES) == 24
    assert [row["method_id"] for row in protocol_method_registry()] == list(EXTENDED_METHOD_FACTORIES)


def test_protocol_config_and_target_linkage() -> None:
    config = load_protocol_config()
    root = Path(__file__).resolve().parents[1]
    target, _ = load_target_config(root)
    assert config.protocol_id == "QSA-PROTOCOL-V1"
    assert target["parent_protocol_id"] == config.protocol_id
    assert target["parent_protocol_sha256"] == config.sha256
