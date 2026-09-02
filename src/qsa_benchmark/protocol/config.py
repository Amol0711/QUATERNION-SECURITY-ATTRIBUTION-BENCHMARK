from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jsonschema

from qsa_benchmark.benchmark.registry import EXTENDED_METHOD_FACTORIES
from qsa_benchmark.benchmark.utils import canonical_json_bytes


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_config_path() -> Path:
    return repository_root() / "configs/protocol/experiment.json"


def default_schema_path() -> Path:
    return repository_root() / "configs/protocol/experiment.schema.json"


@dataclass(frozen=True)
class ProtocolConfig:
    path: Path
    payload: Mapping[str, Any]

    @property
    def protocol_id(self) -> str:
        return str(self.payload["protocol_id"])

    @property
    def methods(self) -> list[str]:
        return list(self.payload["methods"])

    @property
    def master_key(self) -> bytes:
        return bytes.fromhex(str(self.payload["master_key_hex"]))

    @property
    def master_seed(self) -> int:
        return int(self.payload["master_seed"])

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.payload)).hexdigest()


def _semantic_validate(payload: Mapping[str, Any]) -> None:
    methods = list(payload["methods"])
    expected = list(EXTENDED_METHOD_FACTORIES)
    if methods != expected:
        raise ValueError("method order must equal the extended registry order")
    method_set = set(methods)
    corpus = payload["corpus"]
    image_set = set(corpus["image_ids"])
    for panel in ("primary_panel", "ensemble_panel", "secondary_panel", "timing_panel"):
        if not set(corpus[panel]).issubset(image_set):
            raise ValueError(f"{panel} contains an unknown image identifier")
    perturbations = payload["perturbations"]
    if set(perturbations["primary"]).intersection(perturbations["secondary"]):
        raise ValueError("primary and secondary perturbation identifiers must be disjoint")
    for tier_id, tier in payload["execution_tiers"].items():
        tier_methods = methods if tier["methods"] == "all" else list(tier["methods"])
        if not set(tier_methods).issubset(method_set):
            raise ValueError(f"{tier_id} contains an unknown method")
        if tier["images"] not in corpus:
            raise ValueError(f"{tier_id} references an unknown corpus panel")
        tier_perturbations = tier["perturbations"]
        if isinstance(tier_perturbations, str) and tier_perturbations not in perturbations:
            raise ValueError(f"{tier_id} references an unknown perturbation panel")
    p1 = payload["protocols"]["P1_common_context"]
    p2 = payload["protocols"]["P2_fresh_randomness"]
    if not (p1["same_master_key"] and p1["same_nonce"] and p1["same_public_seed"] and p1["same_pregenerated_public_material"]):
        raise ValueError("P1 must hold all external context fields common")
    if not p2["same_master_key"] or p2["same_nonce"] or p2["same_public_seed"] or p2["same_pregenerated_public_material"]:
        raise ValueError("P2 must retain the key and refresh nonce, seed, and public material")
    if not payload["endpoint_policy"]["forbid_primary_scalar_attack_score"]:
        raise ValueError("the primary scalar attack score must remain disabled")
    if payload["object_model"]["evaluation_identifiers_in_object"]:
        raise ValueError("evaluation identifiers cannot be adversary-visible metadata")
    if int(payload["replication_policy"]["independent_runs"]) < 2:
        raise ValueError("at least two independent runs are required")


def load_protocol_config(path: str | Path | None = None, schema_path: str | Path | None = None) -> ProtocolConfig:
    source = Path(path) if path is not None else default_config_path()
    schema_source = Path(schema_path) if schema_path is not None else default_schema_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    schema = json.loads(schema_source.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(payload)
    _semantic_validate(payload)
    return ProtocolConfig(source.resolve(), payload)
