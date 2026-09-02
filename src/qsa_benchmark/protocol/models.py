from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from qsa_benchmark.benchmark.models import RunContext


class DifferentialProtocol(str, Enum):
    P1_COMMON_CONTEXT = "P1_common_context"
    P2_FRESH_RANDOMNESS = "P2_fresh_randomness"


@dataclass(frozen=True)
class ContextPair:
    protocol: DifferentialProtocol
    left: RunContext
    right: RunContext
    same_master_key: bool
    same_nonce: bool
    same_seed: bool
    same_public_material: bool
    operation_regime: str


@dataclass(frozen=True)
class MethodPolicy:
    method_id: str
    metric_body_source: str
    body_metric_domain: str
    deterministic_plaintext_leakage: tuple[str, ...]
    public_randomness: tuple[str, ...]
    public_recovery_material: tuple[str, ...]
    authenticated_coverage: tuple[str, ...]
    permitted_functionality: tuple[str, ...]
    leakage_equivalence_rule: str
    prechallenge_entropy_rule: str
    descriptor_entropy_rule: str
    post_object_recovery: str
    p1_semantics: str
    p2_semantics: str
    p1_correct_use: bool
    p2_applicable: bool
    publicly_invertible: bool
    nonce_length: int
    common_map_class: str
    timing_path: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        for key, value in tuple(row.items()):
            if isinstance(value, tuple):
                row[key] = "|".join(value)
        return row


@dataclass(frozen=True)
class PerturbationSpec:
    perturbation_id: str
    family: str
    location: str
    channels: tuple[int, ...]
    magnitude: int
    expected_pixel_count: int | None
    inferential_role: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["channels"] = "|".join(str(v) for v in self.channels)
        return row


@dataclass(frozen=True)
class DifferentialDecision:
    sample_count: int
    npcr: float
    uaci: float
    npcr_p_value: float
    uaci_p_value: float
    npcr_reject: bool
    uaci_reject: bool
    npcr_lower_count: int
    uaci_lower: float
    uaci_upper: float


@dataclass(frozen=True)
class StageTimingRecord:
    run_id: str
    method_id: str
    image_id: str
    image_size: int
    direction: str
    stage: str
    elapsed_ns: int
    input_bytes: int
    output_bytes: int
    repetition: int
    warmup: bool
    memory_run: bool
    metadata: Mapping[str, Any]
