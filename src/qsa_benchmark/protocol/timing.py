from __future__ import annotations

import contextlib
import contextvars
import time
from dataclasses import asdict
from typing import Any, Iterator, Mapping

import numpy as np

from .models import StageTimingRecord
from .registry import POLICIES

FORWARD_STAGES = (
    "input_encode", "descriptor_compute", "geometry_forward", "transform_serialize",
    "primitive_protect", "envelope_serialize", "total_encrypt",
)
INVERSE_STAGES = (
    "envelope_parse", "primitive_unprotect", "transform_deserialize", "geometry_inverse",
    "release_check", "output_decode", "total_decrypt",
)
ALL_STAGES = FORWARD_STAGES + INVERSE_STAGES
_ACTIVE_STAGE: contextvars.ContextVar[str | None] = contextvars.ContextVar("qsa_active_stage", default=None)


class StageTrace:
    """Nonoverlapping component-stage recorder.

    Total-path measurements are intentionally collected in a separate pass and
    use ``total_encrypt`` or ``total_decrypt`` as the sole stage in that pass.
    This prevents stage sums from double-counting nested work.
    """

    def __init__(
        self,
        *,
        run_id: str,
        method_id: str,
        image_id: str,
        image_size: int,
        repetition: int,
        warmup: bool,
        memory_run: bool,
        measurement_pass: str,
    ) -> None:
        if measurement_pass not in {"component", "total"}:
            raise ValueError("measurement_pass must be component or total")
        self.run_id = run_id
        self.method_id = method_id
        self.image_id = image_id
        self.image_size = int(image_size)
        self.repetition = int(repetition)
        self.warmup = bool(warmup)
        self.memory_run = bool(memory_run)
        self.measurement_pass = measurement_pass
        self.records: list[StageTimingRecord] = []

    @contextlib.contextmanager
    def span(
        self,
        stage: str,
        direction: str,
        *,
        input_bytes: int = 0,
        output_bytes: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[None]:
        if stage not in ALL_STAGES:
            raise ValueError(f"unknown timing stage: {stage}")
        if direction not in {"encrypt", "decrypt"}:
            raise ValueError("direction must be encrypt or decrypt")
        is_total = stage in {"total_encrypt", "total_decrypt"}
        if (self.measurement_pass == "total") != is_total:
            raise ValueError("total and component stages must be measured in separate passes")
        if _ACTIVE_STAGE.get() is not None:
            raise RuntimeError("nested timing spans are forbidden")
        token = _ACTIVE_STAGE.set(stage)
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - start
            _ACTIVE_STAGE.reset(token)
            record_metadata = {"measurement_pass": self.measurement_pass, **dict(metadata or {})}
            self.records.append(StageTimingRecord(
                run_id=self.run_id,
                method_id=self.method_id,
                image_id=self.image_id,
                image_size=self.image_size,
                direction=direction,
                stage=stage,
                elapsed_ns=int(elapsed),
                input_bytes=int(input_bytes),
                output_bytes=int(output_bytes),
                repetition=self.repetition,
                warmup=self.warmup,
                memory_run=self.memory_run,
                metadata=record_metadata,
            ))

    def rows(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]


def applicable_stages(method_id: str, direction: str) -> tuple[str, ...]:
    path = POLICIES[method_id].timing_path
    if direction == "encrypt":
        common = ["input_encode", "envelope_serialize"]
        if path in {"geometry_only", "adaptive_geometry_only", "geometry_plus_primitive", "geometry_plus_custom_primitive"}:
            common += ["geometry_forward", "transform_serialize"]
        if path in {"adaptive_geometry_only", "geometry_plus_custom_primitive", "preview_plus_primitive"}:
            common += ["descriptor_compute"]
        if path not in {"geometry_only", "adaptive_geometry_only"}:
            common += ["primitive_protect"]
        return tuple(stage for stage in FORWARD_STAGES if stage in set(common))
    if direction == "decrypt":
        common = ["envelope_parse", "output_decode"]
        if path in {"geometry_only", "adaptive_geometry_only", "geometry_plus_primitive", "geometry_plus_custom_primitive"}:
            common += ["transform_deserialize", "geometry_inverse"]
        if path not in {"geometry_only", "adaptive_geometry_only"}:
            common += ["primitive_unprotect"]
        if path in {"primitive_plus_release_check", "primitive_only", "geometry_plus_primitive", "geometry_plus_custom_primitive", "preview_plus_primitive"} and bool(POLICIES[method_id].authenticated_coverage):
            common += ["release_check"]
        return tuple(stage for stage in INVERSE_STAGES if stage in set(common))
    raise ValueError("direction must be encrypt or decrypt")


def timing_plan_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timing = config["timing"]
    for method_id in config["methods"]:
        for size in timing["sizes"]:
            for direction in ("encrypt", "decrypt"):
                applicable = set(applicable_stages(method_id, direction))
                stages = FORWARD_STAGES if direction == "encrypt" else INVERSE_STAGES
                for stage in stages:
                    rows.append({
                        "method_id": method_id,
                        "image_size": int(size),
                        "direction": direction,
                        "stage": stage,
                        "applicable": stage in applicable or stage == f"total_{direction}",
                        "measurement_pass": "total" if stage.startswith("total_") else "component",
                        "warmups": int(timing["warmups"]),
                        "timed_repetitions": int(timing["timed_repetitions"]),
                        "memory_repetitions": int(timing["memory_repetitions"]),
                        "timing_path": POLICIES[method_id].timing_path,
                        "shared_setup_excluded": bool(timing["shared_setup_excluded"]),
                    })
    return rows


def validate_stage_rows(rows: list[dict[str, Any]], required_methods: set[str]) -> list[str]:
    errors: list[str] = []
    if not rows:
        return ["stage timing table is empty"]
    methods = {str(row.get("method_id")) for row in rows}
    if methods != required_methods:
        errors.append(f"method mismatch: missing={sorted(required_methods-methods)}, extra={sorted(methods-required_methods)}")
    for row in rows:
        stage = str(row.get("stage"))
        direction = str(row.get("direction"))
        if stage not in ALL_STAGES:
            errors.append(f"unknown stage {stage}")
        if direction not in {"encrypt", "decrypt"}:
            errors.append(f"invalid direction in {row.get('run_id')}")
        if int(row.get("elapsed_ns", -1)) < 0:
            errors.append(f"negative elapsed_ns in {row.get('run_id')}")
        if int(row.get("input_bytes", -1)) < 0 or int(row.get("output_bytes", -1)) < 0:
            errors.append(f"negative byte count in {row.get('run_id')}")
        metadata = row.get("metadata", {})
        measurement_pass = metadata.get("measurement_pass") if isinstance(metadata, Mapping) else None
        if stage.startswith("total_") and measurement_pass != "total":
            errors.append(f"total stage not recorded in total pass: {row.get('run_id')}")
        if not stage.startswith("total_") and measurement_pass != "component":
            errors.append(f"component stage not recorded in component pass: {row.get('run_id')}")
    return errors


def fit_serial_cost_model(samples: list[Mapping[str, float]]) -> dict[str, float]:
    if len(samples) < 4:
        raise ValueError("at least four timing samples are required")
    x = np.asarray([[1.0, float(row["pixel_count"]), float(row["protected_bytes"])] for row in samples], dtype=float)
    y = np.asarray([float(row["elapsed_ns"]) for row in samples], dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residual = y - fitted
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    n, p = len(y), x.shape[1]
    adjusted = 1.0 - (1.0 - r2) * (n - 1) / (n - p) if n > p else float("nan")
    relative = np.abs(residual) / np.maximum(np.abs(y), 1.0)
    return {
        "beta0_ns": float(beta[0]),
        "beta_geo_ns_per_pixel": float(beta[1]),
        "beta_primitive_ns_per_byte": float(beta[2]),
        "r2": r2,
        "adjusted_r2": adjusted,
        "maximum_relative_residual": float(np.max(relative)),
    }
