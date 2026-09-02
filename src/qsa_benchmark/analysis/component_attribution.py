"""Auxiliary component-attribution arithmetic; not cryptography."""
from __future__ import annotations

def component_factorial_contrasts(success_00: float, success_10: float, success_01: float, success_11: float) -> dict[str, float]:
    values = (success_00, success_10, success_01, success_11)
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("success probabilities must lie in [0,1]")
    return {
        "transform_main_effect": 0.5 * ((success_10-success_00)+(success_11-success_01)),
        "primitive_main_effect": 0.5 * ((success_01-success_00)+(success_11-success_10)),
        "interaction": success_11-success_10-success_01+success_00,
        "transform_given_primitive": success_11-success_01,
        "primitive_given_transform": success_11-success_10,
    }
