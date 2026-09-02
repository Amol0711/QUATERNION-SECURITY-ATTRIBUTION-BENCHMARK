from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .finite import apply_steps, center_image, restore_image, steps_from_rotation
from .models import RunContext, TransformOutput
from .plugins import TransformPlugin
from .quaternion import (
    caseii_schedule,
    curvature_frame,
    fixed_axis_rotation,
    proper_frame_from_pca,
    random_so3,
)
from .serialization import bytes_to_int32, int32_to_bytes


def _descriptor(kind: str, **kwargs: Any) -> dict[str, Any]:
    return {"kind": kind, "finite_format": "signed-int32-lifting-v1", **kwargs}


class IdentityTransform(TransformPlugin):
    component_id = "identity"

    def forward(self, image: np.ndarray, context: RunContext) -> TransformOutput:
        values = center_image(image)
        return TransformOutput(int32_to_bytes(values), _descriptor("identity"), tuple(image.shape))

    def inverse(self, output: TransformOutput, context: RunContext) -> np.ndarray:
        return restore_image(bytes_to_int32(output.payload, output.shape))


class ReversibleColorTransform(TransformPlugin):
    component_id = "rct"

    def forward(self, image: np.ndarray, context: RunContext) -> TransformOutput:
        x = np.asarray(image, dtype=np.int64)
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        y = (r + 2 * g + b) // 4
        u = b - g
        v = r - g
        values = np.stack([y - 128, u, v], axis=-1)
        return TransformOutput(int32_to_bytes(values), _descriptor("rct"), tuple(image.shape))

    def inverse(self, output: TransformOutput, context: RunContext) -> np.ndarray:
        z = bytes_to_int32(output.payload, output.shape)
        y, u, v = z[..., 0] + 128, z[..., 1], z[..., 2]
        g = y - ((u + v) // 4)
        r = v + g
        b = u + g
        values = np.stack([r, g, b], axis=-1)
        return np.clip(values, 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class StaticLiftingTransform(TransformPlugin):
    component_id: str
    kind: str

    def target_matrix(self, image: np.ndarray, context: RunContext) -> np.ndarray:
        raise NotImplementedError

    def forward(self, image: np.ndarray, context: RunContext) -> TransformOutput:
        matrix = self.target_matrix(image, context)
        steps = steps_from_rotation(matrix)
        values = apply_steps(center_image(image), steps)
        descriptor = _descriptor(self.kind, steps=steps)
        return TransformOutput(int32_to_bytes(values), descriptor, tuple(image.shape))

    def inverse(self, output: TransformOutput, context: RunContext) -> np.ndarray:
        values = bytes_to_int32(output.payload, output.shape)
        restored = apply_steps(values, output.descriptor["steps"], inverse=True)
        return restore_image(restored)


class PCATransform(StaticLiftingTransform):
    def __init__(self) -> None:
        super().__init__("pca-klt", "pca-klt")

    def target_matrix(self, image: np.ndarray, context: RunContext) -> np.ndarray:
        return proper_frame_from_pca(image).T


class RandomSO3Transform(StaticLiftingTransform):
    def __init__(self) -> None:
        super().__init__("random-so3", "random-so3")

    def target_matrix(self, image: np.ndarray, context: RunContext) -> np.ndarray:
        return random_so3(context.seed)


class FixedAxisQuaternionTransform(StaticLiftingTransform):
    def __init__(self) -> None:
        super().__init__("fixed-axis-quaternion", "fixed-axis-quaternion")

    def target_matrix(self, image: np.ndarray, context: RunContext) -> np.ndarray:
        return fixed_axis_rotation()


class CaseIITransform(TransformPlugin):
    component_id = "caseii"

    def __init__(self, curvature_oriented: bool = False, scale: float = 0.55, states: int = 8) -> None:
        self.curvature_oriented = curvature_oriented
        self.scale = float(scale)
        self.states = int(states)
        self.component_id = "caseii-curvature" if curvature_oriented else "caseii"

    def _frame_steps(self, image: np.ndarray) -> list[dict[str, int]]:
        if not self.curvature_oriented:
            return []
        return steps_from_rotation(curvature_frame(image, self.scale, self.states).T)

    def forward(self, image: np.ndarray, context: RunContext) -> TransformOutput:
        shape = tuple(image.shape)
        flat = center_image(image).reshape(-1, 3)
        frame_steps = self._frame_steps(image)
        if frame_steps:
            flat = apply_steps(flat, frame_steps)
        schedule_steps = [steps_from_rotation(matrix) for matrix in caseii_schedule(len(flat), self.scale, self.states)]
        out = flat.copy()
        for state, steps in enumerate(schedule_steps):
            indices = np.arange(state, len(flat), self.states)
            if len(indices):
                out[indices] = apply_steps(out[indices], steps)
        descriptor = _descriptor(
            "curvature-caseii" if self.curvature_oriented else "caseii",
            scale=self.scale,
            states=self.states,
            frame_steps=frame_steps,
            schedule_steps=schedule_steps,
        )
        return TransformOutput(int32_to_bytes(out.reshape(shape)), descriptor, shape)

    def inverse(self, output: TransformOutput, context: RunContext) -> np.ndarray:
        shape = output.shape
        flat = bytes_to_int32(output.payload, shape).reshape(-1, 3)
        states = int(output.descriptor["states"])
        schedule_steps = output.descriptor["schedule_steps"]
        out = flat.copy()
        for state in reversed(range(states)):
            indices = np.arange(state, len(flat), states)
            if len(indices):
                out[indices] = apply_steps(out[indices], schedule_steps[state], inverse=True)
        frame_steps = output.descriptor.get("frame_steps", [])
        if frame_steps:
            out = apply_steps(out, frame_steps, inverse=True)
        return restore_image(out.reshape(shape))


def transform_from_descriptor(descriptor: dict[str, Any]) -> TransformPlugin:
    kind = descriptor.get("kind")
    if kind == "identity":
        return IdentityTransform()
    if kind == "rct":
        return ReversibleColorTransform()
    if kind == "pca-klt":
        return PCATransform()
    if kind == "random-so3":
        return RandomSO3Transform()
    if kind == "fixed-axis-quaternion":
        return FixedAxisQuaternionTransform()
    if kind == "caseii":
        return CaseIITransform(False, descriptor.get("scale", 0.55), descriptor.get("states", 8))
    if kind == "curvature-caseii":
        return CaseIITransform(True, descriptor.get("scale", 0.55), descriptor.get("states", 8))
    raise ValueError(f"unsupported transform descriptor: {kind}")
