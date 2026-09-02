from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .models import ConstructionOutput, RunContext, TransformOutput


class TransformPlugin(ABC):
    component_id: str

    @abstractmethod
    def forward(self, image: np.ndarray, context: RunContext) -> TransformOutput: ...

    @abstractmethod
    def inverse(self, output: TransformOutput, context: RunContext) -> np.ndarray: ...


class MaskPlugin(ABC):
    component_id: str

    @abstractmethod
    def apply(self, payload: bytes, context: RunContext, descriptor: dict[str, Any]) -> bytes: ...


class PermutationPlugin(ABC):
    component_id: str

    @abstractmethod
    def permute(self, payload: bytes, context: RunContext) -> tuple[bytes, dict[str, Any]]: ...

    @abstractmethod
    def invert(self, payload: bytes, descriptor: dict[str, Any], context: RunContext) -> bytes: ...


class DiffusionPlugin(ABC):
    component_id: str

    @abstractmethod
    def diffuse(self, payload: bytes, context: RunContext) -> bytes: ...

    @abstractmethod
    def invert(self, payload: bytes, context: RunContext) -> bytes: ...


class AuthenticationPlugin(ABC):
    component_id: str

    @abstractmethod
    def tag(self, data: bytes, context: RunContext) -> bytes: ...

    @abstractmethod
    def verify(self, data: bytes, tag: bytes, context: RunContext) -> None: ...


class SerializationPlugin(ABC):
    component_id: str


class ConstructionPlugin(ABC):
    method_id: str
    display_name: str
    family: str
    benchmark_role: str
    authenticated: bool
    secure_control: bool
    exact: bool = True
    component_ids: tuple[str, ...]

    @abstractmethod
    def encrypt(self, image: np.ndarray, context: RunContext) -> ConstructionOutput: ...

    @abstractmethod
    def decrypt(self, object_bytes: bytes, context: RunContext) -> np.ndarray: ...
