from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class RunContext:
    master_key: bytes
    nonce: bytes
    seed: int
    image_id: str
    method_id: str
    run_id: str
    # Experiment provenance is separated from the adversary-visible object.
    # None selects the core metadata view; extended experiments pass a sanitized mapping.
    public_metadata: Mapping[str, Any] | None = None
    # Public controls receive fixed public randomness through this field.
    public_material: bytes = b""
    protocol_id: str = "core-benchmark-v1"


@dataclass(frozen=True)
class EnvelopeParts:
    method_id: str
    image_shape: tuple[int, int, int]
    descriptor: dict[str, Any]
    metadata: dict[str, Any]
    nonce: bytes
    public_payload: bytes
    protected_payload: bytes
    tag: bytes


@dataclass(frozen=True)
class ParsedEnvelope:
    header: dict[str, Any]
    header_bytes: bytes
    nonce: bytes
    public_payload: bytes
    protected_payload: bytes
    tag: bytes


@dataclass(frozen=True)
class ConstructionOutput:
    object_bytes: bytes
    ciphertext_view: bytes


@dataclass(frozen=True)
class TransformOutput:
    payload: bytes
    descriptor: dict[str, Any]
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class DatasetRecord:
    image_id: str
    corpus: str
    split: str
    source: str
    path: str
    width: int
    height: int
    semantic_label: str
    license: str
    sha256: str


ImageArray = np.ndarray
