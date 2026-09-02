from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def deterministic_seed(*values: object) -> int:
    h = hashlib.sha256(b"QSA-BENCHMARK-SEED-V1")
    for value in values:
        blob = str(value).encode("utf-8")
        h.update(len(blob).to_bytes(4, "big"))
        h.update(blob)
    return int.from_bytes(h.digest()[:8], "big")


def derive_key(master_key: bytes, label: str, length: int = 32) -> bytes:
    h = hashlib.shake_256()
    h.update(b"QSA-BENCHMARK-KDF-V1")
    h.update(len(label).to_bytes(4, "big"))
    h.update(label.encode("utf-8"))
    h.update(len(master_key).to_bytes(4, "big"))
    h.update(master_key)
    return h.digest(length)


def environment_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
