from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def regenerated_registry_files() -> dict[str, bytes]:
    from qsa_benchmark.protocol.registry_export import build_registry_files

    return build_registry_files(ROOT)


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return ROOT
