from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import psutil

T = TypeVar("T")


@dataclass(frozen=True)
class Measurement(Generic[T]):
    value: T
    elapsed_ns: int
    peak_tracemalloc_bytes: int
    rss_before_bytes: int
    rss_after_bytes: int


def measure(function: Callable[[], T]) -> Measurement[T]:
    """Measure wall time and process RSS without perturbing Python-heavy controls.

    Full allocation tracing is intentionally deferred to isolated systems runs,
    because tracemalloc changes the relative cost of byte-wise legacy baselines.
    The field is retained as zero in the development calibration table.
    """
    process = psutil.Process(os.getpid())
    rss_before = int(process.memory_info().rss)
    start = time.perf_counter_ns()
    value = function()
    elapsed = time.perf_counter_ns() - start
    rss_after = int(process.memory_info().rss)
    return Measurement(value, elapsed, 0, rss_before, rss_after)
