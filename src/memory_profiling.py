"""Memory and wall-time measurement for the data-handling deliverable.

The assignment requires evidence of memory usage before and after optimisation.
Peak resident set size is the figure that matters here: the naive pandas path
materialises a whole day of CDR records as Python objects, and it is the *peak*
during parsing -- not the steady-state size of the resulting frame -- that
decides whether the pipeline survives on a constrained machine.

RSS is sampled on a background thread rather than read once at the end, because
the peak occurs mid-parse and is gone by the time the context manager exits.

``tracemalloc`` is reported alongside RSS but is deliberately optional. It only
sees allocations made through Python's allocator, so for Polars -- which holds
its data in Arrow buffers allocated by Rust -- it materially understates usage.
The two numbers together make that distinction visible, which is itself part of
the story the report needs to tell.
"""

from __future__ import annotations

import gc
import os
import threading
import time
import tracemalloc
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import psutil

__all__ = ["MemoryReport", "measure", "rss_bytes", "deep_size_bytes", "format_bytes"]

_SAMPLE_INTERVAL_S = 0.05


def rss_bytes() -> int:
    """Current resident set size of this process, in bytes."""
    return psutil.Process(os.getpid()).memory_info().rss


def format_bytes(n: float) -> str:
    """Render a byte count as a human-readable string."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def deep_size_bytes(obj: Any) -> int | None:
    """Best-effort in-memory footprint of a dataframe or array.

    Returns None when the object type is not recognised, so callers can record
    a null rather than a misleading zero.
    """
    # pandas
    memory_usage = getattr(obj, "memory_usage", None)
    if callable(memory_usage):
        try:
            return int(memory_usage(deep=True).sum())
        except TypeError:
            pass
    # polars
    estimated = getattr(obj, "estimated_size", None)
    if callable(estimated):
        try:
            return int(estimated())
        except TypeError:
            pass
    # numpy
    nbytes = getattr(obj, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    return None


@dataclass
class MemoryReport:
    """Measurements captured across one :func:`measure` block."""

    label: str
    wall_s: float = 0.0
    rss_before: int = 0
    rss_after: int = 0
    rss_peak: int = 0
    tracemalloc_peak: int | None = None
    n_samples: int = 0
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def rss_delta(self) -> int:
        """Net RSS change across the block."""
        return self.rss_after - self.rss_before

    @property
    def rss_peak_delta(self) -> int:
        """Peak RSS above the level at block entry.

        This is the figure to quote when comparing strategies: it isolates the
        memory the strategy itself demanded from whatever the process was
        already holding.
        """
        return self.rss_peak - self.rss_before

    def to_row(self) -> dict[str, Any]:
        """Flatten to a dict suitable for a CSV row."""
        row: dict[str, Any] = {
            "label": self.label,
            "wall_s": round(self.wall_s, 4),
            "rss_before_bytes": self.rss_before,
            "rss_after_bytes": self.rss_after,
            "rss_peak_bytes": self.rss_peak,
            "rss_peak_delta_bytes": self.rss_peak_delta,
            "rss_delta_bytes": self.rss_delta,
            "tracemalloc_peak_bytes": self.tracemalloc_peak,
            "n_samples": self.n_samples,
            "notes": self.notes,
        }
        row.update(self.extra)
        return row

    def summary(self) -> str:
        """One-line human-readable summary for logs and notebooks."""
        tm = format_bytes(self.tracemalloc_peak) if self.tracemalloc_peak is not None else "n/a"
        return (
            f"[{self.label}] wall={self.wall_s:.2f}s "
            f"peak_rss={format_bytes(self.rss_peak)} "
            f"(+{format_bytes(self.rss_peak_delta)} over entry) "
            f"tracemalloc_peak={tm}"
        )


class _RSSSampler(threading.Thread):
    """Polls process RSS on a daemon thread to capture the true peak."""

    def __init__(self, interval_s: float = _SAMPLE_INTERVAL_S) -> None:
        super().__init__(daemon=True)
        self._interval = interval_s
        # Named _stop_event, not _stop: threading.Thread has a private _stop()
        # method that join() calls, and shadowing it breaks joining the thread.
        self._stop_event = threading.Event()
        self._process = psutil.Process(os.getpid())
        self.peak = 0
        self.n_samples = 0

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                current = self._process.memory_info().rss
            except psutil.Error:  # process info briefly unavailable
                break
            self.peak = max(self.peak, current)
            self.n_samples += 1
            self._stop_event.wait(self._interval)

    def stop(self) -> None:
        """Signal the sampler to finish and wait for it to exit."""
        self._stop_event.set()
        self.join(timeout=2.0)


@contextmanager
def measure(
    label: str,
    *,
    trace_python_allocs: bool = True,
    collect_first: bool = True,
    notes: str = "",
) -> Iterator[MemoryReport]:
    """Measure peak RSS, wall time and Python allocation peak across a block.

    The yielded report is populated on exit, so read its fields after the
    ``with`` block rather than inside it.

    Args:
        label: Name of the strategy being measured, used as the CSV key.
        trace_python_allocs: Enable ``tracemalloc``. Adds substantial overhead
            and undercounts non-Python allocations, so disable it when timing
            Polars or when the wall-clock figure is the one being reported.
        collect_first: Run a full ``gc.collect()`` before the baseline reading
            so memory from earlier work is not attributed to this block.
        notes: Free text carried through to the CSV row.

    Yields:
        A :class:`MemoryReport` filled in once the block completes.

    Example:
        >>> with measure("polars_lazy") as report:  # doctest: +SKIP
        ...     df = ingest_day(path)
        >>> print(report.summary())  # doctest: +SKIP
    """
    if collect_first:
        gc.collect()

    report = MemoryReport(label=label, notes=notes)
    report.rss_before = rss_bytes()
    report.rss_peak = report.rss_before

    started_tracemalloc = False
    if trace_python_allocs and not tracemalloc.is_tracing():
        tracemalloc.start()
        started_tracemalloc = True

    sampler = _RSSSampler()
    sampler.start()
    t0 = time.perf_counter()
    try:
        yield report
    finally:
        report.wall_s = time.perf_counter() - t0
        sampler.stop()
        report.rss_after = rss_bytes()
        # The sampler may have missed a spike between its last poll and here.
        report.rss_peak = max(sampler.peak, report.rss_before, report.rss_after)
        report.n_samples = sampler.n_samples

        if tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            report.tracemalloc_peak = peak
            if started_tracemalloc:
                tracemalloc.stop()
