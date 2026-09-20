"""Timing measurement and environment capture.

Assignment requirement IV asks for exact training and execution times together
with the hardware they were recorded on. Two things make such numbers
defensible rather than decorative:

1. Repeats. A single wall-clock reading on a shared or thermally throttled
   machine is noise. Every timing here is a mean over repeats with a standard
   deviation reported alongside it.
2. A recorded environment. :func:`record_hardware` snapshots CPU, RAM, GPU,
   OS and library versions to ``results/environment.json`` so the timings in
   the report can be attributed to a specific machine.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import psutil

__all__ = ["TimingResult", "time_it", "record_hardware", "collect_environment", "Stopwatch"]

T = TypeVar("T")

# Libraries whose versions materially affect results or timings.
_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "scipy",
    "statsmodels",
    "scikit-learn",
    "lightgbm",
    "torch",
    "optuna",
    "matplotlib",
)


@dataclass
class TimingResult:
    """Wall-time statistics over repeated executions."""

    label: str
    repeats: int
    times_s: list[float] = field(default_factory=list)

    @property
    def mean_s(self) -> float:
        return statistics.fmean(self.times_s) if self.times_s else float("nan")

    @property
    def std_s(self) -> float:
        """Sample standard deviation; 0.0 when there is only one measurement."""
        return statistics.stdev(self.times_s) if len(self.times_s) > 1 else 0.0

    @property
    def min_s(self) -> float:
        return min(self.times_s) if self.times_s else float("nan")

    @property
    def max_s(self) -> float:
        return max(self.times_s) if self.times_s else float("nan")

    def to_row(self) -> dict[str, Any]:
        """Flatten to a dict suitable for a CSV row."""
        return {
            "label": self.label,
            "repeats": self.repeats,
            "mean_s": round(self.mean_s, 6),
            "std_s": round(self.std_s, 6),
            "min_s": round(self.min_s, 6),
            "max_s": round(self.max_s, 6),
            "times_s": json.dumps([round(t, 6) for t in self.times_s]),
        }

    def __str__(self) -> str:
        return f"{self.label}: {self.mean_s:.4f} s +/- {self.std_s:.4f} (n={self.repeats})"


class Stopwatch:
    """Context manager recording the wall time of a single block.

    Example:
        >>> with Stopwatch("inference") as sw:  # doctest: +SKIP
        ...     model.walk_forward(series, start, end)
        >>> sw.elapsed_s  # doctest: +SKIP
    """

    def __init__(self, label: str = "block") -> None:
        self.label = label
        self.elapsed_s: float = float("nan")
        self._t0: float = 0.0

    def __enter__(self) -> Stopwatch:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_s = time.perf_counter() - self._t0


def time_it(
    fn: Callable[[], T],
    *,
    label: str = "",
    repeats: int = 3,
    warmup: int = 0,
) -> tuple[TimingResult, T]:
    """Execute ``fn`` repeatedly and return timing statistics with its result.

    Args:
        fn: Zero-argument callable to time. Bind arguments with a lambda or
            ``functools.partial``.
        label: Name used in the timing table.
        repeats: Number of timed executions. Must be at least 1.
        warmup: Untimed executions run first, to exclude one-off costs such as
            lazy imports, JIT warmup or CUDA context creation from the reported
            figure. Warmup runs are excluded from the statistics.

    Returns:
        A ``(TimingResult, last_return_value)`` pair. The return value comes
        from the final timed execution.

    Raises:
        ValueError: If ``repeats`` is less than 1.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    for _ in range(warmup):
        fn()

    times: list[float] = []
    result: Any = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)

    return TimingResult(
        label=label or getattr(fn, "__name__", "fn"), repeats=repeats, times_s=times
    ), result


# --------------------------------------------------------------------------
# Environment capture
# --------------------------------------------------------------------------


def _package_versions() -> dict[str, str | None]:
    """Installed versions of the libraries this study depends on."""
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _cpu_info() -> dict[str, Any]:
    """CPU model, core counts and clock, as far as the platform exposes them."""
    freq = None
    try:
        f = psutil.cpu_freq()
        freq = round(f.max or f.current, 1) if f else None
    except (NotImplementedError, AttributeError, OSError):
        pass
    return {
        "processor": platform.processor() or platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "max_freq_mhz": freq,
    }


def _gpu_info() -> dict[str, Any]:
    """CUDA device details when torch is present and a GPU is visible."""
    info: dict[str, Any] = {"available": False, "devices": []}
    try:
        import torch
    except ImportError:
        info["note"] = "torch not installed"
        return info

    info["torch_version"] = torch.__version__
    info["cuda_compiled_version"] = torch.version.cuda
    if not torch.cuda.is_available():
        info["note"] = "no CUDA device visible; training runs on CPU"
        return info

    info["available"] = True
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        info["devices"].append(
            {
                "index": idx,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
                "capability": f"{props.major}.{props.minor}",
            }
        )
    return info


def collect_environment() -> dict[str, Any]:
    """Snapshot the hardware and software this run executed on."""
    vm = psutil.virtual_memory()
    return {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "node": platform.node(),
        },
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "cpu": _cpu_info(),
        "memory": {
            "total_bytes": vm.total,
            "available_bytes_at_capture": vm.available,
        },
        "gpu": _gpu_info(),
        "packages": _package_versions(),
        "runtime": {
            "is_kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
            or Path("/kaggle/input").exists(),
            "kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
            "cwd": str(Path.cwd()),
        },
    }


def record_hardware(destination: str | Path) -> dict[str, Any]:
    """Write the environment snapshot to ``destination`` as JSON.

    Args:
        destination: Path to ``environment.json``. Parent directories are
            created if absent.

    Returns:
        The snapshot that was written, so callers can log or display it.
    """
    env = collect_environment()
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return env


def describe_environment(env: dict[str, Any] | None = None) -> str:
    """Render a short hardware description suitable for the report's methods section."""
    env = env or collect_environment()
    cpu = env["cpu"]
    gpu = env["gpu"]
    ram_gb = env["memory"]["total_bytes"] / 1024**3
    gpu_desc = (
        ", ".join(
            f"{d['name']} ({d['total_memory_bytes'] / 1024**3:.0f} GB)" for d in gpu["devices"]
        )
        if gpu.get("available")
        else "none (CPU only)"
    )
    return (
        f"{cpu['processor']} "
        f"({cpu['physical_cores']} physical / {cpu['logical_cores']} logical cores), "
        f"{ram_gb:.1f} GB RAM, GPU: {gpu_desc}, "
        f"{env['platform']['system']} {env['platform']['release']}, "
        f"Python {env['python']['version']}"
    )
