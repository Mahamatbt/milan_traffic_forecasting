"""Benchmark the ingest strategies and produce the memory-management evidence.

Assignment section 1 asks for memory usage before and after optimisation. This
module produces that table by running each strategy over the same day files and
recording peak resident set size, wall time and what ends up resident.

Three measurement decisions determine whether the numbers mean anything:

1. **One process per measurement.** Measuring several strategies in one process
   produces nonsense: CPython and the allocators beneath it do not return freed
   arenas to the OS, so whichever strategy runs first pays for the heap growth
   and later ones appear artificially cheap. Running the same ingest three times
   in one process was observed to report 544, 184 and 384 MiB for identical
   work. Every measurement here therefore runs in a fresh subprocess.
2. **Peak RSS, not final size.** The peak happens during parsing and is gone by
   the time a frame is returned, so a sampler thread polls it at 20 ms. The
   naive strategy's cost is almost entirely transient.
3. **Available memory travels with each row.** Under memory pressure the OS
   trims working sets and compresses pages, which *depresses* measured peak RSS.
   A run on a loaded desktop understates demand, so the conditions are recorded
   rather than assumed away.

The naive strategy is measured on a single day and its full-dataset figure is
an extrapolation, flagged as such in the ``extrapolated`` column. Running it
over all 62 days would take hours to establish a point one file already makes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from src.config import PROJECT_ROOT, Config, load_config
from src.memory_profiling import format_bytes
from src.timing import describe_environment

__all__ = ["StrategyMeasurement", "run_benchmark", "REPORT_COLUMNS", "STRATEGY_NOTES"]

REPORT_COLUMNS = (
    "strategy",
    "day",
    "source_file_bytes",
    "per_day_inmem_bytes",
    "per_day_peak_rss_bytes",
    "per_day_wall_s",
    "projected_full_ram_bytes",
    "final_artifact_ram_bytes",
    "final_artifact_disk_bytes",
    "available_ram_at_start_bytes",
    "extrapolated",
    "notes",
)

STRATEGY_NOTES = {
    "naive_pandas": (
        "All 8 columns, inferred dtypes (int64/float64), whole file resident. "
        "Projection is one day x 62 days; not run over the full dataset."
    ),
    "pandas_chunked": (
        "3 of 8 columns, narrow dtypes, 1M-row chunks aggregated incrementally. "
        "chunksize bounds residency by construction."
    ),
    "polars_lazy": (
        "Lazy scan, projection pushdown to 3 of 8 columns, streaming group-by. "
        "Peak is dominated by the CSV reader holding the whole file."
    ),
}

_SAMPLE_INTERVAL_S = 0.02


@dataclass
class StrategyMeasurement:
    """One (strategy, day) measurement, taken in its own process."""

    strategy: str
    day: str
    source_file_bytes: int
    per_day_inmem_bytes: int | None
    per_day_peak_rss_bytes: int
    per_day_wall_s: float
    projected_full_ram_bytes: int | None
    final_artifact_ram_bytes: int | None
    final_artifact_disk_bytes: int | None
    available_ram_at_start_bytes: int
    extrapolated: bool
    notes: str

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["per_day_wall_s"] = round(self.per_day_wall_s, 3)
        return row


def _final_artifact_bytes(config: Config) -> int:
    """Size of the assembled traffic matrix in memory, as float32."""
    rows, cols = config.dataset.matrix_shape
    return rows * cols * np.dtype(np.float32).itemsize


# --------------------------------------------------------------------------
# Worker: runs exactly one measurement in a clean process
# --------------------------------------------------------------------------


def _run_worker(strategy: str, path: Path, config: Config) -> dict[str, Any]:
    """Measure one strategy on one file, in this process, and return raw numbers.

    Invoked as a subprocess by :func:`run_benchmark`, never directly.
    """
    from src.ingest import ingest_day_pandas_chunked, ingest_day_polars, load_day_naive
    from src.memory_profiling import deep_size_bytes

    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], process.memory_info().rss)
            except psutil.Error:
                return
            stop.wait(_SAMPLE_INTERVAL_S)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()

    available = psutil.virtual_memory().available
    started = time.perf_counter()
    if strategy == "naive_pandas":
        frame = load_day_naive(path)
        inmem = deep_size_bytes(frame)
        total = None
    else:
        fn = ingest_day_polars if strategy == "polars_lazy" else ingest_day_pandas_chunked
        block = fn(path, config)
        inmem = block.nbytes
        total = block.stats.total_internet
    wall = time.perf_counter() - started

    stop.set()
    sampler.join(timeout=1.0)

    return {
        "strategy": strategy,
        "peak_rss_delta": int(peak[0] - baseline),
        "baseline_rss": int(baseline),
        "inmem_bytes": int(inmem) if inmem else None,
        "wall_s": wall,
        "available_at_start": int(available),
        "total_internet": total,
    }


# --------------------------------------------------------------------------
# Parent: orchestrates one subprocess per measurement
# --------------------------------------------------------------------------


def _worker_environment() -> dict[str, str]:
    """Environment for a worker, with the repository on ``PYTHONPATH``.

    The caller's working directory is not the repository root in every
    environment that matters. A Kaggle notebook runs from ``/kaggle/working``
    while the clone lives in ``/kaggle/working/repo``, so a worker launched with
    the caller's cwd fails with ``No module named 'src'``. The parent's
    ``sys.path`` does not help either: it is not inherited by a subprocess.

    Anchoring on :data:`src.config.PROJECT_ROOT` -- which is derived from this
    package's own location rather than from cwd -- makes the worker find ``src``
    wherever it was launched from.
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    root = str(PROJECT_ROOT)
    parts = [root, *(p for p in existing.split(os.pathsep) if p and p != root)]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _measure_isolated(strategy: str, path: Path, config_path: str | None) -> dict[str, Any]:
    """Run one measurement in a fresh interpreter and parse its result.

    Raises:
        RuntimeError: If the worker fails or emits no parseable result. An OOM
            kill shows up here, and is worth reporting rather than swallowing.
    """
    cmd = [
        sys.executable,
        "-m",
        "src.memory_report",
        "--worker",
        "--strategy",
        strategy,
        "--path",
        str(Path(path).resolve()),
    ]
    if config_path:
        cmd += ["--config", str(Path(config_path).resolve())]

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=_worker_environment(),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{strategy} on {path.name} exited {completed.returncode}: "
            f"{completed.stderr.strip()[-500:]}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"{strategy} on {path.name} produced no result line")


def run_benchmark(
    paths: list[Path],
    config: Config,
    *,
    include_naive: bool = True,
    config_path: str | None = None,
) -> list[StrategyMeasurement]:
    """Measure every strategy across the given days, each in its own process.

    Args:
        paths: Raw daily files to benchmark over.
        config: Study configuration.
        include_naive: Measure the unoptimised baseline, on the first day only.
        config_path: Config file to hand each worker, so it sees the same
            settings as the parent.

    Returns:
        Every measurement taken. Also written to
        ``results/tables/memory_report.csv``.
    """
    measurements: list[StrategyMeasurement] = []
    totals: dict[str, set[float]] = {}

    print(f"\nhardware: {describe_environment()}")
    print(
        f"available RAM at start: "
        f"{format_bytes(psutil.virtual_memory().available)} of "
        f"{format_bytes(psutil.virtual_memory().total)}"
    )
    print(f"benchmarking {len(paths)} day(s), one process per measurement\n")

    for path in paths:
        wanted = ["pandas_chunked", "polars_lazy"]
        if include_naive and path == paths[0]:
            wanted.insert(0, "naive_pandas")

        for strategy in wanted:
            raw = _measure_isolated(strategy, path, config_path)
            measurement = _to_measurement(raw, path, config)
            measurements.append(measurement)
            _print_measurement(measurement)

            if raw.get("total_internet") is not None:
                totals.setdefault(measurement.day, set()).add(round(raw["total_internet"], 2))

    _cross_check(totals)
    _write_report(measurements, config)
    _print_summary(measurements, config)
    return measurements


def _to_measurement(raw: dict[str, Any], path: Path, config: Config) -> StrategyMeasurement:
    """Turn a worker's raw numbers into a report row."""
    from src.ingest import date_from_path

    strategy = raw["strategy"]
    naive = strategy == "naive_pandas"
    inmem = raw["inmem_bytes"]

    return StrategyMeasurement(
        strategy=strategy,
        day=date_from_path(path),
        source_file_bytes=path.stat().st_size,
        per_day_inmem_bytes=inmem,
        per_day_peak_rss_bytes=raw["peak_rss_delta"],
        per_day_wall_s=raw["wall_s"],
        # Naive keeps the whole day resident, so holding the dataset scales with
        # the number of days. The optimised paths hold one block at a time, so
        # their working set is flat in the number of days.
        projected_full_ram_bytes=(
            (inmem or 0) * config.dataset.n_days if naive else raw["peak_rss_delta"]
        ),
        final_artifact_ram_bytes=None if naive else _final_artifact_bytes(config),
        final_artifact_disk_bytes=None if naive else (inmem or 0) * config.dataset.n_days,
        available_ram_at_start_bytes=raw["available_at_start"],
        extrapolated=naive,
        notes=STRATEGY_NOTES[strategy],
    )


def _cross_check(totals: dict[str, float | set[float]], *, rel_tol: float = 1e-6) -> None:
    """Assert the optimised strategies agree on each day's total.

    Two implementations are only worth having if they are compared; a real
    disagreement means one of them is aggregating incorrectly.

    The comparison is relative, not exact. Both paths accumulate ~4.8 M float32
    values in different orders, so the totals differ in the last representable
    digit -- 82,479,246.57 against 82,479,246.58 is float32 rounding, not a bug.
    A tolerance of 1e-6 is far tighter than any aggregation error would be while
    still admitting that.
    """
    for day, values in sorted(totals.items()):
        observed = sorted(values)
        if len(observed) < 2:
            continue
        spread = observed[-1] - observed[0]
        scale = max(abs(observed[0]), abs(observed[-1]), 1.0)
        if spread / scale > rel_tol:
            raise RuntimeError(
                f"{day}: strategies disagree on total internet activity by "
                f"{spread:.4f} ({spread / scale:.2e} relative): {observed}"
            )


def _print_measurement(m: StrategyMeasurement) -> None:
    inmem = format_bytes(m.per_day_inmem_bytes) if m.per_day_inmem_bytes else "n/a"
    print(
        f"  {m.strategy:<16} {m.day}  resident {inmem:>10}  "
        f"peak {format_bytes(m.per_day_peak_rss_bytes):>10}  {m.per_day_wall_s:>6.2f}s"
    )


def _write_report(measurements: list[StrategyMeasurement], config: Config) -> Path:
    """Write the CSV the notebook and the report table read from."""
    path = config.paths.tables / "memory_report.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(REPORT_COLUMNS))
        writer.writeheader()
        for m in measurements:
            writer.writerow(m.to_row())
    print(f"\nwrote {path}")
    return path


def _print_summary(measurements: list[StrategyMeasurement], config: Config) -> None:
    """Print the headline comparison the report quotes."""
    grouped: dict[str, list[StrategyMeasurement]] = {}
    for m in measurements:
        grouped.setdefault(m.strategy, []).append(m)

    print("\n--- mean across measured days ---")
    baseline = grouped.get("naive_pandas")
    baseline_peak = (
        sum(m.per_day_peak_rss_bytes for m in baseline) / len(baseline) if baseline else None
    )

    for name in ("naive_pandas", "pandas_chunked", "polars_lazy"):
        group = grouped.get(name)
        if not group:
            continue
        peak = sum(m.per_day_peak_rss_bytes for m in group) / len(group)
        wall = sum(m.per_day_wall_s for m in group) / len(group)
        line = f"{name:<16} peak {format_bytes(peak):>10}  {wall:>6.2f}s/day"
        if baseline_peak and name != "naive_pandas":
            line += f"   ({baseline_peak / peak:.1f}x lower peak)"
        print(line)

    if baseline:
        projected = baseline[0].projected_full_ram_bytes or 0
        print(
            f"\nholding all {config.dataset.n_days} days the naive way: "
            f"{format_bytes(projected)} resident"
        )
    optimised = grouped.get("polars_lazy") or grouped.get("pandas_chunked")
    if optimised:
        print(
            f"one day at a time:                  "
            f"{format_bytes(optimised[0].per_day_inmem_bytes or 0)} resident per block"
        )
    print(
        f"final artifact: {config.dataset.matrix_shape[0]} x "
        f"{config.dataset.matrix_shape[1]} float32 = "
        f"{format_bytes(_final_artifact_bytes(config))}"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description="Benchmark ingest strategies.")
    parser.add_argument("--config", default=None, help="Path to a config YAML.")
    parser.add_argument("--days", type=int, default=1, help="Number of days to benchmark over.")
    parser.add_argument("--skip-naive", action="store_true", help="Omit the unoptimised baseline.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--strategy", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--path", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark, or a single isolated measurement in worker mode."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.worker:
        result = _run_worker(args.strategy, Path(args.path), config)
        print(json.dumps(result))
        return 0

    from src.ingest import raw_day_files

    files = raw_day_files(config.paths.raw)
    if not files:
        print(f"no raw day files in {config.paths.raw}; nothing to benchmark")
        return 1

    config.paths.mkdirs()
    run_benchmark(
        files[: args.days],
        config,
        include_naive=not args.skip_naive,
        config_path=args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
