"""Assemble the per-day blocks into the full traffic matrix.

Produces the ``(8928, 10000)`` float32 matrix that every later phase reads,
together with the timestamp index and the per-square totals used to pick the
study areas.

Assembly is deliberately strict. The daily blocks are stacked into a single
pre-allocated array rather than concatenated, so peak memory is one matrix
(~341 MiB) plus one day (~5.5 MiB) rather than two copies of the matrix. Each
day is memory-mapped in and copied straight into its slice.

What is checked rather than assumed
-----------------------------------
* Every expected day is present, in order, with no duplicate dates.
* The union of timestamps is exactly the expected 10-minute grid -- no gaps, no
  duplicates, no drift.
* No DST transition falls inside 2013-11-01..2014-01-01 (Italy switched on
  2013-10-27 and 2014-03-30), so every local day is exactly 144 intervals.
  This is asserted, not assumed, because a 23- or 25-hour day would silently
  misalign everything downstream.

Missing data
------------
Two different things get conflated if one is not careful:

``absent cells``
    A ``(square, time)`` pair with no row in the raw file. Means no activity
    was recorded, so it is 0.0. Counted at ingest, carried through here.
``missing intervals``
    A timestamp absent from the grid entirely. That is data loss, not silence,
    and is treated as such: short gaps are interpolated, longer ones stay NaN
    and are reported, and a gap inside the test week is escalated rather than
    quietly filled.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import zoneinfo
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Config, load_config
from src.memory_profiling import format_bytes, measure

__all__ = [
    "MatrixReport",
    "assemble_matrix",
    "apply_missing_policy",
    "expected_days",
    "to_local",
    "save_matrix",
    "load_matrix",
]


class MatrixError(RuntimeError):
    """Raised when the assembled matrix does not match the expected structure."""


@dataclass
class MatrixReport:
    """Provenance and data-quality summary for the assembled matrix."""

    shape: tuple[int, int]
    n_days: int
    first_timestamp_utc: str
    last_timestamp_utc: str
    first_timestamp_local: str
    last_timestamp_local: str
    timezone: str
    total_absent_cells: int
    missing_intervals: list[str] = field(default_factory=list)
    interpolated_intervals: list[str] = field(default_factory=list)
    unfilled_intervals: list[str] = field(default_factory=list)
    days_with_wrong_interval_count: list[str] = field(default_factory=list)
    total_internet: float = 0.0
    n_nan_after_policy: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def expected_days(config: Config) -> list[str]:
    """Every date the observation period should contain, in order."""
    start, end = config.dataset.start_date, config.dataset.end_date
    return [
        (start + dt.timedelta(days=offset)).isoformat() for offset in range((end - start).days + 1)
    ]


def _available_days(interim_dir: Path) -> dict[str, Path]:
    """Map date string to block path for every ingested day found."""
    days: dict[str, Path] = {}
    for path in sorted(interim_dir.glob("*.npy")):
        if path.name.endswith(".times.npy"):
            continue
        days[path.stem] = path
    return days


def to_local(timestamps: np.ndarray, timezone: str) -> np.ndarray:
    """Convert UTC ``datetime64[ns]`` to wall-clock time in ``timezone``.

    Returned as naive ``datetime64`` carrying local wall time, which is what
    the splits and the diurnal analysis are expressed in.
    """
    zone = zoneinfo.ZoneInfo(timezone)
    as_datetimes = timestamps.astype("datetime64[s]").astype(dt.datetime)
    local = [d.replace(tzinfo=dt.UTC).astimezone(zone).replace(tzinfo=None) for d in as_datetimes]
    return np.array(local, dtype="datetime64[ns]")


def _check_grid(timestamps: np.ndarray, config: Config) -> list[str]:
    """Verify the timestamps form the exact expected 10-minute grid.

    Returns:
        Timestamps that should exist but do not, as ISO strings.
    """
    step_ns = config.dataset.interval_minutes * 60 * 1_000_000_000
    expected = np.arange(
        timestamps[0].astype("datetime64[ns]").astype("int64"),
        timestamps[-1].astype("datetime64[ns]").astype("int64") + step_ns,
        step_ns,
    ).astype("datetime64[ns]")

    present = set(timestamps.astype("int64").tolist())
    missing = [str(ts) for ts in expected if ts.astype("int64") not in present]

    duplicates = timestamps.size - len(present)
    if duplicates:
        raise MatrixError(f"{duplicates} duplicate timestamp(s) in the assembled index")
    return missing


def assemble_matrix(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, MatrixReport]:
    """Stack every ingested day into the full traffic matrix.

    Args:
        config: Study configuration.

    Returns:
        ``(matrix, utc_timestamps, square_ids, report)``.

    Raises:
        MatrixError: If days are missing, mis-sized, or the timestamp grid is
            not contiguous.
    """
    interim = config.paths.interim
    available = _available_days(interim)
    all_days = expected_days(config)

    absent_days = [d for d in all_days if d not in available]
    if absent_days:
        raise MatrixError(
            f"{len(absent_days)} day(s) not ingested, first missing {absent_days[0]}. "
            "Run the ingest before building the matrix."
        )

    n_times = config.dataset.n_timestamps
    n_squares = config.dataset.n_squares
    matrix = np.zeros((n_times, n_squares), dtype=np.float32)
    all_times = np.empty(n_times, dtype="datetime64[ns]")

    total_absent = 0
    wrong_length: list[str] = []
    row = 0
    for date in all_days:
        block = np.load(available[date], mmap_mode="r")
        times = np.load(interim / f"{date}.times.npy")
        stats = json.loads((interim / f"{date}.json").read_text(encoding="utf-8"))
        total_absent += int(stats.get("n_absent_cells", 0))

        if block.shape[0] != config.dataset.daily_period:
            wrong_length.append(date)
        if block.shape[1] != n_squares:
            raise MatrixError(f"{date}: block has {block.shape[1]} squares, expected {n_squares}")

        end = row + block.shape[0]
        if end > n_times:
            raise MatrixError(f"{date}: blocks exceed the expected {n_times} timestamps")
        matrix[row:end] = block
        all_times[row:end] = times
        row = end
        del block

    if row != n_times:
        raise MatrixError(f"assembled {row} timestamps, expected {n_times}")
    if wrong_length:
        raise MatrixError(
            f"{len(wrong_length)} day(s) do not have "
            f"{config.dataset.daily_period} intervals: {wrong_length[:5]}"
        )

    missing = _check_grid(all_times, config)
    local = to_local(all_times, config.dataset.timezone)
    _assert_no_dst_shift(local, config)

    square_ids = np.arange(1, n_squares + 1, dtype=np.uint16)
    report = MatrixReport(
        shape=(int(matrix.shape[0]), int(matrix.shape[1])),
        n_days=len(all_days),
        first_timestamp_utc=str(all_times[0]),
        last_timestamp_utc=str(all_times[-1]),
        first_timestamp_local=str(local[0]),
        last_timestamp_local=str(local[-1]),
        timezone=config.dataset.timezone,
        total_absent_cells=total_absent,
        missing_intervals=missing,
        days_with_wrong_interval_count=wrong_length,
        total_internet=float(matrix.sum(dtype=np.float64)),
    )
    return matrix, all_times, square_ids, report


def _assert_no_dst_shift(local: np.ndarray, config: Config) -> None:
    """Assert every local day holds exactly ``daily_period`` intervals.

    Italy's 2013 DST change was on 27 October, before the window opens, so a
    deviation here means either a wrong timezone or a malformed day -- both of
    which would shift the diurnal cycle that the whole study rests on.
    """
    days = local.astype("datetime64[D]")
    _, counts = np.unique(days, return_counts=True)
    odd = counts[counts != config.dataset.daily_period]
    if odd.size:
        raise MatrixError(
            f"{odd.size} local day(s) do not have {config.dataset.daily_period} "
            f"intervals (found counts {sorted(set(odd.tolist()))}). "
            "A DST transition or a malformed day is the likely cause."
        )


# --------------------------------------------------------------------------
# Missing-interval policy
# --------------------------------------------------------------------------


def apply_missing_policy(
    matrix: np.ndarray,
    local_times: np.ndarray,
    config: Config,
    report: MatrixReport,
) -> np.ndarray:
    """Interpolate short all-NaN gaps; leave longer ones and report them.

    Only whole missing intervals are treated here. A cell that is simply zero
    is genuine silence and is left alone.

    Raises:
        MatrixError: If an unfillable gap lands inside the test week and the
            configuration asks for that to be escalated.
    """
    nan_rows = np.flatnonzero(np.isnan(matrix).all(axis=1))
    if nan_rows.size == 0:
        report.n_nan_after_policy = int(np.isnan(matrix).sum())
        return matrix

    limit = config.missing_data.max_interpolate_gap
    for start, length in _runs(nan_rows):
        stamps = [str(local_times[start + offset]) for offset in range(length)]
        if length <= limit and start > 0 and start + length < matrix.shape[0]:
            before = matrix[start - 1]
            after = matrix[start + length]
            for offset in range(length):
                weight = (offset + 1) / (length + 1)
                matrix[start + offset] = before * (1 - weight) + after * weight
            report.interpolated_intervals.extend(stamps)
        else:
            report.unfilled_intervals.extend(stamps)

    if config.missing_data.escalate_if_in_test and report.unfilled_intervals:
        test = config.splits.test
        inside = [
            ts
            for ts in report.unfilled_intervals
            if test.start <= dt.date.fromisoformat(ts[:10]) <= test.end
        ]
        if inside:
            raise MatrixError(
                f"{len(inside)} unfilled interval(s) fall inside the test week "
                f"({test}): {inside[:5]}. This must be resolved, not imputed."
            )

    report.n_nan_after_policy = int(np.isnan(matrix).sum())
    return matrix


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    """Group sorted indices into ``(start, length)`` consecutive runs."""
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) != 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [indices.size - 1]))
    return [
        (int(indices[s]), int(indices[e] - indices[s] + 1))
        for s, e in zip(starts, ends, strict=False)
    ]


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save_matrix(
    matrix: np.ndarray,
    utc_times: np.ndarray,
    square_ids: np.ndarray,
    report: MatrixReport,
    config: Config,
) -> dict[str, Path]:
    """Write the matrix, index, square ids, per-square totals and the report."""
    import polars as pl

    processed = config.paths.processed
    processed.mkdir(parents=True, exist_ok=True)
    local = to_local(utc_times, config.dataset.timezone)

    paths = {
        "matrix": processed / "traffic_matrix.npy",
        "timestamps": processed / "timestamps.parquet",
        "square_ids": processed / "square_ids.npy",
        "totals": processed / "totals.parquet",
        "report": processed / "matrix_report.json",
    }

    np.save(paths["matrix"], matrix)
    np.save(paths["square_ids"], square_ids)

    pl.DataFrame(
        {
            "index": np.arange(utc_times.size, dtype=np.int32),
            "utc": utc_times,
            "local": local,
        }
    ).write_parquet(paths["timestamps"], compression="zstd")

    totals = matrix.sum(axis=0, dtype=np.float64)
    pl.DataFrame(
        {
            "square_id": square_ids.astype(np.int32),
            "total_internet": totals,
            "mean_internet": totals / matrix.shape[0],
        }
    ).write_parquet(paths["totals"], compression="zstd")

    paths["report"].write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return paths


def load_matrix(config: Config, *, mmap: bool = True) -> tuple[np.ndarray, Any, np.ndarray]:
    """Load the assembled matrix, timestamp table and square ids.

    Args:
        config: Study configuration.
        mmap: Memory-map the matrix instead of reading it into RAM. Downstream
            phases only ever touch a handful of columns, so mapping keeps the
            resident set at a few MB rather than 341 MiB.
    """
    import polars as pl

    processed = config.paths.processed
    matrix = np.load(processed / "traffic_matrix.npy", mmap_mode="r" if mmap else None)
    timestamps = pl.read_parquet(processed / "timestamps.parquet")
    square_ids = np.load(processed / "square_ids.npy")
    return matrix, timestamps, square_ids


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Assemble per-day blocks into the full traffic matrix."
    )
    parser.add_argument("--config", default=None, help="Path to a config YAML.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Assemble, validate, apply the missing-data policy and persist."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    config.paths.mkdirs()

    with measure("assemble_matrix", trace_python_allocs=False) as report_mem:
        matrix, utc_times, square_ids, report = assemble_matrix(config)
        local = to_local(utc_times, config.dataset.timezone)
        matrix = apply_missing_policy(matrix, local, config, report)
        paths = save_matrix(matrix, utc_times, square_ids, report, config)

    print(f"shape            : {report.shape}")
    print(f"days             : {report.n_days}")
    print(f"UTC range        : {report.first_timestamp_utc} .. {report.last_timestamp_utc}")
    print(f"local range      : {report.first_timestamp_local} .. {report.last_timestamp_local}")
    print(f"absent cells     : {report.total_absent_cells:,}")
    print(f"missing intervals: {len(report.missing_intervals)}")
    print(f"interpolated     : {len(report.interpolated_intervals)}")
    print(f"left as NaN      : {len(report.unfilled_intervals)}")
    print(f"total internet   : {report.total_internet:,.2f}")
    print(f"peak RSS         : {format_bytes(report_mem.rss_peak_delta)}")
    print(f"wall             : {report_mem.wall_s:.1f}s")
    for name, path in paths.items():
        print(f"  {name:<11} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
