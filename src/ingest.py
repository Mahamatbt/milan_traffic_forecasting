"""Convert raw daily CDR text into dense per-day traffic blocks.

The dataset ships as 62 tab-separated files averaging 5.16 M rows each, 19.38
GiB in total. The whole memory-management problem is getting from that to a
``8928 x 10000`` float32 matrix (340.58 MiB) without ever holding more than one
day in RAM.

Why the raw files are so much larger than the information they carry
--------------------------------------------------------------------
Rows are split by the counterparty's country code, so a single
``(square_id, time_ms)`` pair appears once per country that generated traffic
there -- 3.6 rows on average across the period, and at most 36. Aggregating over
country is therefore not an optional tidy-up; it is what turns ~5.16 M rows into
the 1.44 M cells a day actually contains.

(A file contains up to 246 distinct country codes, which is not the same as the
number of rows per cell; conflating the two overstates the duplication by 7x.)

Only three of the eight columns are needed (``square_id``, ``time_ms``,
``internet``), so projection pushdown discards 5/8 of the parsed data before it
is ever materialised.

Three strategies are implemented, and all three are measured:

``load_day_naive``
    Deliberately unoptimised pandas: no ``usecols``, no ``dtype``. This is the
    "before" measurement for the report, not a recommendation.
``ingest_day_pandas_chunked``
    Middle ground: projection, narrow dtypes and incremental aggregation over
    ``chunksize`` blocks.
``ingest_day_polars``
    Lazy scan with projection pushdown and a streaming group-by.

Measured trade-off
------------------
The two optimised strategies do not rank the same way on both axes, and the
assumption that the faster engine is also the leaner one turned out to be
wrong. Polars is roughly 6x faster per day but peaks at 552 MiB, because its
CSV reader pulls the whole ~322 MB file into memory before parsing; batched
reading (697-862 MiB) and ``low_memory=True`` (559 MiB) were both measured and
made it worse, not better. Chunked pandas peaks at 173 MiB because ``chunksize``
bounds residency by construction.

These are the subprocess-isolated figures. An earlier in-process benchmark
reported 544, 184 and 384 MiB for identical work, because freed arenas are not
returned to the OS and whichever strategy ran first absorbed the heap growth.

Which to prefer therefore depends on the machine, not on a general ranking, so
both are kept and ``--strategy`` selects. What actually matters is that both
are O(1) in the number of days: only one 5.49 MiB block is ever resident, where
the naive path would need 17.90 GiB to hold the same 62 days.

Missing cells
-------------
A row exists only where some activity was recorded, so a ``(square, time)``
pair absent from the file means no activity, not lost data. Those cells are
filled with 0.0 and the count is reported per day rather than silently
absorbed -- on 2013-11-01 there are 18 such cells out of 1,440,000 (0.0012%).
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import zoneinfo
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.config import Config, load_config
from src.memory_profiling import measure

__all__ = [
    "COLUMN_NAMES",
    "DayBlock",
    "DayStats",
    "load_day_naive",
    "ingest_day_polars",
    "ingest_day_pandas_chunked",
    "STRATEGIES",
]

# The files have no header; this is the documented column order.
COLUMN_NAMES = (
    "square_id",
    "time_ms",
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet",
)

# Indices of the only columns this study needs.
SQUARE_ID_COL, TIME_MS_COL, INTERNET_COL = 0, 1, 7
USED_COLUMNS = ("square_id", "time_ms", "internet")

_ROME = zoneinfo.ZoneInfo("Europe/Rome")


class IngestError(RuntimeError):
    """Raised when a day file does not match the expected structure."""


@dataclass
class DayStats:
    """Per-day provenance recorded alongside the block.

    These counts are what let the report state how much data was actually
    present rather than assuming a full grid.
    """

    date: str
    source: str
    n_raw_rows: int
    n_pairs: int
    n_absent_cells: int
    n_timestamps: int
    n_squares_seen: int
    total_internet: float
    strategy: str
    wall_s: float = 0.0

    @property
    def absent_fraction(self) -> float:
        """Share of the dense grid that had no row at all."""
        denominator = self.n_timestamps * self.n_squares_seen
        return self.n_absent_cells / denominator if denominator else 0.0

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["absent_fraction"] = self.absent_fraction
        return row


@dataclass
class DayBlock:
    """One day of traffic as a dense ``(n_timestamps, n_squares)`` array."""

    values: np.ndarray
    timestamps: np.ndarray  # datetime64[ns] in UTC
    stats: DayStats

    def __post_init__(self) -> None:
        if self.values.dtype != np.float32:
            raise IngestError(f"expected float32 block, got {self.values.dtype}")
        if self.values.shape[0] != self.timestamps.shape[0]:
            raise IngestError(
                f"block has {self.values.shape[0]} rows but "
                f"{self.timestamps.shape[0]} timestamps"
            )

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def date_from_path(path: Path) -> str:
    """Extract the ISO date encoded in a daily filename.

    Raises:
        IngestError: If the filename does not carry a parseable date.
    """
    stem = path.stem  # sms-call-internet-mi-2013-11-01
    parts = stem.split("-")
    if len(parts) < 3:
        raise IngestError(f"cannot parse a date from {path.name}")
    candidate = "-".join(parts[-3:])
    try:
        dt.date.fromisoformat(candidate)
    except ValueError as exc:
        raise IngestError(f"cannot parse a date from {path.name}") from exc
    return candidate


def _densify(
    square_ids: np.ndarray,
    time_ms: np.ndarray,
    values: np.ndarray,
    *,
    n_squares: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Scatter aggregated pairs into a dense ``(n_times, n_squares)`` array.

    Columns are ordered by square id 1..n_squares so every day shares one
    layout and the daily blocks can be stacked without a join.

    Returns:
        ``(block, unique_time_ms, n_absent_cells)``. Cells with no row are left
        at 0.0, which is the correct reading of an absent record: activity is
        only logged where it occurred.
    """
    unique_times, row_index = np.unique(time_ms, return_inverse=True)
    col_index = square_ids.astype(np.int64) - 1
    if col_index.min() < 0 or col_index.max() >= n_squares:
        raise IngestError(
            f"square ids outside 1..{n_squares}: " f"[{square_ids.min()}, {square_ids.max()}]"
        )

    block = np.zeros((unique_times.size, n_squares), dtype=np.float32)
    block[row_index, col_index] = values.astype(np.float32, copy=False)

    n_absent = unique_times.size * n_squares - values.size
    return block, unique_times, int(n_absent)


def _to_timestamps(time_ms: np.ndarray) -> np.ndarray:
    """Convert epoch milliseconds to UTC ``datetime64[ns]``.

    The stored values stay in UTC; conversion to ``Europe/Rome`` happens once,
    in :mod:`src.build_matrix`, so there is a single place where the timezone
    can be wrong.
    """
    return time_ms.astype("datetime64[ms]").astype("datetime64[ns]")


def _validate_day(block: np.ndarray, times: np.ndarray, config: Config, path: Path) -> None:
    """Assert a day has the expected shape before it is written."""
    expected = config.dataset.daily_period
    if times.size != expected:
        raise IngestError(
            f"{path.name}: expected {expected} intervals, found {times.size}. "
            "A short or long day must be investigated, not padded."
        )
    if block.shape[1] != config.dataset.n_squares:
        raise IngestError(
            f"{path.name}: expected {config.dataset.n_squares} squares, got {block.shape[1]}"
        )


# --------------------------------------------------------------------------
# Strategy 1: naive pandas (the "before" measurement)
# --------------------------------------------------------------------------


def load_day_naive(path: Path) -> Any:
    """Read a day with pandas defaults: every column, inferred dtypes.

    This exists to be measured, not used. With no ``usecols`` all eight columns
    are parsed and retained, and with no ``dtype`` pandas widens integers to
    ``int64`` and floats to ``float64`` -- roughly 64 bytes per row against the
    10 bytes the three needed columns require at their natural widths.

    Args:
        path: A raw daily ``.txt`` file.

    Returns:
        The full ``DataFrame``, held entirely in memory.
    """
    import pandas as pd

    return pd.read_csv(path, sep="\t", header=None, names=list(COLUMN_NAMES))


# --------------------------------------------------------------------------
# Strategy 2: chunked pandas
# --------------------------------------------------------------------------


def ingest_day_pandas_chunked(
    path: Path,
    config: Config,
    *,
    chunksize: int = 1_000_000,
) -> DayBlock:
    """Aggregate a day with pandas, one bounded chunk at a time.

    Peak memory is governed by ``chunksize`` rather than file size: each chunk
    is reduced to per-pair sums and discarded, so only the running aggregate
    and one chunk are ever resident.

    Args:
        path: A raw daily ``.txt`` file.
        config: Study configuration, for grid dimensions.
        chunksize: Rows per read block.

    Returns:
        The day as a :class:`DayBlock`.
    """
    import pandas as pd

    totals: dict[tuple[int, int], float] = {}
    n_raw = 0

    reader = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=list(COLUMN_NAMES),
        usecols=[SQUARE_ID_COL, TIME_MS_COL, INTERNET_COL],
        dtype={"square_id": "uint16", "time_ms": "int64", "internet": "float32"},
        chunksize=chunksize,
    )
    aggregated: Any = None
    for chunk in reader:
        n_raw += len(chunk)
        chunk["internet"] = chunk["internet"].fillna(config.ingest.null_internet_fill)
        partial = chunk.groupby(["square_id", "time_ms"], sort=False)["internet"].sum()
        aggregated = partial if aggregated is None else aggregated.add(partial, fill_value=0.0)
        del chunk, partial

    if aggregated is None:
        raise IngestError(f"{path.name}: file contained no rows")

    aggregated = aggregated.reset_index()
    block, times, n_absent = _densify(
        aggregated["square_id"].to_numpy(),
        aggregated["time_ms"].to_numpy(),
        aggregated["internet"].to_numpy(),
        n_squares=config.dataset.n_squares,
    )
    _validate_day(block, times, config, path)
    del aggregated, totals
    gc.collect()

    return DayBlock(
        values=block,
        timestamps=_to_timestamps(times),
        stats=DayStats(
            date=date_from_path(path),
            source=path.name,
            n_raw_rows=n_raw,
            n_pairs=int(block.size - n_absent),
            n_absent_cells=n_absent,
            n_timestamps=int(times.size),
            n_squares_seen=config.dataset.n_squares,
            total_internet=float(block.sum(dtype=np.float64)),
            strategy="pandas_chunked",
        ),
    )


# --------------------------------------------------------------------------
# Strategy 3: Polars lazy scan
# --------------------------------------------------------------------------


def _scan_day(path: Path) -> pl.LazyFrame:
    """Build the lazy plan for one day, projecting to the three needed columns.

    The schema is declared rather than inferred so no sampling pass is needed
    and ``country_code`` is never widened; the ``select`` immediately after the
    scan is what lets Polars push the projection into the CSV reader.
    """
    return pl.scan_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=list(COLUMN_NAMES),
        schema_overrides={
            "square_id": pl.UInt16,
            "time_ms": pl.Int64,
            "country_code": pl.Int32,
            "sms_in": pl.Float32,
            "sms_out": pl.Float32,
            "call_in": pl.Float32,
            "call_out": pl.Float32,
            "internet": pl.Float32,
        },
    ).select(USED_COLUMNS)


def _collect_streaming(plan: pl.LazyFrame) -> pl.DataFrame:
    """Collect a lazy plan with the streaming engine, across Polars versions.

    The spelling changed: ``collect(streaming=True)`` up to Polars 1.2x, then
    ``collect(engine="streaming")``. Both are tried so the same code runs on the
    pinned local version and on whatever Kaggle's image ships.
    """
    try:
        return plan.collect(engine="streaming")
    except (ValueError, TypeError):
        return plan.collect(streaming=True)


def ingest_day_polars(path: Path, config: Config) -> DayBlock:
    """Aggregate a day with a lazy Polars scan and streaming group-by.

    The ~5.16 M rows are never materialised: projection pushdown drops five
    of eight columns at parse time, and the group-by reduces to ~1.44 M pairs
    inside the engine.

    Args:
        path: A raw daily ``.txt`` file.
        config: Study configuration, for grid dimensions.

    Returns:
        The day as a :class:`DayBlock`.
    """
    plan = (
        _scan_day(path)
        .with_columns(pl.col("internet").fill_null(config.ingest.null_internet_fill))
        .group_by(["square_id", "time_ms"])
        # pl.len() counts the country-split rows per pair, so the raw row count
        # falls out of the same pass. Counting it with a second scan would
        # double both the parse time and the peak memory for one integer.
        .agg(pl.col("internet").sum(), pl.len().alias("n_rows"))
    )
    aggregated = _collect_streaming(plan)
    n_raw = int(aggregated["n_rows"].sum())

    block, times, n_absent = _densify(
        aggregated["square_id"].to_numpy(),
        aggregated["time_ms"].to_numpy(),
        aggregated["internet"].to_numpy(),
        n_squares=config.dataset.n_squares,
    )
    _validate_day(block, times, config, path)
    del aggregated
    gc.collect()

    return DayBlock(
        values=block,
        timestamps=_to_timestamps(times),
        stats=DayStats(
            date=date_from_path(path),
            source=path.name,
            n_raw_rows=int(n_raw),
            n_pairs=int(block.size - n_absent),
            n_absent_cells=n_absent,
            n_timestamps=int(times.size),
            n_squares_seen=config.dataset.n_squares,
            total_internet=float(block.sum(dtype=np.float64)),
            strategy="polars_lazy",
        ),
    )


STRATEGIES: dict[str, Callable[[Path, Config], DayBlock]] = {
    "polars_lazy": ingest_day_polars,
    "pandas_chunked": ingest_day_pandas_chunked,
}


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def write_day(block: DayBlock, interim_dir: Path) -> Path:
    """Persist one day's block and its provenance.

    The dense block is stored as ``.npy`` rather than Parquet. Parquet is
    columnar and would need 10,000 column chunks per day, whose metadata and
    per-column overhead cost more than they save on a small dense float32
    array; ``.npy`` also memory-maps, which is what :mod:`src.build_matrix`
    wants when stacking 62 days. The per-day statistics that make the ingest
    auditable live beside it as JSON.
    """
    interim_dir.mkdir(parents=True, exist_ok=True)
    values_path = interim_dir / f"{block.stats.date}.npy"
    np.save(values_path, block.values)
    np.save(interim_dir / f"{block.stats.date}.times.npy", block.timestamps)
    (interim_dir / f"{block.stats.date}.json").write_text(
        json.dumps(block.stats.to_row(), indent=2) + "\n", encoding="utf-8"
    )
    return values_path


def day_is_complete(date: str, interim_dir: Path) -> bool:
    """True when a day has already been ingested and can be skipped.

    Checkpointing per day is what makes the ingest survive Kaggle's 12-hour
    session cap: a timeout costs the remaining days, not the whole run.
    """
    return all(
        (interim_dir / f"{date}{suffix}").exists() for suffix in (".npy", ".times.npy", ".json")
    )


def raw_day_files(raw_dir: Path) -> list[Path]:
    """Every raw daily file present, in date order."""
    return sorted(raw_dir.glob("sms-call-internet-mi-*.txt"))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Convert raw daily CDR text into dense per-day blocks."
    )
    parser.add_argument("--config", default=None, help="Path to a config YAML.")
    parser.add_argument(
        "--strategy",
        default="polars_lazy",
        choices=sorted(STRATEGIES),
        help="Ingest implementation to use for the full run.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Ingest at most N days.")
    parser.add_argument(
        "--force", action="store_true", help="Re-ingest days that are already complete."
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        metavar="N",
        help="Measure every strategy on the first N days and write the memory report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Ingest available raw days, optionally benchmarking the strategies first."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    config.paths.mkdirs()

    files = raw_day_files(config.paths.raw)
    if not files:
        print(f"no raw day files found in {config.paths.raw}")
        return 1

    if args.benchmark:
        from src.memory_report import run_benchmark

        run_benchmark(files[: args.benchmark], config, config_path=args.config)

    todo = files[: args.limit] if args.limit else files
    ingest = STRATEGIES[args.strategy]

    print(f"ingesting {len(todo)} day(s) with {args.strategy} -> {config.paths.interim}")
    for index, path in enumerate(todo, start=1):
        date = date_from_path(path)
        if not args.force and day_is_complete(date, config.paths.interim):
            print(f"[{index:>2}/{len(todo)}] {date}  already complete, skipped")
            continue

        with measure(f"ingest:{date}", trace_python_allocs=False) as report:
            block = ingest(path, config)
            write_day(block, config.paths.interim)

        s = block.stats
        print(
            f"[{index:>2}/{len(todo)}] {date}  "
            f"{s.n_raw_rows:>9,} rows -> {s.n_pairs:>9,} cells  "
            f"absent={s.n_absent_cells:<5} "
            f"peak={report.rss_peak_delta / 1024**2:>6.0f} MiB  "
            f"{report.wall_s:>5.1f}s",
            flush=True,
        )
        del block
        gc.collect()

        if config.ingest.delete_raw_after_ingest:
            path.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
