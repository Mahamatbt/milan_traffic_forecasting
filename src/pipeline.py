"""Streaming download-ingest-delete pipeline.

Running download and ingest as separate phases is fine locally, where the raw
files persist. On Kaggle it is the wrong shape: ``/kaggle/temp`` is scratch that
is discarded when the session ends, and sessions are capped at 12 hours. A run
that downloads all 19.4 GiB and then ingests would, on a timeout at hour 11,
lose every raw file and have nothing to show for it.

This module interleaves the two instead. For each day in turn it downloads the
file, converts it to a dense block, writes the block to persistent storage, and
deletes the raw text before moving on. That gives two properties the phased
version cannot:

* **Peak disk of roughly one file** (~350 MB) rather than 19.4 GiB.
* **Progress that survives a timeout.** Every completed day is already a
  finished artefact, so a resumed run skips it and continues. Nothing is
  re-downloaded.

The same command runs locally; only ``ingest.delete_raw_after_ingest`` differs,
so a local run keeps its raw files while a Kaggle run discards them.
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config import Config, load_config
from src.download import (
    DataverseClient,
    DataverseFile,
    GuestbookResponse,
    download_file,
    load_dotenv,
)
from src.ingest import STRATEGIES, date_from_path, day_is_complete, write_day
from src.memory_profiling import format_bytes, measure

__all__ = ["DayOutcome", "PipelineResult", "run_streaming"]


@dataclass
class DayOutcome:
    """What happened to one day."""

    date: str
    skipped: bool
    downloaded_bytes: int
    download_s: float
    ingest_s: float
    peak_rss_bytes: int
    n_raw_rows: int
    n_absent_cells: int
    raw_deleted: bool

    @property
    def total_s(self) -> float:
        return self.download_s + self.ingest_s


@dataclass
class PipelineResult:
    """Summary across every day the run touched."""

    outcomes: list[DayOutcome] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    @property
    def processed(self) -> list[DayOutcome]:
        return [o for o in self.outcomes if not o.skipped]

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    def summary(self) -> str:
        """Human-readable wrap-up for the notebook output."""
        done = self.processed
        if not done:
            return f"{len(self.outcomes)} day(s) already complete; nothing to do."
        downloaded = sum(o.downloaded_bytes for o in done)
        peak = max(o.peak_rss_bytes for o in done)
        return "\n".join(
            [
                f"days processed   : {len(done)} ({len(self.outcomes) - len(done)} skipped)",
                f"downloaded       : {format_bytes(downloaded)}",
                f"download time    : {sum(o.download_s for o in done) / 60:.1f} min",
                f"ingest time      : {sum(o.ingest_s for o in done) / 60:.1f} min",
                f"peak RSS         : {format_bytes(peak)}",
                f"raw rows         : {sum(o.n_raw_rows for o in done):,}",
                f"absent cells     : {sum(o.n_absent_cells for o in done):,}",
                f"wall elapsed     : {self.elapsed_s / 60:.1f} min",
            ]
        )


def _remaining_days(
    files: list[DataverseFile], config: Config, *, force: bool
) -> list[tuple[str, DataverseFile]]:
    """Pair each file with its date, dropping days already ingested."""
    pairs = []
    for entry in files:
        date = date_from_path(Path(entry.filename))
        if not force and day_is_complete(date, config.paths.interim):
            continue
        pairs.append((date, entry))
    return sorted(pairs, key=lambda pair: pair[0])


def process_day(
    client: DataverseClient,
    entry: DataverseFile,
    date: str,
    config: Config,
    guestbook: GuestbookResponse,
    *,
    strategy: str,
    show_progress: bool,
) -> DayOutcome:
    """Download, ingest, persist and discard a single day.

    The raw file is deleted only after its block has been written, so an
    interruption between the two leaves the raw file in place to be resumed
    from rather than losing it.

    Args:
        client: Configured Dataverse client.
        entry: File metadata from the Dataverse listing.
        date: ISO date the file covers.
        config: Study configuration.
        guestbook: Identity recorded against each download request.
        strategy: Ingest implementation name.
        show_progress: Render a download progress bar.

    Returns:
        A :class:`DayOutcome` recording timings and data-quality counts.
    """
    ingest = STRATEGIES[strategy]

    started = time.perf_counter()
    result = download_file(client, entry, config.paths.raw, guestbook, show_progress=show_progress)
    download_s = time.perf_counter() - started

    with measure(f"ingest:{date}", trace_python_allocs=False) as report:
        block = ingest(result.path, config)
        write_day(block, config.paths.interim)

    stats = block.stats
    del block
    gc.collect()

    deleted = False
    if config.ingest.delete_raw_after_ingest:
        result.path.unlink(missing_ok=True)
        deleted = True

    return DayOutcome(
        date=date,
        skipped=False,
        downloaded_bytes=0 if result.skipped else result.size_bytes,
        download_s=download_s,
        ingest_s=report.wall_s,
        peak_rss_bytes=report.rss_peak_delta,
        n_raw_rows=stats.n_raw_rows,
        n_absent_cells=stats.n_absent_cells,
        raw_deleted=deleted,
    )


def run_streaming(
    config: Config,
    guestbook: GuestbookResponse,
    *,
    strategy: str = "polars_lazy",
    limit: int | None = None,
    force: bool = False,
    show_progress: bool = True,
    time_budget_s: float | None = None,
) -> PipelineResult:
    """Download and ingest every outstanding day, one at a time.

    Args:
        config: Study configuration.
        guestbook: Identity recorded against each download request.
        strategy: Ingest implementation name.
        limit: Process at most this many outstanding days.
        force: Re-process days that are already complete.
        show_progress: Render download progress bars.
        time_budget_s: Stop cleanly once this much wall time has elapsed,
            rather than being killed mid-day by a session timeout. The next run
            resumes from the first unfinished day.

    Returns:
        A :class:`PipelineResult` covering every day considered.
    """
    config.paths.mkdirs()
    client = DataverseClient(config.dataset.dataverse_base)

    print(f"listing {config.dataset.doi_traffic} ...", flush=True)
    files = client.list_files(config.dataset.doi_traffic)
    outstanding = _remaining_days(files, config, force=force)
    already = len(files) - len(outstanding)

    if limit:
        outstanding = outstanding[:limit]

    print(
        f"{len(files)} day(s) in dataset, {already} already ingested, "
        f"{len(outstanding)} to process"
    )
    if config.ingest.delete_raw_after_ingest:
        print("raw files are deleted after ingest; peak disk stays near one file\n")
    else:
        print("raw files are retained after ingest\n")

    result = PipelineResult()
    for index, (date, entry) in enumerate(outstanding, start=1):
        if time_budget_s is not None and result.elapsed_s > time_budget_s:
            print(
                f"\ntime budget of {time_budget_s / 60:.0f} min reached; "
                f"stopping cleanly after {index - 1} day(s). Re-run to resume."
            )
            break

        outcome = process_day(
            client,
            entry,
            date,
            config,
            guestbook,
            strategy=strategy,
            show_progress=show_progress,
        )
        result.outcomes.append(outcome)
        print(
            f"[{index:>2}/{len(outstanding)}] {date}  "
            f"dl {outcome.download_s:>6.1f}s  "
            f"ingest {outcome.ingest_s:>5.1f}s  "
            f"{outcome.n_raw_rows:>9,} rows  "
            f"absent {outcome.n_absent_cells:>3}  "
            f"peak {outcome.peak_rss_bytes / 1024**2:>5.0f} MiB",
            flush=True,
        )

    print("\n" + result.summary())
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Download and ingest the dataset one day at a time.",
        epilog=(
            "Guestbook identity comes from DATAVERSE_GB_* environment variables "
            "or a gitignored .env. On Kaggle, set them as Secrets."
        ),
    )
    parser.add_argument("--config", default=None, help="Path to a config YAML.")
    parser.add_argument(
        "--strategy", default="polars_lazy", choices=sorted(STRATEGIES), help="Ingest strategy."
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N days.")
    parser.add_argument("--force", action="store_true", help="Re-process completed days.")
    parser.add_argument("--no-progress", action="store_true", help="Suppress progress bars.")
    parser.add_argument(
        "--time-budget-min",
        type=float,
        default=None,
        help="Stop cleanly after this many minutes so a session cap cannot kill a day mid-write.",
    )
    parser.add_argument("--gb-name", default=None)
    parser.add_argument("--gb-email", default=None)
    parser.add_argument("--gb-institution", default=None)
    parser.add_argument("--gb-position", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the streaming pipeline from the command line."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    load_dotenv()
    guestbook = GuestbookResponse.from_args_or_env(args)

    run_streaming(
        config,
        guestbook,
        strategy=args.strategy,
        limit=args.limit,
        force=args.force,
        show_progress=not args.no_progress,
        time_budget_s=args.time_budget_min * 60 if args.time_budget_min else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
