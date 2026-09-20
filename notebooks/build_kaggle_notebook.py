"""Generate the Kaggle ingest notebook.

The notebook is generated rather than hand-edited so it stays thin and stays in
sync with ``src/``. Editing JSON by hand invites drift between what the
notebook runs and what the repository actually contains, which is exactly the
failure mode the "logic lives in src/" rule exists to prevent.

Run from the repository root:

    python notebooks/build_kaggle_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent / "00_kaggle_ingest.ipynb"


def md(text: str) -> dict:
    """A markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    """A code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


CELLS = [
    md(
        """
# Milan traffic forecasting — dataset ingest

Turns the 19.4 GiB Telecom Italia CDR dataset into the artefacts every later
stage reads: a `8928 x 10000` float32 matrix (~341 MiB), its timestamp index,
and per-square totals.

**Run this once.** Save the output as a Kaggle Dataset and the EDA and modelling
notebooks attach it directly — no second 19 GiB download, by you or anyone
reproducing this.

### Before running

1. **Settings → Internet → On** (needs a phone-verified account).
2. **Add-ons → Secrets**, add four secrets. Harvard Dataverse requires a
   Guestbook response and records one per download request, so these must be
   your real details:
   `DATAVERSE_GB_NAME`, `DATAVERSE_GB_EMAIL`,
   `DATAVERSE_GB_INSTITUTION`, `DATAVERSE_GB_POSITION`
3. **Accelerator → None.** This stage is I/O-bound; a GPU would burn quota for
   nothing. Turn it on for the modelling notebook instead.
4. Prefer **Save & Run All (Commit)** over an interactive run. `/kaggle/working`
   is only persisted by a committed version, and a batch run gets the full
   session without the 60-minute idle timeout.

### Why this notebook streams

`/kaggle/temp` is scratch that is discarded when the session ends, and sessions
are capped at 12 hours. Downloading all 62 files and *then* ingesting would, on
a timeout at hour 11, lose every raw file and leave nothing behind. Instead each
day is downloaded, converted to a dense block, written to `/kaggle/working`, and
its raw text deleted before the next day starts. Peak disk stays near one file
(~350 MB) instead of 19.4 GiB, and every finished day is a durable artefact.
"""
    ),
    md("## 1. Settings"),
    code(
        """
# Point this at your own fork/clone before running.
REPO_URL = "https://github.com/YOUR_USERNAME/milan_traffic_forecasting.git"
BRANCH = "main"

# Stop cleanly before Kaggle's 12 h cap so a day is never killed mid-write.
# Re-running resumes from the first unfinished day.
TIME_BUDGET_MIN = 660

# polars_lazy is ~6x faster per day; pandas_chunked peaks ~3x lower.
# Kaggle has ~30 GB of RAM, so speed is the right trade here.
STRATEGY = "polars_lazy"

# Process only the first N days (useful for a quick dry run). None = all 62.
LIMIT = None
"""
    ),
    md("## 2. Clone the repository and install dependencies"),
    code(
        """
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path("/kaggle/working/repo")

if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO_DIR)],
    check=True,
)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r",
     str(REPO_DIR / "requirements-kaggle.txt")],
    check=True,
)

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# Drop any previously imported src.* modules. Re-running this cell replaces the
# files on disk, but sys.modules still holds the code objects compiled from the
# old ones, so the kernel would keep executing the previous version -- visible
# as a traceback whose line numbers land on docstrings.
for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[name]
importlib.invalidate_caches()

print("repo:", subprocess.run(
    ["git", "-C", str(REPO_DIR), "log", "-1", "--format=%h %s"],
    capture_output=True, text=True).stdout.strip())

# Dynamic, because the package only exists once the clone above has run.
_check = importlib.import_module("src.memory_report")
assert hasattr(_check, "_worker_environment"), (
    "stale src/ still loaded: restart the kernel "
    "(Run -> Restart & Clear Cell Outputs), then run all cells again"
)
print("src version: OK")
"""
    ),
    md(
        """
## 3. Load configuration and record the hardware

`src.config` detects the Kaggle session automatically and layers
`config/kaggle.yaml` over the defaults, so paths point at `/kaggle/temp/raw`
and `/kaggle/working/...` with no edits here.

The environment snapshot is what assignment requirement IV's timings are
attributed to, so it is captured before any work starts.
"""
    ),
    code(
        """
from src.config import load_config
from src.timing import describe_environment, record_hardware

config = load_config()
config.paths.mkdirs()

assert config.is_kaggle, f"expected the Kaggle overrides, got env={config.env!r}"
assert config.ingest.delete_raw_after_ingest, "raw files must be deleted as we go"

env = record_hardware(config.paths.environment_json)
print(describe_environment(env))
print()
print("raw (scratch, discarded at session end):", config.paths.raw)
print("interim (persisted via Save Version)   :", config.paths.interim)
print("processed                              :", config.paths.processed)
print("expected matrix                        :", config.dataset.matrix_shape)
"""
    ),
    md(
        """
## 4. Guestbook credentials

Read from Kaggle Secrets into environment variables. `src.download` reads the
same `DATAVERSE_GB_*` variables locally from a gitignored `.env`, so the code
path is identical in both places and no credential is ever written to disk here.
"""
    ),
    code(
        """
import argparse
import os

from kaggle_secrets import UserSecretsClient

from src.download import GuestbookResponse

secrets = UserSecretsClient()
for key in ("DATAVERSE_GB_NAME", "DATAVERSE_GB_EMAIL",
            "DATAVERSE_GB_INSTITUTION", "DATAVERSE_GB_POSITION"):
    try:
        os.environ[key] = secrets.get_secret(key)
    except Exception as exc:
        raise RuntimeError(
            f"Secret {key} is not set. Add it under Add-ons -> Secrets. ({exc})"
        ) from exc

guestbook = GuestbookResponse.from_args_or_env(
    argparse.Namespace(gb_name=None, gb_email=None, gb_institution=None, gb_position=None)
)
print(f"guestbook: {guestbook.name} <{guestbook.email}>, {guestbook.institution}")
"""
    ),
    md(
        """
## 5. Resume from a previous run, if there is one

`/kaggle/working` survives only as a committed version's output. To continue an
interrupted run, attach that previous version's output as an input dataset and
this cell seeds the already-finished blocks back in, so they are skipped rather
than re-downloaded.

On a first run it finds nothing and does nothing.
"""
    ),
    code(
        """
import shutil
from pathlib import Path

from src.build_matrix import expected_days
from src.ingest import day_is_complete

seeded = 0
for candidate in sorted(Path("/kaggle/input").glob("*/interim")):
    for block in candidate.glob("*"):
        target = config.paths.interim / block.name
        if not target.exists():
            shutil.copy2(block, target)
            seeded += 1

complete = [d for d in expected_days(config) if day_is_complete(d, config.paths.interim)]
print(f"seeded {seeded} file(s) from a previous run")
print(f"{len(complete)} of {config.dataset.n_days} day(s) already ingested")
"""
    ),
    md(
        """
## 6. Stream: download → ingest → delete, one day at a time

Each day is fetched, aggregated over country codes into a dense
`144 x 10000` block, written to `/kaggle/working/interim`, and its raw text
deleted before the next begins.

Expect the download to dominate; the ingest itself is around a second per day.
"""
    ),
    code(
        """
from src.pipeline import run_streaming

result = run_streaming(
    config,
    guestbook,
    strategy=STRATEGY,
    limit=LIMIT,
    show_progress=False,          # progress bars are noise in a committed log
    time_budget_s=TIME_BUDGET_MIN * 60,
)
"""
    ),
    md(
        """
## 7. Assemble the matrix

Stacks the 62 daily blocks into one pre-allocated array, so peak memory is one
matrix plus one day rather than two copies of the matrix.

This step is strict on purpose. It fails rather than proceeding if a day is
missing, if any local day is not exactly 144 intervals, if the timestamp grid
has a gap or duplicate, or if an unfillable gap lands inside the 16–22 December
test week. A quiet failure here would corrupt every downstream metric while
leaving the shape correct.
"""
    ),
    code(
        """
from src.build_matrix import (
    apply_missing_policy,
    assemble_matrix,
    expected_days,
    save_matrix,
    to_local,
)
from src.memory_profiling import format_bytes, measure

missing = [d for d in expected_days(config) if not day_is_complete(d, config.paths.interim)]
if missing:
    raise SystemExit(
        f"{len(missing)} day(s) still outstanding (first: {missing[0]}). "
        "Save this version, attach its output as an input dataset, and re-run to resume."
    )

with measure("assemble", trace_python_allocs=False) as mem:
    matrix, utc_times, square_ids, report = assemble_matrix(config)
    local_times = to_local(utc_times, config.dataset.timezone)
    matrix = apply_missing_policy(matrix, local_times, config, report)
    paths = save_matrix(matrix, utc_times, square_ids, report, config)

print(f"shape             : {report.shape}")
print(f"local range       : {report.first_timestamp_local} .. {report.last_timestamp_local}")
print(f"absent cells      : {report.total_absent_cells:,}")
print(f"missing intervals : {len(report.missing_intervals)}")
print(f"interpolated      : {len(report.interpolated_intervals)}")
print(f"left as NaN       : {len(report.unfilled_intervals)}")
print(f"total activity    : {report.total_internet:,.2f}")
print(f"peak RSS          : {format_bytes(mem.rss_peak_delta)}   wall {mem.wall_s:.1f}s")
"""
    ),
    md("## 8. Memory-management evidence"),
    code(
        """
# Benchmarks the three ingest strategies, each in its own process so that
# allocator reuse cannot make whichever runs second look artificially cheap.
# Needs raw files, so it re-fetches one day; skip if you are short on time.
RUN_BENCHMARK = True

if RUN_BENCHMARK:
    from src.download import DataverseClient, download_file
    from src.ingest import raw_day_files
    from src.memory_report import run_benchmark

    client = DataverseClient(config.dataset.dataverse_base)
    entry = sorted(client.list_files(config.dataset.doi_traffic), key=lambda f: f.filename)[0]
    download_file(client, entry, config.paths.raw, guestbook, show_progress=False)

    run_benchmark(raw_day_files(config.paths.raw)[:1], config)

    for leftover in raw_day_files(config.paths.raw):
        leftover.unlink()
"""
    ),
    md("## 9. Verify what will be published"),
    code(
        """
import numpy as np
import polars as pl

m = np.load(config.paths.processed / "traffic_matrix.npy", mmap_mode="r")
ts = pl.read_parquet(config.paths.processed / "timestamps.parquet")
totals = pl.read_parquet(config.paths.processed / "totals.parquet")

assert m.shape == config.dataset.matrix_shape, m.shape
assert m.dtype == np.float32
assert ts.height == m.shape[0]
assert totals.height == config.dataset.n_squares

top = totals.sort("total_internet", descending=True).head(5)
print("top 5 squares by total activity over the full period:")
for row in top.iter_rows(named=True):
    print(f"  square {row['square_id']:>5}: {row['total_internet']:,.0f}")

print()
for square in config.areas.fixed:
    value = totals.filter(pl.col("square_id") == square)["total_internet"][0]
    rank = totals.filter(pl.col("total_internet") > value).height + 1
    print(f"  square {square} (from the brief): {value:,.0f}  rank {rank}/10000")

print()
print("output files:")
for path in sorted(config.paths.processed.glob("*")):
    print(f"  {path.name:<24} {path.stat().st_size / 1024**2:>8.2f} MiB")
"""
    ),
    md(
        """
## 10. Free the scratch and publish

The repository clone and any stray raw files are removed so the saved output
contains only the artefacts — `/kaggle/working` is capped at 20 GB and the
output becomes a dataset others download.

### After this finishes

1. **Save Version → Save & Run All (Commit)**, and wait for it to complete.
2. On the finished version: **Output → New Dataset**, name it
   `milan-traffic-processed`.
3. In the EDA and modelling notebooks, attach that dataset and read from
   `/kaggle/input/milan-traffic-processed/`. Neither needs internet, raw data,
   or this notebook again.
"""
    ),
    code(
        """
import shutil
from pathlib import Path

shutil.rmtree(REPO_DIR, ignore_errors=True)
shutil.rmtree(Path(config.paths.raw), ignore_errors=True)

total = sum(p.stat().st_size for p in Path("/kaggle/working").rglob("*") if p.is_file())
print(f"/kaggle/working now holds {total / 1024**2:.1f} MiB (limit 20 GB)")
for path in sorted(Path("/kaggle/working").rglob("*")):
    if path.is_file():
        print(f"  {path.relative_to('/kaggle/working')}  {path.stat().st_size / 1024**2:.2f} MiB")
"""
    ),
]


def main() -> int:
    """Write the notebook to disk."""
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    # nbformat >= 4.5 requires a stable id per cell. Deriving them from the
    # index keeps them deterministic, so regenerating produces no spurious diff.
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"cell-{index:02d}"

    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
