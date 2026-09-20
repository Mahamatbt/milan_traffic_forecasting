"""Generate the Kaggle GPU notebook for the LSTM sweep.

Generated rather than hand-edited, for the same reason as the other two: a
hand-maintained .ipynb drifts from ``src/`` and the "logic lives in src/" rule
stops meaning anything.

This notebook exists because of a measurement, not a preference. The LSTM sweep
was started locally and abandoned after 16 hours with 2 of 5 stages done. A
single fit at ``hidden_size=128, num_layers=2, sequence_length=288`` took 13.8
hours, and the process averaged 0.88 of 8 cores throughout: a 288-step
recurrence over small matrices is latency-bound on a sequential dependency that
CPU threads cannot break up. Projected to completion the sweep was about six
days. The other two models finish locally in minutes and stay there.

Run from the repository root:

    python notebooks/build_lstm_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent / "01_kaggle_lstm.ipynb"


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
# Milan traffic forecasting — LSTM hyperparameter sweep

Runs the staged LSTM search on a GPU and writes the selected configuration back
into `results/tables/selected_hyperparameters.json`, merging with the harmonic
ARIMA and LightGBM selections already made locally.

### Why this one model runs here

The sweep was first run on the development machine and abandoned after 16 hours
with 2 of 5 stages complete:

| Stage | Candidates | Wall time |
|---|---:|---:|
| `sequence_length` | 4 | 36 min |
| `capacity` | 6 | **15 h** |
| remaining three | 12 | ~6 days, projected |

The winning capacity fit alone took 13.8 h, and the process held an average of
0.88 of 8 cores the whole time. That is the signature of a sequential
bottleneck rather than an undersized machine: a 288-step recurrence over
64x128 matrices cannot be spread across CPU threads, because each step depends
on the one before it. A GPU parallelises the batch dimension instead, which is
the dimension that is actually wide here.

Harmonic ARIMA and LightGBM are not affected and are tuned locally — together
they take under ten minutes.

### Before running

1. **Accelerator → GPU** (T4 or P100). The notebook asserts CUDA is present and
   stops rather than silently spending hours on the CPU.
2. **Settings → Internet → On**, needed for the `git clone` and `pip install`.
3. No Kaggle Dataset needs attaching. `data/processed/selected_series.parquet`
   and `results/tables/selected_areas.json` are committed, and tuning reads only
   the one study area — so a clean clone is the entire input.
4. Prefer **Save & Run All (Commit)**: `/kaggle/working` is only persisted by a
   committed version.

### What to bring back

Two files, from the committed version's output:
`results/tables/selected_hyperparameters.json` and `results/experiments.csv`.
"""
    ),
    md("## 1. Settings"),
    code(
        """
# Point this at your own fork/clone before running.
REPO_URL = "https://github.com/Mahamatbt/milan_traffic_forecasting.git"
BRANCH = "main"

# Only the LSTM is tuned here. The other two are selected locally and their
# entries in selected_hyperparameters.json are preserved, not overwritten.
MODELS = ("lstm",)
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
_check = importlib.import_module("src.models.lstm")
assert hasattr(_check, "resolve_device"), (
    "stale src/ still loaded: restart the kernel "
    "(Run -> Restart & Clear Cell Outputs), then run all cells again"
)
print("src version: OK")
"""
    ),
    md(
        """
## 3. Confirm the GPU is actually present

The whole reason this notebook exists is the device. Running it on a CPU
accelerator would not fail — it would quietly take days — so the assertion is
worth more than the convenience of falling back.
"""
    ),
    code(
        """
import torch

from src.models.lstm import resolve_device

device = resolve_device("auto")
assert device.type == "cuda", (
    f"resolved device is {device!r}, not CUDA. Set Accelerator -> GPU in the "
    "sidebar and restart. On a CPU this sweep takes roughly six days."
)
print("device        :", device)
print("gpu           :", torch.cuda.get_device_name(0))
print("torch         :", torch.__version__)
print("capability    :", torch.cuda.get_device_capability(0))
"""
    ),
    md(
        """
## 4. Configuration and hardware record

`src.config` detects the Kaggle session and layers `config/kaggle.yaml` over the
defaults. The hardware snapshot is recorded before any fitting, because the
training times reported in the results table are only meaningful alongside the
device that produced them.
"""
    ),
    code(
        """
from src.config import load_config
from src.timing import describe_environment, record_hardware

config = load_config()
config.paths.mkdirs()

assert config.is_kaggle, f"expected the Kaggle overrides, got env={config.env!r}"

env = record_hardware(config.paths.environment_json)
print(describe_environment(env))
"""
    ),
    md(
        """
## 5. Stage the committed inputs

Tuning reads three things that live in the repository rather than in
`/kaggle/working`: the extracted series, the study-area selection, and the
existing experiment log and selections to append to. They are copied into the
configured paths so `src.train` runs here exactly as it does locally, with no
Kaggle-specific branching in `src/`.

Copying the experiment log in is what keeps it append-only across machines: the
rows written here continue the local ones rather than starting a new file.
"""
    ),
    code(
        """
import shutil
from pathlib import Path

staged = [
    (REPO_DIR / "data/processed/selected_series.parquet",
     config.paths.processed / "selected_series.parquet"),
    (REPO_DIR / "results/tables/selected_areas.json",
     config.paths.tables / "selected_areas.json"),
    (REPO_DIR / "results/tables/selected_hyperparameters.json",
     config.paths.tables / "selected_hyperparameters.json"),
    (REPO_DIR / "results/experiments.csv", config.paths.experiments_csv),
]

for source, destination in staged:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copy2(source, destination)
        print(f"staged  {source.name:<34} -> {destination}")
    else:
        print(f"absent  {source.name:<34} (will be created)")

selections = config.paths.tables / "selected_hyperparameters.json"
assert selections.exists(), (
    "selected_hyperparameters.json is not in the clone. Run the local tuning "
    "first (`python run.py train --models harmonic_arima,lightgbm`) and commit "
    "the result, so this run has something to merge into."
)
"""
    ),
    md(
        """
## 6. Run the sweep

Five stages, searched one axis at a time and carrying the winner forward:
sequence length, capacity, optimisation, regularisation, and a calendar-feature
ablation. Every candidate appends its own row to the experiment log with the
numbers that justified keeping or rejecting it — not only the stage winner.
"""
    ),
    code(
        """
import json
import time

from src.train import train_all

started = time.perf_counter()
selected = train_all(config, models=MODELS)
elapsed = time.perf_counter() - started

print(f"\\nsweep wall time: {elapsed / 60:.1f} min")
print(json.dumps(selected.get("lstm", {}), indent=2))
"""
    ),
    md(
        """
## 7. Review what the search decided

The per-candidate rows are the evidence for the selection. A stage whose spread
across the axis is small is a stage whose parameter did not matter — which is
worth reporting, not hiding.
"""
    ),
    code(
        """
import polars as pl

rows = pl.read_csv(config.paths.experiments_csv)
lstm_rows = rows.filter(pl.col("model") == "lstm")

print(f"{len(lstm_rows)} LSTM rows logged\\n")
with pl.Config(fmt_str_lengths=60, tbl_rows=60):
    print(
        lstm_rows.select("stage", "valid_mae", "valid_mase", "n_params", "train_wall_s")
        .sort("valid_mae")
    )
"""
    ),
    md(
        """
## 8. Confirm what to download

These two files are the output of this notebook. Everything else here is
reproducible from them plus the repository.
"""
    ),
    code(
        """
for path in (config.paths.tables / "selected_hyperparameters.json",
             config.paths.experiments_csv):
    size = path.stat().st_size if path.exists() else 0
    print(f"{'OK ' if size else 'MISSING'} {path}  ({size:,} bytes)")

print("\\nDownload both from the committed version's Output tab, replace the")
print("local copies, and commit them. The final runs read selected_hyperparameters.json.")
"""
    ),
]


def build() -> dict:
    """Assemble the notebook document.

    Cell ids are required from nbformat 4.5 and are assigned from position, so
    regenerating an unchanged notebook produces an identical file and the drift
    check in ``tests/test_notebook.py`` stays meaningful.
    """
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"cell-{index:02d}"
    return notebook


def main() -> int:
    """Write the notebook to disk."""
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
