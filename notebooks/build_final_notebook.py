"""Generate the Kaggle GPU notebook for the final fits.

Generated rather than hand-edited, like the other three.

Everything runs here rather than only the LSTM, and that is deliberate. The
final fits produce the timing table for requirement IV, and a table whose rows
were measured on different machines compares the machines as much as the
models. Running all three in one session makes the comparison a comparison.
The device column still distinguishes the LSTM's GPU fits from the two CPU
models, because that difference is a finding rather than an inconsistency.

Run from the repository root:

    python notebooks/build_final_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent / "03_kaggle_final.ipynb"


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
# Milan traffic forecasting — final fits and test-week evaluation

Refits every selected model on train **plus** validation and evaluates on the
held-out test week (16–22 Dec 2013) across all three study areas. Writes the
per-area metric tables, the timing table, and the prediction series the Phase 6
figures are drawn from.

### Why all three models run here, not just the LSTM

The timing table is a deliverable. Measuring harmonic ARIMA on an 8-core laptop
and the LSTM on a T4 would produce a table that compares hardware as much as
models. Running all three in one session removes that confound. The `device`
column still separates the LSTM's GPU fits from the two CPU models, because
*that* difference is the finding: on CPU a single LSTM fit took 13.8 h against
2 s for LightGBM.

### What the final fit is allowed to see

Hyperparameters were chosen on validation, so the refit uses train + validation
and the test week stays untouched until prediction. Neither model that stops
early has a held-out set left, so each carries the *complexity* tuning chose:
the LSTM trains for `best_epoch + 1` epochs with patience disabled, LightGBM
uses the `best_iteration` tree count with early stopping off. Both numbers were
fixed before this notebook runs.

The LSTM is run over **3 seeds** and reported as mean ± std, because a single
seed reports one draw and the spread is often comparable to the gap between
models.

### Before running

1. **Accelerator → GPU**. Asserted below rather than silently falling back.
2. **Settings → Internet → On**, for the clone and pip install.
3. No dataset to attach — a clean clone is the whole input.
4. Prefer **Save & Run All (Commit)** so `/kaggle/working` persists.

### What to bring back

`results/tables/` (the metric and timing tables), `results/predictions/`, and
**`results/environment.json`**.

That last one is easy to skip and must not be. The timing table records
`device` as `cpu` or `cuda`, but not *which* CPU or GPU, and the local
`environment.json` describes a Windows laptop with no CUDA device at all. Left
behind, the repository ends up attributing GPU timings to a machine that
cannot produce them.
"""
    ),
    md("## 1. Settings"),
    code(
        """
# Point this at your own fork/clone before running.
REPO_URL = "https://github.com/Mahamatbt/milan_traffic_forecasting.git"
BRANCH = "main"

# The test week is the reported result. "stress" (23 Dec - 1 Jan) is the
# held-out holiday period used for failure analysis in Phase 6.
SPLITS = ("test", "stress")
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
# old ones, so the kernel would keep executing the previous version.
for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[name]
importlib.invalidate_caches()

print("repo:", subprocess.run(
    ["git", "-C", str(REPO_DIR), "log", "-1", "--format=%h %s"],
    capture_output=True, text=True).stdout.strip())

_check = importlib.import_module("src.final_runs")
assert hasattr(_check, "MODEL_ORDER"), (
    "stale src/ still loaded: restart the kernel "
    "(Run -> Restart & Clear Cell Outputs), then run all cells again"
)
print("src version: OK")
"""
    ),
    md("## 3. Confirm the GPU and record the hardware"),
    code(
        """
import torch

from src.config import load_config
from src.models.lstm import resolve_device
from src.timing import describe_environment, record_hardware

device = resolve_device("auto")
assert device.type == "cuda", (
    f"resolved device is {device!r}, not CUDA. Set Accelerator -> GPU and restart. "
    "On CPU the nine LSTM fits take roughly 31 hours."
)

config = load_config()
config.paths.mkdirs()
assert config.is_kaggle, f"expected the Kaggle overrides, got env={config.env!r}"

env = record_hardware(config.paths.environment_json)
print(describe_environment(env))
print("gpu:", torch.cuda.get_device_name(0))
"""
    ),
    md(
        """
## 4. Stage the committed inputs

The series, the area selection and the tuned hyperparameters all live in the
repository. Staging them into the configured paths lets `src.final_runs` run
here exactly as it does locally, with no Kaggle-specific branching in `src/`.
"""
    ),
    code(
        """
staged = [
    (REPO_DIR / "data/processed/selected_series.parquet",
     config.paths.processed / "selected_series.parquet"),
    (REPO_DIR / "results/tables/selected_areas.json",
     config.paths.tables / "selected_areas.json"),
    (REPO_DIR / "results/tables/selected_hyperparameters.json",
     config.paths.tables / "selected_hyperparameters.json"),
]

for source, destination in staged:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"staged  {source.name:<34} -> {destination}")

import json

selected = json.loads(
    (config.paths.tables / "selected_hyperparameters.json").read_text(encoding="utf-8")
)
missing = [m for m in ("harmonic_arima", "lstm", "lightgbm") if m not in selected]
assert not missing, f"no tuned hyperparameters for {missing}; run the tuning stages first"

# The stopping points must have survived tuning, or the refits silently train to
# their ceilings instead of the complexity that was selected.
assert selected["lightgbm"].get("best_iteration", 0) > 0, "LightGBM has no best_iteration"
assert selected["lstm"].get("best_epoch", -1) >= 0, "LSTM has no best_epoch"

print("\\ntuned on area", selected["tuning_area"])
print("lstm best_epoch      :", selected["lstm"]["best_epoch"])
print("lightgbm best_iter   :", selected["lightgbm"]["best_iteration"])
print("harmonic order       :", selected["harmonic_arima"]["order"])
"""
    ),
    md(
        """
## 5. Final fits on the test week

Three areas x (2 baselines + 3 models), with the LSTM repeated over three
seeds. `walk_forward` with true observed history is the only inference path —
not a recursive rollout — so each forecast uses real observations up to the
step before the one it predicts.
"""
    ),
    code(
        """
import time

from src.final_runs import run_final

started = time.perf_counter()
results = run_final(config, split_name="test")
print(f"\\ntest-week fits: {(time.perf_counter() - started) / 60:.1f} min")
"""
    ),
    md(
        """
## 6. The held-out stress split

23 Dec – 1 Jan, never tuned on and never used for selection. The baselines move
in opposite directions here: persistence gets *easier* over the holidays while
seasonal-naive collapses past MASE 1.0, because the weekly pattern it depends on
is exactly what Christmas and New Year destroy. This is where the LightGBM
prediction — that a tree ensemble cannot extrapolate outside its training range
— is tested directly.
"""
    ),
    code(
        """
started = time.perf_counter()
stress = run_final(config, split_name="stress")
print(f"\\nstress-split fits: {(time.perf_counter() - started) / 60:.1f} min")
"""
    ),
    md("## 7. Results and timing"),
    code(
        """
import polars as pl

for split in SPLITS:
    path = config.paths.tables / f"final_metrics_all_{split}.csv"
    if not path.exists():
        continue
    print(f"\\n===== {split} =====")
    table = pl.read_csv(path).select(
        "square_id", "model", "device", "n_seeds", "mae", "mae_std",
        "mase", "mase_std", "r2", "lag1_copy_ratio", "train_wall_s",
    )
    with pl.Config(tbl_rows=40, tbl_width_chars=160):
        print(table.sort(["square_id", "mase"]))
"""
    ),
    code(
        """
timing = pl.read_csv(config.paths.tables / "timing_test.csv")
with pl.Config(tbl_rows=40, tbl_width_chars=160):
    print(timing.sort(["square_id", "train_wall_s"], descending=[False, True]))
"""
    ),
    md(
        """
## 8. Confirm what to download

The metric tables, the timing table and the prediction series. Phase 6 draws
every figure from these, so nothing further needs to be re-run to produce them.
"""
    ),
    code(
        """
written = sorted(config.paths.tables.glob("final_metrics_*.csv"))
written += sorted(config.paths.tables.glob("timing_*.csv"))
written += sorted(config.paths.predictions.glob("*.parquet"))
# The hardware record belongs with the timings it explains, not beside them.
written += [config.paths.environment_json]

for path in written:
    print(f"{path.stat().st_size:>10,}  {path.relative_to(config.paths.results.parent)}")
print(f"\\n{len(written)} files to download from the committed version's Output tab.")
print("environment.json is the one most easily missed: without it the cuda rows")
print("in timing_*.csv have no machine attached to them.")
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
