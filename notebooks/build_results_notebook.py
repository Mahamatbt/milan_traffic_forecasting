"""Generate the results and failure-analysis notebook.

Generated rather than hand-edited, like the other three, so it cannot drift
from ``src/``. Every cell calls a tested function in ``src.diagnostics`` or
``src.plots`` and renders its output; nothing is computed here.

The notebook reads the committed prediction series and refits nothing. That is
what lets it run on a laptop: the LSTM columns came from a GPU session and
reproducing them locally would take about 31 hours.

Run from the repository root:

    python notebooks/build_results_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent / "04_results.ipynb"


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
# Results, comparison and failure analysis

Evaluates the three models and both baselines on the test week (16–22 Dec 2013)
across the three study areas, then asks where and why each one fails.

Runs from a clean clone on a laptop: it reads the committed prediction series in
`results/predictions/` and refits nothing. The LSTM columns were produced on a
GPU and reproducing them on a CPU would take roughly 31 hours.
"""
    ),
    md("## Setup"),
    code(
        """
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv

import numpy as np
import polars as pl

from src.notebook import setup

ctx = setup()
config = ctx.config
AREAS = [5161, 5059, 5259]
"""
    ),
    md(
        """
## 1. Metric tables

Both baselines appear in every table. Persistence is the one that matters:
with a lag-1 autocorrelation of 0.987, repeating the last observation is a
demanding forecast, and a model that does not beat it has not earned its
complexity.

`lstm` is the mean of three seeds' errors; `lstm_ensemble` is the error of
their averaged forecast, which is the series stored in the predictions file
and drawn in every figure below.
"""
    ),
    code(
        """
for area in AREAS:
    table = pl.read_csv(config.paths.tables / f"final_metrics_area_{area}_test.csv")
    print(f"\\n=== square {area}, test week ===")
    with pl.Config(tbl_rows=20, tbl_width_chars=150):
        print(
            table.select("model", "mae", "mae_std", "rmse", "mape", "smape", "mase", "r2")
            .sort("mase")
        )
"""
    ),
    md(
        """
## 2. Cross-area comparison

MASE is the only metric comparable across these areas. Their traffic differs by
an order of magnitude, so an MAE of 83.6 on the busiest square and 13.6 on the
quietest say nothing about which was forecast better; each MASE is already
normalised by that area's own in-sample naive error.
"""
    ),
    code(
        """
from src.plots import plot_cross_area_mase

cross = pl.read_csv(config.paths.tables / "cross_area_mase_test.csv")
with pl.Config(tbl_width_chars=150):
    print(cross)

with (config.paths.tables / "cross_area_mase_test.csv").open(encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
plot_cross_area_mase(
    [{k: (float(v) if v not in ("", None) else None) if k != "model" else v
      for k, v in row.items()} for row in rows],
    areas=AREAS,
)
"""
    ),
    md(
        """
## 3. Actual versus predicted

Each figure shows the full test week above and the busiest single day below.
The zoom is not decoration: at full-week width a thousand points are drawn
across a few hundred pixels and every model traces the observed line. A
one-step lag only becomes visible at the lower scale, with persistence drawn
alongside for comparison.
"""
    ),
    code(
        """
from src.plots import plot_forecast
from src.run_results import load_predictions

for area in AREAS:
    columns = load_predictions(config, area, "test")
    for model in ("harmonic_arima", "lightgbm", "lstm"):
        display(
            plot_forecast(
                columns["local"],
                columns["observed"],
                columns[model],
                model=model,
                square_id=area,
                persistence=columns["persistence"],
                daily_period=config.dataset.daily_period,
            )
        )
"""
    ),
    md(
        """
## 4. Where the error falls

Absolute error scales with traffic level, so the error profile should track the
diurnal cycle. What is worth looking for is a model whose error peaks somewhere
the others' does not — that is a specific failure rather than a general one.
"""
    ),
    code(
        """
from src.diagnostics import error_by_hour
from src.plots import plot_error_by_hour

for area in AREAS:
    columns = load_predictions(config, area, "test")
    by_hour = {
        model: error_by_hour(columns["observed"], columns[model], columns["local"])
        for model in ("persistence", "harmonic_arima", "lightgbm", "lstm")
    }
    display(plot_error_by_hour(by_hour, square_id=area))
"""
    ),
    code(
        """
from src.diagnostics import error_heatmap
from src.plots import plot_error_heatmap

for area in AREAS:
    columns = load_predictions(config, area, "test")
    for model in ("harmonic_arima", "lightgbm", "lstm"):
        display(
            plot_error_heatmap(
                error_heatmap(columns["observed"], columns[model], columns["local"]),
                model=model,
                square_id=area,
            )
        )
"""
    ),
    md(
        """
### Weekday, weekend and holiday

The weekend column is where a model that has only learned the working-week
pattern shows it. The holiday column is reported separately because holidays
are the failure mode the stress split exists to expose.
"""
    ),
    code(
        """
daytype = pl.read_csv(config.paths.tables / "error_by_daytype_test.csv")
with pl.Config(tbl_rows=30, tbl_width_chars=160):
    print(
        daytype.select(
            "square_id", "model", "weekday_mae", "weekend_mae", "weekend_penalty", "n_weekend"
        )
    )
"""
    ),
    md(
        """
## 5. Residual autocorrelation

Structure left in the residuals is signal the model did not use. A spike at lag
144 means the daily shape is still not fully captured; a large value at lag 1
means the errors are serially dependent and the forecast could be improved from
its own recent errors alone.
"""
    ),
    code(
        """
from src.diagnostics import residual_acf
from src.plots import plot_residual_acf

for area in AREAS:
    columns = load_predictions(config, area, "test")
    acfs = {
        model: residual_acf(columns["observed"], columns[model])
        for model in ("harmonic_arima", "lightgbm", "lstm")
    }
    display(plot_residual_acf(acfs, square_id=area, daily_period=config.dataset.daily_period))
"""
    ),
    md(
        """
## 6. The copying check

With a lag-1 autocorrelation of 0.987, a model can post a respectable error by
learning to repeat its most recent input. It would look successful while having
learned nothing, so the plan asks for this to be tested explicitly and reported
honestly either way.

**Reading the cross-correlation correctly matters here.** A one-step forecast is
built only from observations up to `t-1`, so it cannot contain the innovation at
`t` and will correlate slightly *more* with the previous observation than the
current one. A peak at lag +1 is therefore what a causal forecast looks like,
not evidence of copying. The data makes the point: seasonal naive is the only
model that peaks at lag 0, and it is comfortably the worst forecaster here.

The verdict is therefore decided by `copy_ratio` — the distance between the
forecast and the persistence baseline, scaled by how far the series itself moves
between steps. Zero means the two are the same forecast.
"""
    ),
    code(
        """
from src.diagnostics import cross_correlation
from src.plots import plot_cross_correlation

copying = pl.read_csv(config.paths.tables / "copying_test.csv")
with pl.Config(tbl_rows=30, tbl_width_chars=160):
    print(copying)

for area in AREAS:
    columns = load_predictions(config, area, "test")
    correlations = {
        model: cross_correlation(columns["observed"], columns[model])
        for model in ("persistence", "seasonal_naive", "harmonic_arima", "lightgbm", "lstm")
    }
    display(plot_cross_correlation(correlations, square_id=area))
"""
    ),
    md(
        """
## 7. Failure analysis

### Worst contiguous windows

Aggregate error hides episodes: a model can post a respectable weekly MAE while
being badly wrong for six hours. Windows are non-overlapping, because the top
few of a naive scan are otherwise the same episode shifted one step at a time.
`ratio_to_persistence` above 1 means the baseline would have been better over
exactly that stretch.
"""
    ),
    code(
        """
failures = pl.read_csv(config.paths.tables / "failure_windows_test.csv")
with pl.Config(tbl_rows=50, tbl_width_chars=170):
    print(
        failures.filter(pl.col("model").is_in(["harmonic_arima", "lightgbm", "lstm"]))
        .sort("ratio_to_persistence", descending=True)
        .head(12)
    )
"""
    ),
    md(
        """
### The stress split

23 Dec – 1 Jan, never tuned on and never used for selection, holding four of the
eight Italian public holidays in the study period. The two baselines move in
opposite directions here, which is the clearest evidence of what the holidays do
to the series: persistence gets *easier* because the traffic is smoother, while
seasonal naive collapses past MASE 1.0 because the weekly pattern it depends on
is exactly what Christmas and New Year destroy.
"""
    ),
    code(
        """
for area in AREAS:
    table = pl.read_csv(config.paths.tables / f"final_metrics_area_{area}_stress.csv")
    print(f"\\n=== square {area}, stress split (23 Dec - 1 Jan) ===")
    with pl.Config(tbl_rows=20, tbl_width_chars=150):
        print(table.select("model", "mae", "mase", "r2", "lag1_copy_ratio").sort("mase"))
"""
    ),
    code(
        """
for area in AREAS:
    columns = load_predictions(config, area, "stress")
    for model in ("harmonic_arima", "lightgbm", "lstm"):
        display(
            plot_forecast(
                columns["local"],
                columns["observed"],
                columns[model],
                model=f"{model} (stress)",
                square_id=area,
                persistence=columns["persistence"],
                daily_period=config.dataset.daily_period,
            )
        )
"""
    ),
    md(
        """
## 8. Regenerating everything

`python run.py results --split test` and `--split stress` write every figure and
table above to `results/`. The notebook renders them; it computes nothing that
`src/` does not.
"""
    ),
    code(
        """
summary = json.loads(
    (config.paths.tables / "results_summary_test.json").read_text(encoding="utf-8")
)
print(f"figures: {len(summary['figures'])}")
print(f"tables : {len(summary['tables'])}")
for model, verdict in summary["copying_verdicts"].items():
    print(f"  {model:<16} {verdict}")
"""
    ),
]


def build() -> dict:
    """Assemble the notebook document."""
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
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
