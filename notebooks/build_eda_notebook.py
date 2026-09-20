"""Generate the exploratory analysis notebook.

Generated rather than hand-edited for the same reason as the ingest notebook:
it cannot then drift from ``src/``. Every cell calls a tested function and
renders its output; nothing is computed here that is not computed by
``python run.py eda``.

The markdown records *what was measured and which modelling decision it drives*.
It is not report prose -- the write-up is the author's work, and the assignment
is explicit that it must be.

Run from the repository root:

    python notebooks/build_eda_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent / "02_eda.ipynb"


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
# Exploratory analysis

Characterises the Milan traffic data and records the evidence behind each
modelling decision taken in Section 4.

Reads only `data/processed/`, so it runs from a clean clone without the 19.4 GiB
download. `traffic_matrix.npy` (341 MB) is needed for the full analysis; the
committed `selected_series.parquet` alone is enough for everything from
"Study areas" onward.
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

import numpy as np
import polars as pl

from src.notebook import setup

ctx = setup()
config = ctx.config
print(ctx.describe())
"""
    ),
    code(
        """
from src.run_eda import load_inputs

data = load_inputs(config)
totals = data["totals"]["total_internet"].to_numpy()
local_times = data["local_times"]

print(f"matrix      : {data['matrix'].shape}  {data['matrix'].dtype}")
print(f"period      : {local_times[0]} .. {local_times[-1]}")
print(f"squares     : {data['square_ids'].size:,}")
"""
    ),
    md(
        """
## 1. How traffic is distributed across the 10,000 cells

Both views use log axes. Cell totals span orders of magnitude, so a linear
histogram puts almost the whole grid in the first bin; the CCDF on log-log axes
shows how the tail decays.
"""
    ),
    code(
        """
from src.eda import distribution_stats
from src.plots import plot_total_distribution

stats = distribution_stats(totals)
for line in stats.summary_lines():
    print(" ", line)

plot_total_distribution(totals, stats);
"""
    ),
    md(
        """
**Measured.** Gini 0.608; the top 1% of cells carry 11.0% of all traffic and the
top 10% carry 48.4%, while the bottom half carries 11.6%. Skewness 4.26,
kurtosis 25.50, max/median 45.8x.

**On the lognormal fit.** KS rejects it at p = 4.3e-05, but that is a
sample-size effect rather than a poor fit: the statistic is 0.0232 against a
critical value of 0.0136 at n = 10,000, and log(x) has skewness +0.023 and
kurtosis +0.013. Quoting the rejection without the statistic would misrepresent
it.

**Modelling implication.** Traffic is concentrated but not pathologically so.
Because volume varies by more than an order of magnitude between the study
areas, raw MAE and RMSE are not comparable across them, so the cross-area
comparison in Section 4 uses MASE.
"""
    ),
    md("## 2. Where those cells are"),
    code(
        """
from src.eda import grid_position, select_areas, totals_as_grid
from src.plots import plot_spatial_totals

areas = select_areas(totals, data["square_ids"], config)
for square in areas.all_plotted:
    row, col = grid_position(square)
    print(
        f"  square {square:>5}  rank {areas.ranks[str(square)]:>5}/10000  "
        f"total {areas.totals[str(square)]:>13,.0f}  grid ({row:>2}, {col:>2})"
    )
print(f"\\nforecasting on: {areas.forecast}")

plot_spatial_totals(totals_as_grid(totals), highlight={s: f"#{areas.ranks[str(s)]}" for s in areas.all_plotted});
"""
    ),
    code(
        """
# How far apart are the three busiest cells?
import itertools

CELL_METRES = 235
for a, b in itertools.combinations(areas.top_three, 2):
    (ra, ca), (rb, cb) = grid_position(a), grid_position(b)
    chebyshev = max(abs(ra - rb), abs(ca - cb))
    print(f"  {a} to {b}: {chebyshev} cell(s) = {chebyshev * CELL_METRES} m")
"""
    ),
    md(
        """
**Measured.** The three highest-traffic cells (5161, 5059, 5259) lie within
470 m of each other, and the top ten fit inside a 3.05 x 2.35 km box in central
Milan.

**Modelling implication.** This is why Section 4 forecasts {5161, 5059, 5259}
rather than the top three. Neighbouring cells of one hotspot cannot answer how
performance varies across areas with different traffic characteristics; the
chosen set spans ranks 1, 424 and 109.
"""
    ),
    md("## 3. The five series, first fortnight"),
    code(
        """
from src.eda import area_summary, extract_series
from src.plots import plot_series_overlay, plot_series_panel
from src.run_eda import window_mask

series = extract_series(data["matrix"], data["square_ids"], areas.all_plotted)
mask = window_mask(local_times, config.dataset.start_date, 14)

plot_series_panel(
    series[mask], areas.all_plotted, local_times[mask],
    title="Internet traffic, 1-14 November 2013",
    labels={s: f"rank {areas.ranks[str(s)]}" for s in areas.all_plotted},
);
"""
    ),
    code(
        """
plot_series_overlay(
    series[mask], areas.all_plotted, local_times[mask],
    title="Normalised traffic, 1-14 November 2013",
);
"""
    ),
    code(
        """
summary = pl.DataFrame(area_summary(series, areas.all_plotted, local_times))
summary.select([
    "square_id", "mean", "std", "cv", "max",
    "peak_to_trough", "night_floor_over_mean", "weekend_over_weekday",
])
"""
    ),
    md(
        """
**What to read from the table.** The coefficient of variation and the night
floor matter more than absolute volume: a series that keeps a high floor
overnight is easier to forecast than one that collapses, whatever its scale.
The weekend-to-weekday ratio separates areas whose activity is driven by work
from those driven by residents.
"""
    ),
    md(
        """
## 4. Analysis one: multi-seasonal decomposition

MSTL with daily (144) and weekly (1008) periods. Classical decomposition takes
a single period and cannot represent both at once, which is itself an argument
about model choice.
"""
    ),
    code(
        """
from src.plots import plot_decomposition
from src.tsanalysis import decompose

top = areas.top_traffic
y = series[:, areas.all_plotted.index(top)].astype(np.float64)

decomposition = decompose(y, periods=(config.dataset.daily_period, config.dataset.weekly_period))
for line in decomposition.summary_lines():
    print(" ", line)
print(f"\\n  residual std / observed std = {np.std(decomposition.residual) / np.std(y):.4f}")

plot_decomposition(decomposition, local_times);
"""
    ),
    md(
        """
**Measured (square 5161).** Daily seasonality carries 82.9% of the variance
(strength 0.959), weekly 10.0% (0.746), trend 2.8%, residual 3.6%. The residual
standard deviation is 19% of the observed.

**Modelling implication.** About 81% of the variation is structure a model can
capture, and it is overwhelmingly the daily cycle. Two periods are present at
once, so a seasonal model needs to represent both: this is the argument for
Fourier terms at two frequencies rather than a single seasonal ARIMA, for which
s = 144 would in any case be intractable.
"""
    ),
    md(
        """
## 5. Analysis two: autocorrelation and stationarity

ADF and KPSS are reported together because their null hypotheses are opposites,
so agreement is informative and disagreement is diagnostic rather than hidden.
"""
    ),
    code(
        """
from src.tsanalysis import stationarity_table

rows = stationarity_table(y, seasonal_period=config.dataset.daily_period)
pl.DataFrame([r.to_row() for r in rows]).select(
    ["transform", "adf_stat", "adf_pvalue", "kpss_stat", "kpss_pvalue", "verdict"]
)
"""
    ),
    code(
        """
from src.plots import plot_acf_pacf
from src.tsanalysis import autocorrelation

acf = autocorrelation(y, nlags=config.dataset.weekly_period + 92)
print(f"  lag-1 ACF          : {acf['lag_1']:.4f}")
print(f"  ACF at 144 (1 day) : {acf['acf'][config.dataset.daily_period]:.4f}")
print(f"  ACF at 1008 (1 wk) : {acf['acf'][config.dataset.weekly_period]:.4f}")
print(f"  notable lags       : {acf['notable_lags'][:8]}")

plot_acf_pacf(acf, daily_period=config.dataset.daily_period, weekly_period=config.dataset.weekly_period);
"""
    ),
    code(
        """
from src.plots import plot_rolling_stats
from src.tsanalysis import rolling_stats

plot_rolling_stats(rolling_stats(y, window=config.dataset.daily_period), local_times);
"""
    ),
    md(
        """
**Measured.** Both tests call the raw series stationary, and its first and
seasonal differences too. Lag-1 ACF is 0.987, with local maxima at 144, 288,
432, 720, 864 and 1008 -- every multiple of the daily period, and the weekly
lag among them.

**A caveat the tests do not cover.** "Stationary" here means no stochastic
trend. The mean still varies enormously with time of day, which neither ADF nor
KPSS tests for. The series is strongly seasonally structured, and reporting it
as simply stationary would be misleading.

**Modelling implication.** No differencing is required for a stochastic trend,
so d = 0. The very high lag-1 correlation means persistence alone is a strong
predictor, which is exactly why Section 4 reports a persistence baseline: any
model that fails to beat it has learnt nothing. The ACF peaks confirm the lag
set used for the gradient-boosted model.
"""
    ),
    md("## 6. Supporting: which cycles actually carry power"),
    code(
        """
from src.plots import plot_periodogram
from src.tsanalysis import periodogram_peaks

spectrum = periodogram_peaks(y, interval_minutes=config.dataset.interval_minutes, top_n=8)
for peak in spectrum["peaks"][:5]:
    print(f"  {peak['period_hours']:>9.2f} h   power {peak['power']:.3e}")

plot_periodogram(spectrum);
"""
    ),
    md(
        """
**Measured.** 24.00 h dominates, followed by a 12.00 h harmonic and a peak near
168 h.

**Modelling implication.** The 12 h harmonic is not an artefact: the daily cycle
is not a single sinusoid, so the harmonic regression needs at least K1 = 2
Fourier terms for the daily period. A single sine would fit the envelope and
miss the shape.
"""
    ),
    md(
        """
## 7. Supporting: where a seasonal-naive predictor fails

These are the intervals the models are expected to miss, identified in advance
so the failure analysis in Section 4 is a checked prediction rather than an
after-the-fact explanation.

The scale is estimated per position in the daily cycle. Forecast errors here are
strongly heteroscedastic -- the residual's spread varies 25x between the
quietest and busiest hour -- so a single global scale is set by the quiet hours
and flags every busy one, marking 12.7% of all points.
"""
    ),
    code(
        """
from src.holidays_it import HOLIDAYS, describe
from src.tsanalysis import seasonal_naive_anomalies

anomalies = seasonal_naive_anomalies(y, seasonal_period=config.dataset.daily_period)
print(f"  flagged {anomalies['n_flagged']} of {y.size} points ({anomalies['fraction_flagged']:.2%})")

flagged_dates = local_times[anomalies["index"]].astype("datetime64[D]")
counts = pl.DataFrame({"date": flagged_dates}).group_by("date").len().sort("len", descending=True)
counts.head(12)
"""
    ),
    code(
        """
import datetime as dt

from scipy import stats as sps

holiday_days = {h.date for h in HOLIDAYS}
on_holiday = sum(
    1 for d in flagged_dates.astype(dt.date) if d in holiday_days
)
share = len(holiday_days) / config.dataset.n_days
p = sps.binomtest(on_holiday, len(flagged_dates), share, alternative="greater").pvalue

print(f"  holidays            : {len(holiday_days)}/62 days ({share:.1%} of the period)")
print(f"  flagged on holidays : {on_holiday}/{len(flagged_dates)} ({on_holiday / len(flagged_dates):.1%})")
print(f"  enrichment          : {(on_holiday / len(flagged_dates)) / share:.2f}x   binomial p = {p:.2e}")
"""
    ),
    md(
        """
**Measured.** 152 intervals flagged (1.73%). Holidays are 12.9% of the period
but carry 23.0% of the flags -- a 1.78x enrichment, binomial p = 4.3e-04. Both
Milan-specific dates appear: Sant'Ambrogio (7 December) and Immacolata
(8 December).

**One unexplained case.** The single worst day is Monday 2 December with 17
flagged intervals, and it is not a holiday. Carried forward to the failure
analysis rather than explained away here.

**Modelling implication.** Holiday effects are real and a model given only
day-of-week cannot anticipate them, so an `is_holiday` calendar feature is
justified. The stress split (23 December - 1 January) contains four of the eight
holidays, which is why it is held out for failure analysis and never tuned on.
"""
    ),
    md(
        """
## 8. Persist the series used by the modelling stages
"""
    ),
    code(
        """
frame = pl.DataFrame(
    {
        "utc": data["utc_times"],
        "local": local_times,
        **{f"square_{s}": series[:, i] for i, s in enumerate(areas.all_plotted)},
    }
)
path = config.paths.processed / "selected_series.parquet"
frame.write_parquet(path, compression="zstd")
areas.save(config.paths.tables / "selected_areas.json")

print(f"{path}  ({path.stat().st_size / 1024:.0f} KB, committed to the repository)")
frame.head()
"""
    ),
    md(
        """
---

Every figure and table here is also produced headlessly by:

```bash
python run.py eda
```

which writes `results/figures/eda/*.png` at 300 dpi and the tables under
`results/tables/`.
"""
    ),
]


def main() -> int:
    """Write the notebook to disk."""
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"cell-{index:02d}"

    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
