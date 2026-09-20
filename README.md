# Milan Mobile Network Traffic Forecasting

A comparative study of three sequential models for **one-step-ahead** (10-minute)
forecasting of mobile internet traffic in Milan, using the Telecom Italia Big Data
Challenge call-detail-record dataset (1 November 2013 – 1 January 2014).

**Research question.** How do different sequential models compare for one-step-ahead
mobile network traffic forecasting under extreme load, and how does their performance
vary across the network's busiest geographical hotspots?

> **Status: complete.** Ingest, exploratory analysis, model tuning, final fits, evaluation
> and failure analysis have all run. The write-up is **[`report/REPORT.md`](report/REPORT.md)**.
> A clean clone reproduces every reported metric exactly — see
> [Reproducing results](#reproducing-results-without-the-20-gb-download).

**Headline result.** The 22-parameter dynamic harmonic regression achieves the best mean MASE (0.202) across the top three hotspots, closely followed by the LSTM ensemble (0.202) and LightGBM (0.209). In this dense core, all three models consistently beat the persistence baseline (0.238). On the held-out holiday period, however, persistence wins outright on two of three areas while the learned models degrade significantly.

---

## Contents

- [The data](#the-data)
- [Study areas](#study-areas)
- [Where things run](#where-things-run)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [Reproducing results without the 20 GB download](#reproducing-results-without-the-20-gb-download)
- [Configuration](#configuration)
- [Repository layout](#repository-layout)
- [Notebooks](#notebooks)
- [Data handling and memory](#data-handling-and-memory)
- [Exploratory findings](#exploratory-findings)
- [Methodology](#methodology)
- [Results](#results)
- [Failure analysis](#failure-analysis)
- [Testing and engineering practices](#testing-and-engineering-practices)
- [The report](#the-report)
- [Licence and data terms](#licence-and-data-terms)
- [References](#references)

---

## The data

| Property | Value |
|---|---|
| Source | Harvard Dataverse `doi:10.7910/DVN/EGZHFV` |
| Grid geometry | `doi:10.7910/DVN/QJWLFU` (Milano Grid, GeoJSON) |
| Coverage | 2013-11-01 → 2014-01-01, 62 daily TSV files |
| Raw size | 19.38 GiB (20,804,803,507 B), 265–359 MiB per file |
| Raw records | 319,896,289, reducing to 89,280,000 cells |
| Spatial grid | 100 × 100 cells of 235 m, square ids 1–10000 |
| Temporal resolution | 10 minutes → 144 intervals/day → 8,928 timestamps |
| Target variable | `internet` (normalised CDR activity, **not** an integer count) |

Three properties of the raw files drive the whole design:

1. **Rows are split by country code.** A single `(square_id, time_ms)` pair appears once per
   counterparty country. Traffic must be aggregated with
   `group_by(["square_id", "time_ms"]).sum()` before use — this is why a day holds ~5.16 M
   rows rather than 1.44 M (3.6 country rows per cell on average, at most 36).
2. **After aggregation the data is small.** The full `8928 × 10000` matrix is 340.58 MiB as
   `float32`. The memory-management problem is entirely about getting from 19.38 GiB of text
   to that matrix without ever holding more than one day in RAM.
3. **Timestamps are epoch milliseconds in UTC, but files are aligned to *local* midnight.**
   The first record of the 1 November file is `1383260400000` = `2013-10-31 23:00 UTC` =
   `2013-11-01 00:00` CET. All conversion goes through `Europe/Rome`, or every diurnal
   pattern shifts by an hour.

**Data quality.** The assembled matrix contains zero missing intervals and zero `NaN`
values; the gap-interpolation policy never had to execute. 34,682 cells (0.0388%) have no
record, which means no activity was logged rather than data lost, and they are set to 0.0.
The three modelled areas contain no zero at any point in any split.

---

## Study areas

The network's most extreme load is concentrated in the absolute busiest cells, making them the most critical areas to forecast accurately. This study focuses specifically on the three highest-traffic cells in Milan, all located in a dense hotspot near the Duomo, to evaluate whether the models can handle the network's most demanding environments.

The areas modelled are therefore:

| Square | Rank | Mean | CV | Peak/trough | Night floor | Weekend ÷ weekday | Nearest landmark |
|---|---:|---:|---:|---:|---:|---:|---|
| **5161** | 1 | 1,427 | 0.968 | 99.4 | 0.130 | 1.384 | Galleria Vittorio Emanuele II (276 m) |
| **5059** | 2 | 1,251 | 0.768 | 30.9 | 0.235 | 0.861 | Duomo (226 m) |
| **5259** | 3 | 1,174 | 0.939 | 47.0 | 0.333 | 0.425 | Teatro alla Scala (167 m) |

These three areas represent the absolute peak volume of the Milan network. Evaluating models across them provides a robust stress test of their capacity to predict demand where resource allocation matters most.

---

## Where things run

This project is **hybrid** by design, split by measurement rather than assumption:

- **Locally** (Windows, Python 3.12, CPU): `src/`, `config/` and `tests/` are developed and
  version-controlled; harmonic ARIMA and LightGBM tuning; evaluation and all results
  figures. Both searches finish in minutes.
- **On Kaggle** (Linux, Tesla T4): the ingest, the LSTM sweep and the final fits. Notebooks
  `git clone` this repo and import from `src/`; they contain no model or preprocessing logic
  of their own.

Two measurements justify the split. The download ran at 0.44 MiB/s locally (≈12.6 h
projected) against 55.9 MiB/s on Kaggle (5.92 min) — a factor of ~127. And a 288-step
recurrence is latency-bound on a sequential dependency CPU threads cannot divide: locally
the process held 0.88 of 8 cores, and one LSTM configuration took **2,019 s on the CPU
against 2.6 s on the T4**.

Kaggle's disk budget shapes the ingest: `/kaggle/working` is ~20 GB and persisted, while
everything else is ~60 GB of scratch discarded at session end. The raw text therefore
streams into `/kaggle/temp/raw` and each file is **deleted as soon as its per-day block is
written**, holding peak disk near one file. Blocks are checkpointed so a 12-hour session
timeout resumes rather than restarts.

**Consequence for reporting:** `train_wall_s` is not comparable across models, so every
results and timing table carries a `device` column, and hardware is recorded in
`results/environment.json` (local) and `results/environment_final_runs.json` (the T4
session that produced the final fits).

---

## Setup

### Local

```bash
git clone https://github.com/Mahamatbt/milan_traffic_forecasting.git
cd milan_traffic_forecasting

py -V:3.12 -m venv .venv                       # Windows
.venv\Scripts\pip install -r requirements-local.txt

.venv\Scripts\python run.py test               # 501 tests
.venv\Scripts\python run.py env                # record hardware
```

On macOS/Linux substitute `python3.12 -m venv .venv` and `.venv/bin/pip`.

Python 3.12 is required: 3.14 lacks reliable wheels for torch, lightgbm and geopandas.

Three requirements files, for three purposes:

| File | Use |
|---|---|
| `requirements.txt` | Core pipeline, pinned. What a reproducer installs. |
| `requirements-local.txt` | Local development; CPU-only torch build. |
| `requirements-kaggle.txt` | What the notebooks install on Kaggle. |

### Kaggle

Each notebook's first markdown cell lists its own prerequisites. In summary:

1. **Settings → Internet → On** (requires a phone-verified account) — needed for the
   `git clone` and `pip install`.
2. **Accelerator:** *None* for the ingest (it is I/O-bound); **GPU** for
   `01_kaggle_lstm` and `03_kaggle_final`, both of which assert CUDA is present rather
   than silently falling back to a CPU run that would take days.
3. **Add-ons → Secrets** (ingest only): `DATAVERSE_GB_NAME`, `DATAVERSE_GB_EMAIL`,
   `DATAVERSE_GB_INSTITUTION`, `DATAVERSE_GB_POSITION`. Harvard Dataverse gates this
   dataset behind a Guestbook and records one response per download request, so these must
   be real details. They are read from Kaggle Secrets, never written into the notebook.
4. Edit `REPO_URL` in the settings cell to point at your fork.
5. **Save Version → Save & Run All (Commit).** `/kaggle/working` is only persisted by a
   committed version.

`src.config` detects the Kaggle session automatically and layers `config/kaggle.yaml` over
the defaults, so no path edits are needed.

After a modelling notebook finishes, `./merge_kaggle_output.sh <download dir>` copies the
files it produced back into the repo. It copies **only** those files and fails loudly if
any are missing — `results/tables/` also holds exploratory artefacts the Kaggle session
never had, several derived from the uncommitted 341 MB matrix, so replacing the directory
wholesale would cost a full re-ingest.

---

## Running the pipeline

`make` is not available on a default Windows install, so `run.py` is the canonical entry
point on both platforms. The `Makefile` mirrors the same targets for Linux.

```bash
python run.py env        # record hardware -> results/environment.json
python run.py download   # fetch the 62 daily files + grid GeoJSON
python run.py ingest     # raw text -> per-day blocks, with memory evidence
python run.py pipeline   # stream download -> ingest -> delete (what Kaggle runs)
python run.py benchmark  # compare ingest strategies in isolated processes
python run.py matrix     # assemble the 8928 x 10000 matrix
python run.py eda        # exploratory figures, statistics, selected series
python run.py train      # tune the models (--models to select a subset)
python run.py final      # refit on train+validation, evaluate on a split
python run.py results    # evaluation figures, diagnostics, failure analysis
python run.py evaluate   # baseline metrics for any split
python run.py facts      # regenerate the named fact registry
python run.py report     # export figures and tables into report/
python run.py pdf        # render report/REPORT.md as report/REPORT.pdf
python run.py test       # pytest
python run.py lint       # ruff
python run.py all        # everything above, in order
```

Any task accepts `--config` and forwards remaining arguments to its module:

```bash
python run.py ingest  --config config/kaggle.yaml --limit 3
python run.py train   --models harmonic_arima,lightgbm --gbm-trials 30
python run.py final   --split stress
python run.py results --split stress
```

`run.py train --models <names>` tunes a subset and **merges** into
`results/tables/selected_hyperparameters.json`, so a run on one machine never discards
another's selections, and it refuses to mix selections tuned on different areas. The
experiment log is append-only across both machines.

### Reproducing results without the 20 GB download

Four artefacts are committed so that everything after ingest reproduces from a clean clone:
`data/processed/selected_series.parquet` (the extracted series, 283 KB),
`results/tables/selected_areas.json`, `results/tables/selected_hyperparameters.json`, and
`results/predictions/*.parquet` (the forecasts, 312 KB).

```bash
git clone https://github.com/Mahamatbt/milan_traffic_forecasting.git && cd milan_traffic_forecasting
python -m venv .venv && .venv/Scripts/activate      # source .venv/bin/activate on Linux
pip install -r requirements.txt

python run.py test                     # 501 tests, ~90 s
python run.py evaluate --split test    # baseline metrics from the committed series
python run.py results  --split test    # 28 figures + 5 tables from the committed forecasts
python run.py results  --split stress  # the same for the holiday split
```

**Verified end to end:** a fresh clone into an empty virtualenv reproduces every reported
metric and every summary **exactly**, not merely to 3 significant figures. The only
differences between regenerated and committed artefacts are wall-clock timing columns and
line endings.

What this path cannot do, and why:

| Stage | Needs | Why it is excluded |
|---|---|---|
| `download`, `ingest`, `matrix` | 19.38 GiB of raw CDR text | Not redistributable; `data/raw/` is gitignored |
| parts of `eda` | `traffic_matrix.npy` (341 MB) | Derived from the raw text, too large to commit |
| `final` (LSTM rows) | a CUDA device | On CPU the nine LSTM fits take roughly 31 hours |

The committed prediction series exist precisely because of that last row: the LSTM columns
cannot be regenerated on a CPU in reasonable time, so without them the figures could not be
redrawn from a clean clone.

---

## Configuration

`config/default.yaml` is the single source of truth for paths, dates, split boundaries and
preprocessing defaults; `config/kaggle.yaml` overrides only what the Kaggle environment
changes and is always layered *over* the defaults rather than replacing them. Nothing in
`src/` hardcodes a path, date or hyperparameter.

Loading is strict — an unknown key is an error, not a silently ignored typo — and the
splits are validated as contiguous, non-overlapping and inside the observation period:

| Split | Dates (inclusive, `Europe/Rome`) | Points | Purpose |
|---|---|---:|---|
| train | 2013-11-01 → 2013-12-08 | 5,472 | Model fitting; all transforms fit here only |
| validation | 2013-12-09 → 2013-12-15 | 1,008 | Hyperparameter selection |
| test | 2013-12-16 → 2013-12-22 | 1,008 | Reported results (the week the brief fixes) |
| stress | 2013-12-23 → 2014-01-01 | 1,440 | Failure analysis only; never tuned on |

Splits are chronological, never random: a random split on a time series places future
observations in the training set.

---

## Repository layout

```
├── CLAUDE.md                  project brief and non-negotiables
├── Plan.md                    phase-by-phase implementation plan and status
├── README.md                  this file
├── LICENSE                    MIT
├── run.py                     cross-platform task runner
├── Makefile                   equivalent targets for Linux/Kaggle
├── merge_kaggle_output.sh     copy a Kaggle run's outputs back into the repo
├── config/
│   ├── default.yaml           single source of truth
│   └── kaggle.yaml            overlay, merged over the defaults
├── data/
│   ├── raw/                   gitignored (except the grid GeoJSON)
│   ├── interim/               gitignored, per-day blocks
│   └── processed/             selected_series.parquet committed; matrix gitignored
├── src/                       all logic — 36 modules
│   ├── config.py              strict YAML loader into frozen dataclasses
│   ├── download.py            Dataverse client: guestbook, resume, md5
│   ├── pipeline.py            streaming download -> ingest -> delete
│   ├── ingest.py              three ingest strategies
│   ├── build_matrix.py        assembly, timezone, missing-data policy
│   ├── memory_profiling.py    peak-RSS sampler
│   ├── memory_report.py       benchmark, one subprocess per measurement
│   ├── eda.py, tsanalysis.py  statistics: Gini, MSTL, ADF/KPSS, ACF, spectrum
│   ├── geo.py, holidays_it.py grid geometry; Italian + Milanese holidays
│   ├── features.py            scaler, windows, lag and calendar features
│   ├── splits.py, metrics.py  chronological splits; MAE/RMSE/MAPE/sMAPE/MASE/R²
│   ├── models/                baselines, harmonic ARIMA, LSTM, LightGBM
│   ├── train.py               staged hyperparameter search
│   ├── final_runs.py          refit on train+validation, seed ensembling
│   ├── evaluate.py            baseline metrics per split
│   ├── diagnostics.py         error structure, copying check, failure windows
│   ├── plots.py               figures, kept separate from the statistics
│   ├── run_eda.py             orchestrates the exploratory artefacts
│   ├── run_results.py         orchestrates the evaluation artefacts
│   ├── experiments.py         append-only experiment log
│   ├── facts.py               named fact registry
│   └── export_report.py       evidence pack for the write-up
├── notebooks/                 thin: import from src/, render figures only
│   ├── build_*_notebook.py    generators — the notebooks are output, not source
│   └── runs/                  executed notebooks, kept as evidence
├── tests/                     27 modules, 501 tests
├── results/
│   ├── experiments.csv        append-only log of all 101 tuning runs
│   ├── environment*.json      hardware and library versions per machine
│   ├── figures/               56 evaluation figures + eda/
│   ├── tables/                metrics, diagnostics, timing
│   └── predictions/           committed forecast series
└── report/
    ├── REPORT.md              the consolidated write-up
    ├── RESULTS_SUMMARY.md     uninterpreted dump of every measured number
    ├── FACTS.json / .md       133 named facts, each traceable to its artefact
    ├── FIGURE_INDEX.md        figure -> report section map
    ├── references.md          with per-reference verification notes
    ├── figures/               46 figures at 300 dpi
    └── tables/                50 tables
```

---

## Notebooks

Notebooks are **generated** from `notebooks/build_*_notebook.py` rather than hand-edited.
The generator is the source; the `.ipynb` is output. `tests/test_notebook.py` asserts that
every `src` symbol a notebook imports still exists, that it defines no functions or classes
of its own, that it imports no private names, and that it has not drifted from its
generator.

| Notebook | Runs on | What it does |
|---|---|---|
| `00_kaggle_ingest.ipynb` | Kaggle, no GPU | Streams the 19.38 GiB download into per-day blocks |
| `01_kaggle_lstm.ipynb` | Kaggle, **GPU** | The five-stage LSTM sweep (2.9 min on a T4) |
| `02_eda.ipynb` | Local | Exploratory figures and statistics |
| `03_kaggle_final.ipynb` | Kaggle, **GPU** | Final fits for all models, test and stress splits |
| `04_results.ipynb` | Local | Evaluation, diagnostics and failure analysis |

Executed copies are kept under `notebooks/runs/` as evidence of the runs that produced the
committed artefacts.

---

## Data handling and memory

Measured on the hardware recorded in `results/environment.json`, one process per
measurement (see `results/tables/memory_report.csv`):

| Strategy | Resident/day | Peak RSS | Wall/day | Scales with days |
|---|---:|---:|---:|---|
| Naive pandas | 295.6 MiB | 633.1 MiB | 5.56 s | **yes** |
| Chunked pandas | 5.5 MiB | 172.6 MiB | 5.86 s | no |
| Polars lazy | 5.5 MiB | 552.2 MiB | 0.95 s | no |

The headline is **scaling rather than peak**: holding all 62 days the naive way would need
17.90 GiB resident, while both optimised paths hold one 5.49 MiB block at a time regardless
of dataset size, producing a 340.58 MiB final matrix. The pipeline is O(1) in dataset size
where the naive approach is O(*n*).

Polars is ~6× faster but peaks ~3.2× higher, because its CSV reader holds the whole ~322 MB
file; `low_memory=True` (559 MiB) and `read_csv_batched` (697–862 MiB) were both measured
and made it worse. Neither strategy dominates, so `--strategy` selects and both are kept.

**Each measurement runs in its own process.** Measuring three strategies in one interpreter
reported 544, 184 and 384 MiB for *identical* work, because freed arenas are not returned to
the OS and whichever ran first paid for the heap growth. Available RAM is recorded alongside
each row, because memory pressure depresses measured peak RSS and would understate demand.

Full run, 62 days: 319,896,289 records parsed, 5.92 min download + 1.54 min ingest, peak RSS
574 MiB.

---

## Exploratory findings

Six measured constraints drove every modelling decision:

1. **Two seasonal cycles operate simultaneously** — MSTL gives the daily period 82.9% of
   variance (strength 0.959) and the weekly 10.0% (0.746). SARIMA accommodates one, and at
   *s* = 144 is intractable anyway → dynamic harmonic regression.
2. **The daily cycle is not sinusoidal** — a 12.00 h spectral peak sits beside the 24.00 h
   one → at least two daily Fourier harmonics.
3. **No stochastic trend** — ADF and KPSS agree across raw, differenced and
   seasonally-differenced series → *d* = 0. "Stationary" here means no unit root; the mean
   still varies enormously with time of day.
4. **Lag-1 autocorrelation is 0.987** → persistence is a demanding baseline, and a flexible
   model may collapse to reproducing its input. Both a baseline and an explicit check are
   required.
5. **ACF peaks at 144, 288, 432, 720, 864, 1008** → the lag set for LightGBM is identified
   empirically rather than by convention.
6. **Variance scales with level** — daily mean vs daily SD correlates +0.947 raw, +0.265
   after `log1p` → the variance-stabilising transform.

Spatial concentration is substantial but not extreme (Gini 0.608; top 1% of cells carry
11.0%). Traffic character varies at a finer spatial scale than volume rank reveals: the
three busiest cells all sit within 500 m of the Duomo, yet their weekend ratios span
0.425–1.384, a factor of 3.3.

Anomaly detection required a per-position robust scale, because seasonal-naive residual
spread varies 25-fold across the day and a single global scale flagged 12.7% of all
intervals. Under the corrected scale, 152 intervals (1.73%) are flagged, with holidays
**1.78× enriched** (*p* = 4.3 × 10⁻⁴). One case resists explanation: Monday 2 December is
the worst single day and is not a holiday.

---

## Methodology

One-step-ahead (10-minute) forecasting, univariate, one model per area, at the data's native
resolution. Inference is always `walk_forward` with **true observed history** — not a
recursive rollout — so each forecast uses real observations up to the step before the one it
predicts.

**Three models, each chosen for a stated reason.**

| Model | Why it is here | Parameters |
|---|---|---:|
| Dynamic harmonic regression | SARIMAX with Fourier terms. Plain SARIMA cannot express two seasonalities at once, and *s* = 144 is intractable. | 22 |
| LSTM | The only candidate that assumes nothing about seasonal form. | 202,369 |
| LightGBM | Gradient boosting won M5; prior work on this dataset evaluated random forests but not boosting. | 19,639–21,010 |

LightGBM's count is total ensemble leaves — the closest honest analogue to a parameter count
for a tree model. Unlike the other two it varies by area, because it depends on the data
fitted rather than the architecture.

**Preprocessing.** All models fit on `log1p`-transformed values; the LSTM and LightGBM
additionally standardise. Statistics are estimated on the **training split only** and the
scaler raises `LeakageError` if asked to refit. Metrics are reported after
inverse-transforming to original units.

**Input representation.** Harmonic ARIMA sees 16 Fourier terms as exogenous regressors; the
LSTM sees a `(144, 7)` window (one value plus six calendar features); LightGBM sees 27
features (9 causal lags, 3 rolling means and standard deviations, 6 calendar). Every feature
is computed strictly before the target interval.

**Two baselines appear in every table.** With a lag-1 autocorrelation of 0.987, persistence
is not a straw man — it is the bar that decides whether complexity earned its place.

**Tuning** ran on the highest-traffic area only, selecting on validation MAE, one axis at a
time so each decision is attributable. Harmonic order was chosen by AICc; LightGBM used
Optuna over 30 trials because tree parameters interact too strongly for a staged sweep; the
LSTM used five staged sweeps, including a calendar ablation worth 14.7%. Every candidate —
**101 in total** — appends a row to `results/experiments.csv` with the reasoning that
produced the next change.

**The final fit uses train + validation**, since the validation split has done its job by
then. Neither early-stopping model has a held-out set left, so each carries the *complexity*
tuning chose rather than the stopping rule: the LSTM trains for its best epoch count with
patience disabled, LightGBM uses its selected 320 trees. The LSTM is run over three seeds and
reported as mean ± standard deviation.

---

## Results

Test week, 16–22 Dec 2013. MASE is the comparable metric: the areas differ by an order of
magnitude in volume, so raw MAE cannot be compared across them.

| Model | 5161 | 5059 | 5259 | mean |
|---|---:|---:|---:|---:|
| **harmonic ARIMA** | 0.241 | **0.244** | **0.120** | **0.202** |
| LSTM (3-seed ensemble) | **0.233** | 0.255 | 0.120 | 0.202 |
| LightGBM | 0.249 | 0.255 | 0.122 | 0.209 |
| persistence | 0.267 | 0.302 | 0.145 | 0.238 |
| LSTM (mean of 3 seeds) | 0.265 | 0.263 | 0.123 | 0.217 |
| seasonal naive | 0.975 | 0.635 | 0.898 | 0.836 |

**All models beat persistence.** In the extreme hotspots of the city center, the traffic patterns are more predictable than the wider network. Harmonic ARIMA has the best mean MASE and wins two of the three areas; the LSTM ensemble takes the third.

The two LSTM rows are different quantities and both are reported: `lstm` is the mean of three seeds' *errors*, `lstm_ensemble` is the error of their *averaged forecast*.

**Cost inverts between training and inference** (square 5161, 1,008 forecasts):

| Model | Device | Train | Inference | Per step | Parameters |
|---|---|---:|---:|---:|---:|
| harmonic ARIMA | CPU | 18.0 s | 75.2 s | 74.6 ms | 22 |
| LightGBM | CPU | 5.0 s | 0.022 s | 0.022 ms | 19,639 |
| LSTM | GPU (T4) | 9.9 s | 0.041 s | 0.041 ms | 202,369 |

The cheapest model to train is by far the most expensive to serve: `append(refit=False)`
still runs a Kalman update per step, making harmonic ARIMA **~1,825× slower** per forecast
than the LSTM. At city scale that disqualifies it — one forecast for each of the 10,000 cells
would take 12.4 minutes, longer than the ten-minute interval being forecast.

Training times are not comparable across devices. For one configuration fitted on both
machines the same fit took **2,019 s on the CPU against 2.6 s on the T4** (777×). The largest
configuration reached during the CPU sweep took 13.8 hours for a single fit; its GPU
counterpart was never run, so that is a CPU cost rather than half of a ratio.

---

## Failure analysis

**No model collapsed to persistence.** With lag-1 autocorrelation at 0.987 this was a real risk, so it was tested and reported either way; copy ratios run 0.45–0.80 against persistence's 0.00. See `results/tables/copying_test.csv`.

**Complexity did not survive the holidays.** On the held-out stress split (23 Dec – 1 Jan, never tuned on) persistence wins outright on two of the three top areas. While the models generalized reasonably well to normal weeks, they failed to extrapolate the anomalous behavior of extreme hotspots during major holidays. Harmonic ARIMA performed best under stress, staying closest to the persistence baseline.

---

## Testing and engineering practices

**501 tests across 27 modules.** They target the properties the conclusions depend on rather
than line coverage:

- **Leakage** — the scaler refuses to refit; the final fit's inputs stop before the test split.
- **Inference correctness** — the batched `walk_forward` used by LightGBM and the LSTM is
  asserted equal to the step-by-step loop.
- **Metric correctness** — MASE is scaled by training-set error, not by the window scored.
- **Diagnostics** — checked against forecasts whose answer is known by construction: a
  perfect forecast, an exact persistence copy, an error confined to one hour.
- **Artefact integrity** — the committed selections carry their stopping points; timing
  tables with `cuda` rows have a CUDA machine on record; committed summaries hold no absolute
  paths; the report's figures resolve and its numbers match the tables.

Other practices: all logic in `src/` with notebooks rendering only; strict YAML config with
no hardcoded paths or hyperparameters; an append-only experiment log where every row carries
a non-empty rationale; seeded determinism with library versions recorded; `ruff` clean.

Several bugs were caught by tests rather than by inspection — among them a selections file
that silently dropped its early-stopping counts (the final LightGBM fit would have trained
2,000 trees instead of 320), a per-area metrics table reduced to a single row by an
int/string key collision, and a weekday/weekend split that was reading the holiday column.

---

## The report

**[`report/REPORT.md`](report/REPORT.md)** is the consolidated write-up: abstract, 23
figures, 16 tables, 15 references and three appendices, with every number traceable to an
artefact under `report/tables/`. **[`report/REPORT.pdf`](report/REPORT.pdf)** is the
print-ready rendering — 33 pages, A4, Times New Roman, figures at 300 dpi — produced by
`python run.py pdf`.

The renderer (`src/export_pdf.py`) is bespoke: pandoc, wkhtmltopdf and LaTeX are all absent
on the development machine and WeasyPrint needs GTK libraries Windows does not supply, so
the conversion is done directly with ReportLab. Two details there are load-bearing. Fonts
are registered from TrueType files because ReportLab's built-ins are WinAnsi-encoded and
would render U+2212, the arrow, ≥ and the Greek letters as **black boxes**, silently;
`tests/test_export_pdf.py` checks the font's actual glyph table against the report's
character set. And Unicode sub/superscripts are translated to `<sub>`/`<super>` markup
rather than passed through, because Arial omits three of the ones used here entirely.

`tests/test_report.py` checks that each referenced figure exists, that figure and table
numbering is contiguous, that every citation has an entry and every entry is cited, that the
MASE values in the prose match the metrics tables, and that the AI-use declaration is left
for the author.

Supporting material:

| File | Contents |
|---|---|
| `report/RESULTS_SUMMARY.md` | Uninterpreted dump of every measured number, by section |
| `report/FACTS.json` / `.md` | 133 named facts, each with unit, source artefact and description |
| `report/FIGURE_INDEX.md` | Figure → report-section map |
| `report/references.md` | References with per-entry verification notes |
| `report/RELATED_WORK.md` | Literature review behind the model line-up |
| `report/DRAFT_SECTIONS.md` | Section drafts, superseded by `REPORT.md`, kept for history |

Regenerate the whole pack with `python run.py report`, and the PDF with
`python run.py pdf` (`--dpi` controls figure resolution; 300 is print quality).

---

## Licence and data terms

Code in this repository is released under the **MIT Licence** (see `LICENSE`). The
underlying dataset is distributed by Harvard Dataverse under its own terms (ODbL); it is
**not redistributed here**, and `data/raw/` is excluded from version control. Anyone
reproducing the ingest must accept the Dataverse Guestbook themselves.

---

## References

Full IEEE-style reference list with verification notes: `report/references.md`.

1. G. Barlacchi *et al.*, "A multi-source dataset of urban life in the city of Milan and the
   Province of Trentino," *Scientific Data*, vol. 2, art. 150055, 2015.
   doi:10.1038/sdata.2015.55
2. A. Azari, P. Papapetrou, S. Denic, and G. Peters, "Cellular traffic prediction and
   classification: A comparative evaluation of LSTM and ARIMA," in *Discovery Science*,
   Springer, 2019, pp. 129–144. doi:10.1007/978-3-030-33778-0_11
3. C. Zhang and P. Patras, "Long-term mobile traffic forecasting using deep spatio-temporal
   neural networks," in *Proc. MobiHoc '18*, 2018, pp. 231–240. doi:10.1145/3209582.3209606
4. G. L. Santos *et al.*, "Predicting short-term mobile Internet traffic from Internet
   activity using recurrent neural networks," *Int. J. Network Management*, vol. 32, no. 3,
   art. e2191, 2022. doi:10.1002/nem.2191
5. R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed.
   OTexts, 2021.
6. R. J. Hyndman and A. B. Koehler, "Another look at measures of forecast accuracy,"
   *Int. J. Forecasting*, vol. 22, no. 4, pp. 679–688, 2006.
   doi:10.1016/j.ijforecast.2006.03.001

---

**Source code:** `https://github.com/Mahamatbt/milan_traffic_forecasting`
**Demonstration video:** `https://youtu.be/Mn7JPndRAu0`
