# Project brief

Forecasting mobile internet traffic in Milan (Telecom Italia CDR, Nov 2013 - Jan 2014),
one step ahead (10 min), evaluated on 16-22 Dec 2013 across three areas.

## Where things run

Hybrid. `src/`, `config/` and `tests/` are developed and version-controlled locally;
ingest, EDA and the LSTM sweep run on Kaggle via thin notebooks that `git clone` this
repo and import from `src/`. Notebooks never define model or preprocessing logic.

Split by what the work needs, measured rather than assumed:

- **Local CPU** — harmonic ARIMA and LightGBM tuning, the final fits, evaluation.
  Both searches finish in minutes.
- **Kaggle GPU** — the LSTM sweep only (`notebooks/01_kaggle_lstm.ipynb`). A
  288-step recurrence is latency-bound on a sequential dependency that CPU threads
  cannot split: locally it held 0.88 of 8 cores and one fit at
  `hidden_size=128, num_layers=2` took 13.8 h, projecting to ~6 days for the sweep.

Consequence for reporting: `train_wall_s` is **not comparable across models**, so
every results and timing table carries a `device` column. The LSTM's CPU cost is a
finding to report, not an embarrassment to hide.

`python run.py train --models <names>` tunes a subset and merges into
`results/tables/selected_hyperparameters.json`, so a run on one machine never
discards another's selections. The experiment log is append-only across both.

Kaggle disk budget: `/kaggle/working` is ~20 GB and persisted; everything else is
~60 GB of scratch that vanishes at session end. The 20 GB of raw text therefore goes
to `/kaggle/temp/raw` and is deleted file-by-file as it is converted to parquet.
Session cap is 12 h, so the ingest checkpoints per day and resumes.

## Non-negotiables

- Logic in `src/`, notebooks only render figures.
- All config in `config/default.yaml` (+ `config/kaggle.yaml` overrides); no hardcoded
  paths, dates, or hyperparameters anywhere in `src/`.
- Fit scalers and transforms on the training split ONLY.
- Every experiment appends a row to `results/experiments.csv` with a non-empty
  `rationale_for_next_change`.
- Never commit `data/raw/` or `data/interim/`. Always commit
  `data/processed/selected_series.parquet` and `results/predictions/*.parquet`
  (~312 KB) — the LSTM columns cannot be regenerated without a CUDA device, so
  without them a clean clone cannot redraw the figures or recheck the reported
  metrics against the series they came from.
- Timestamps are epoch-ms UTC; always convert to Europe/Rome.
- Rows are split by country code - always aggregate before use.
- Report metrics in original units, after inverse-transforming.
- Persistence and seasonal-naive baselines appear in every results table.
- `walk_forward` with true observed history is the only inference path used for
  reported results. Not a recursive rollout.

## Resolved decisions

- Python 3.12 (3.14 lacks reliable wheels for torch/lightgbm/geopandas).
- Forecasting areas: {top-1 by total traffic, 5059, 5259}. Top-2/top-3 in an appendix.
- Model line-up: dynamic harmonic regression (SARIMAX + Fourier), LSTM, LightGBM.
- Native 10-minute resolution, univariate, per-area models.
- Dec 23 - Jan 1 is a held-out stress split, never tuned on.

## Stop and ask

- Before changing the model line-up.
- Before changing split dates or the evaluation protocol.
- If any data gap falls inside the test week.
- Before writing any report narrative prose.
