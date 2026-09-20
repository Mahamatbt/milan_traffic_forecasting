# Results summary

Every measured number, grouped by report section, each traceable to the
artefact it came from. Facts only: no interpretation, because the
interpretation is the author's work.

Regenerate with `python -m src.export_report`.

---

## Dataset and Data Preparation

### Ingest run (source: `ingest_summary.json`, `ingest_log.csv`)

- Days processed: **62** (2013-11-01 to 2014-01-01)
- Raw rows parsed: **319,896,289**
- Download: **5.92 min** total, 5.73 s/day +/- 0.64
- Ingest: **1.54 min** total, 1.49 s/day +/- 0.1
- Peak RSS: 574 MiB max, 325.8 MiB mean
- Absent cells: **34,682** total, 559.4/day mean, max 2267 on 2013-12-26

### Memory strategies (source: `memory_report.csv`)

| strategy | day | resident | peak RSS | wall | extrapolated |
|---|---|---|---|---|---|
| naive_pandas | 2013-11-01 | 295.57 MiB | 633.06 MiB | 5.56 s | True |
| pandas_chunked | 2013-11-01 | 5.49 MiB | 166.99 MiB | 5.82 s | False |
| polars_lazy | 2013-11-01 | 5.49 MiB | 559.74 MiB | 0.86 s | False |
| pandas_chunked | 2013-11-02 | 5.49 MiB | 177.52 MiB | 6.05 s | False |
| polars_lazy | 2013-11-02 | 5.49 MiB | 548.94 MiB | 0.88 s | False |
| pandas_chunked | 2013-11-03 | 5.49 MiB | 173.12 MiB | 5.71 s | False |
| polars_lazy | 2013-11-03 | 5.49 MiB | 547.97 MiB | 1.11 s | False |

- Holding all 62 days the naive way: **17.90 GiB** resident (extrapolated from one day, not executed).
- Both optimised paths hold one 5.49 MiB block at a time, independent of the number of days.
- Final matrix: 8928 x 10000 float32 = **340.58 MiB**.

### Assembled matrix (source: `data/processed/matrix_report.json`)

- Shape: **(8928, 10000)**, 62 days
- Local range: 2013-11-01T00:00:00.000000000 to 2014-01-01T23:50:00.000000000 (Europe/Rome)
- UTC range: 2013-10-31T23:00:00.000000000 to 2014-01-01T22:50:00.000000000
- Missing intervals: **0**; interpolated: 0; left as NaN: 0
- NaN after policy: **0**
- Total activity: 5,552,894,187.94

---

## Exploratory Analysis

### Distribution across the grid (source: `distribution_stats.json`)

- Cells: 10,000; total activity 5,552,894,188
- Mean 555,289; median 277,871; max/median **45.8x**
- Skewness 4.26; kurtosis 25.50
- Gini **0.608**
- Top 1% hold **11.0%**, top 5% 33.6%, top 10% 48.4%; bottom 50% 11.6%
- Cells with zero total: 0
- Lognormal fit: mu=12.496, sigma=1.211, KS=0.0232, p=4.26e-05
  - KS critical value at n=10,000, alpha=0.05 is 0.0136, so the statistic is 1.7x the threshold. log(x) has skewness +0.023 and kurtosis +0.013.

### Study areas (source: `selected_areas.json`)

- Top three by total: **[5161, 5059, 5259]**
- Highest-traffic area: **5161**
- Fixed by the brief: [5059, 5259]
- Forecast in Section 4: **[5161, 5059, 5259]**

| square | rank | total |
|---|---|---|
| 5161 | 1 | 12,740,060 |
| 5059 | 2 | 11,170,854 |
| 5259 | 3 | 10,485,779 |
| 5059 | 424 | 2,454,134 |
| 5259 | 109 | 4,574,671 |

- Squares 5161, 5059 and 5259 lie within **470 m** of one another; the top ten fit inside a 3.05 x 2.35 km box (grid rows 48-60, columns 54-63).

### Per-area characteristics (source: `area_summary.csv`)

| square | rank | mean | CV | peak/trough | night/mean | wknd/wkday |
|---|---|---|---|---|---|---|
| 5161 | 1 | 1,427 | 0.968 | 99.4 | 0.130 | 1.384 |
| 5059 | 2 | 1,251 | 0.768 | 30.9 | 0.235 | 0.861 |
| 5259 | 3 | 1,174 | 0.939 | 47.0 | 0.333 | 0.425 |
| 5059 | 424 | 275 | 0.660 | 14.9 | 0.489 | 0.587 |
| 5259 | 109 | 512 | 0.485 | 17.0 | 0.534 | 1.140 |

### MSTL decomposition, square 5161 (source: `decomposition.json`)

- Periods: [144, 1008]
  - trend: **2.8%** of variance, strength 0.451
  - seasonal_144: **82.9%** of variance, strength 0.959
  - seasonal_1008: **10.0%** of variance, strength 0.746
  - residual: **3.6%** of variance
- Residual std 262.23 against observed std 1381.57 (**19.0%**)

### Stationarity (source: `stationarity.csv`)

| transform | ADF stat | ADF p | KPSS stat | KPSS p | verdict |
|---|---|---|---|---|---|
| raw | -19.03 | 0.0000 | 0.237 | 0.100 | stationary (both agree) |
| first difference | -15.16 | 0.0000 | 0.003 | 0.100 | stationary (both agree) |
| seasonal difference (lag 144) | -11.83 | 0.0000 | 0.069 | 0.100 | stationary (both agree) |

- Note: KPSS p-values are clamped to the tabulated range, so 0.100 means '>= 0.10', not an exact value.
- Note: these tests address stochastic trend only. They do not test whether the mean varies with time of day, which it does strongly.

### Autocorrelation, square 5161 (source: `autocorrelation.json`)

- Lag-1 ACF: **0.9871**
- ACF at lag 144 (one day): **0.8783**
- ACF at lag 1008 (one week): **0.8377**
- Notable lags: [144, 1008, 864, 288, 432, 720, 576]
- Their ACF values: [0.878, 0.838, 0.798, 0.77, 0.741, 0.733, 0.73]

### Dominant cycles (source: `spectral_peaks.csv`)

| period (h) | power |
|---|---|
| 24.00 | 2.097e+09 |
| 12.00 | 1.392e+08 |
| 165.33 | 6.539e+07 |
| 20.96 | 5.535e+07 |
| 28.08 | 4.317e+07 |
| 1488.00 | 3.259e+07 |

### Seasonal-naive anomalies (source: `anomalies.csv`)

- Flagged: **152** intervals, **1.73%** of the 8,784 that can be evaluated (the first day has no seasonal-naive comparison), at z > 4 with the scale estimated per position in the daily cycle
- On holidays: **35/152** (**23.0%**), against 12.9% of days being holidays = **1.78x** enrichment, binomial p = 4.3e-04

Days with the most flagged intervals:

| date | intervals | holiday |
|---|---|---|
| 2013-12-02 | 17 |  |
| 2014-01-01 | 13 | Capodanno (New Year's Day) |
| 2013-12-25 | 9 | Natale (Christmas Day) |
| 2013-11-03 | 8 |  |
| 2013-11-16 | 8 |  |
| 2013-11-25 | 7 |  |
| 2013-11-15 | 6 |  |
| 2013-11-18 | 6 |  |

---

## Methodology

### Protocol

- One-step-ahead (10 minutes), univariate, one model per area, native resolution.
- Inference is `walk_forward` with true observed history, never a recursive rollout.
- Tuning ran on square 5161 only, selecting on validation MAE.
- Transforms are fitted on the training split alone; `LogStandardScaler` raises
  `LeakageError` if asked to refit.
- Final fits use train + validation; the test week is untouched until prediction.

### Selected hyperparameters

| Model | Selection | Value |
|---|---|---|
| harmonic ARIMA | Fourier orders (AICc) | K1=6, K2=2 |
| harmonic ARIMA | ARIMA order (validation MAE) | (3, 0, 1) |
| harmonic ARIMA | AICc | -6096.6 |
| LightGBM | num_leaves | 68 |
| LightGBM | learning_rate | 0.03449 |
| LightGBM | trees after early stopping | 320 of 2000 ceiling |
| LSTM | sequence_length | 144 |
| LSTM | hidden_size x layers | 128 x 2 |
| LSTM | batch_size, learning_rate | 32, 0.001 |
| LSTM | dropout, weight_decay | 0.0, 0.0001 |
| LSTM | best epoch on validation | 10 |

---

## Results

### Test week (16-22 Dec 2013), MASE by area

MASE is the comparable metric: the areas differ by an order of magnitude in
volume, so raw MAE cannot be compared across them. Lower is better.

| model | 5161 | 5059 | 5259 | mean | worst |
|---|---:|---:|---:|---:|---:|
| harmonic_arima | 0.24076 | 0.24444 | 0.11982 | 0.20167 | 0.24444 |
| lstm_ensemble | 0.23257 | 0.25459 | 0.12006 | 0.20241 | 0.25459 |
| lightgbm | 0.24926 | 0.25492 | 0.12167 | 0.20862 | 0.25492 |
| lstm | 0.26485 | 0.26297 | 0.12332 | 0.21705 | 0.26485 |
| persistence | 0.26714 | 0.3016 | 0.14507 | 0.23794 | 0.3016 |
| seasonal_naive | 0.97467 | 0.63539 | 0.89812 | 0.83606 | 0.97467 |

### Square 5161, test week

| model | mae | mae_std | rmse | mape | smape | mase | r2 | n_seeds |
|---|---|---|---|---|---|---|---|---|
| lstm_ensemble | 80.7949 | 0.0 | 124.4857 | 7.6062 | 7.5706 | 0.23257 | 0.99165 | 3 |
| harmonic_arima | 83.6387 | 0.0 | 128.247 | 7.7983 | 7.6727 | 0.24076 | 0.99113 | 1 |
| lightgbm | 86.59 | 0.0 | 130.8902 | 7.9732 | 7.8059 | 0.24926 | 0.99077 | 1 |
| lstm | 92.0057 | 3.0315 | 142.6728 | 8.4759 | 8.4375 | 0.26485 | 0.98899 | 3 |
| persistence | 92.8019 | 0.0 | 134.8778 | 9.1943 | 9.0965 | 0.26714 | 0.99019 | 1 |
| seasonal_naive | 338.5937 | 0.0 | 619.04 | 25.9398 | 22.8281 | 0.97467 | 0.79345 | 1 |

### Square 5059, test week

| model | mae | mae_std | rmse | mape | smape | mase | r2 | n_seeds |
|---|---|---|---|---|---|---|---|---|
| harmonic_arima | 66.0689 | 0.0 | 94.2472 | 6.51 | 6.4297 | 0.24444 | 0.98991 | 1 |
| lstm_ensemble | 68.8109 | 0.0 | 97.4804 | 6.8252 | 6.6718 | 0.25459 | 0.9892 | 3 |
| lightgbm | 68.9005 | 0.0 | 99.6871 | 6.6342 | 6.5227 | 0.25492 | 0.98871 | 1 |
| lstm | 71.0772 | 3.1175 | 101.1439 | 7.0481 | 6.889 | 0.26297 | 0.98837 | 3 |
| persistence | 81.5169 | 0.0 | 114.3751 | 7.9591 | 7.9089 | 0.3016 | 0.98514 | 1 |
| seasonal_naive | 171.7357 | 0.0 | 245.8702 | 18.0185 | 16.621 | 0.63539 | 0.93132 | 1 |

### Square 5259, test week

| model | mae | mae_std | rmse | mape | smape | mase | r2 | n_seeds |
|---|---|---|---|---|---|---|---|---|
| harmonic_arima | 62.7443 | 0.0 | 88.9614 | 6.9417 | 6.8845 | 0.11982 | 0.9937 | 1 |
| lstm_ensemble | 62.8704 | 0.0 | 90.8402 | 6.8529 | 6.7586 | 0.12006 | 0.99343 | 3 |
| lightgbm | 63.7142 | 0.0 | 94.3325 | 6.8331 | 6.6334 | 0.12167 | 0.99292 | 1 |
| lstm | 64.5769 | 1.4566 | 94.3231 | 6.9616 | 6.8654 | 0.12332 | 0.99292 | 3 |
| persistence | 75.9675 | 0.0 | 109.5784 | 8.1148 | 8.0741 | 0.14507 | 0.99045 | 1 |
| seasonal_naive | 470.3205 | 0.0 | 861.6206 | 71.621 | 42.6127 | 0.89812 | 0.40934 | 1 |

### Computational cost (square 5161, 1,008 forecasts)

Training times are **not comparable across devices**; the device column says
which produced each number. Hardware is recorded in
`results/environment.json` (local) and `results/environment_final_runs.json`
(the Kaggle T4 session that produced these).

| model | device | n_seeds | train_wall_s | inference_wall_s | inference_ms_per_step | n_params |
|---|---|---|---|---|---|---|
| persistence | cpu | 1 | 0.0 | 0.0 | 0.0 | 0 |
| seasonal_naive | cpu | 1 | 0.0 | 0.0 | 0.0 | 0 |
| harmonic_arima | cpu | 1 | 18.186 | 70.4349 | 69.8759 | 22 |
| lstm | cuda | 3 | 11.119 | 0.0435 | 0.0432 | 202369 |
| lstm_ensemble | cuda | 3 | 33.356 | 0.1305 | 0.1295 | 607107 |
| lightgbm | cpu | 1 | 5.338 | 0.0208 | 0.0206 | 19639 |

---

## Discussion and failure analysis

### Collapse-to-persistence check

The lag-1 autocorrelation of the study series is 0.987, so a model can post a
respectable error by repeating its last input. `copy_ratio` is the distance
between the forecast and persistence, divided by how far the series moves
between steps; 0.00 means the two are the same forecast.

`peak_lag` is reported but is **not** the test. A one-step forecast is built
only from data up to t-1, so it cannot contain the innovation at t and will
correlate slightly more with the previous observation than the current one; a
lag-1 peak is what a causal forecast looks like. Seasonal naive is the only
model here peaking at lag 0 and it is the worst forecaster in the study.

| square_id | model | peak_lag | lag_margin | copy_ratio | verdict |
|---|---|---|---|---|---|
| 5161 | persistence | 1 | 0.00419 | 0.0 | collapsed |
| 5161 | seasonal_naive | 0 | -0.0017 | 3.6807 | independent |
| 5161 | harmonic_arima | 1 | 0.00168 | 0.6308 | independent |
| 5161 | lightgbm | 1 | 0.00078 | 0.7845 | independent |
| 5161 | lstm | 1 | 0.00073 | 0.7102 | independent |
| 5161 | lstm_ensemble | 1 | 0.00073 | 0.7102 | independent |
| 5059 | persistence | 1 | 0.00663 | 0.0 | collapsed |
| 5059 | seasonal_naive | 0 | -0.00133 | 2.1269 | independent |
| 5059 | harmonic_arima | 1 | 0.00186 | 0.562 | near-persistence |
| 5059 | lightgbm | 1 | 0.00107 | 0.7236 | independent |
| 5059 | lstm | 1 | 0.001 | 0.6786 | independent |
| 5059 | lstm_ensemble | 1 | 0.001 | 0.6786 | independent |
| 5259 | persistence | 1 | 0.00393 | 0.0 | collapsed |
| 5259 | seasonal_naive | 0 | -0.0012 | 6.2286 | independent |
| 5259 | harmonic_arima | 1 | 0.00132 | 0.45 | near-persistence |
| 5259 | lightgbm | 0 | -0.00062 | 0.8031 | independent |
| 5259 | lstm | 1 | 0.00011 | 0.68 | independent |
| 5259 | lstm_ensemble | 1 | 0.00011 | 0.68 | independent |

### Error by day type (test week)

| square_id | model | weekday_mae | weekend_mae | weekend_penalty | n_weekend |
|---|---|---|---|---|---|
| 5161 | persistence | 88.36353874206543 | 103.89788638220892 | 1.1758004247146383 | 288 |
| 5161 | seasonal_naive | 341.42116078270806 | 331.5251055293613 | 0.9710151086398393 | 288 |
| 5161 | harmonic_arima | 74.96089379570206 | 105.33314535917995 | 1.4051746187319247 | 288 |
| 5161 | lightgbm | 82.64678722680885 | 96.4480543322399 | 1.1669909692624352 | 288 |
| 5161 | lstm | 77.26350291118595 | 89.62353381592669 | 1.1599724376844334 | 288 |
| 5161 | lstm_ensemble | 77.26350291118595 | 89.62353381592669 | 1.1599724376844334 | 288 |
| 5059 | persistence | 84.55371877882216 | 73.92488140530057 | 0.8742948562519789 | 288 |
| 5059 | seasonal_naive | 156.12208713955349 | 210.769883579678 | 1.350032448587984 | 288 |
| 5059 | harmonic_arima | 67.02020707919324 | 63.69076521900903 | 0.950321820756984 | 288 |
| 5059 | lightgbm | 70.3594747945167 | 65.25302071236456 | 0.9274233627089254 | 288 |
| 5059 | lstm | 72.67899791429383 | 59.1405536619486 | 0.8137227446598763 | 288 |
| 5059 | lstm_ensemble | 72.67899791429383 | 59.1405536619486 | 0.8137227446598763 | 288 |
| 5259 | persistence | 89.49097396002875 | 42.15882762273153 | 0.47109586316003016 | 288 |
| 5259 | seasonal_naive | 407.07644752926296 | 628.4306655989753 | 1.5437657211887705 | 288 |
| 5259 | harmonic_arima | 71.59309522106891 | 40.62242997921796 | 0.5674070921753264 | 288 |
| 5259 | lightgbm | 71.51836250698815 | 44.20364360144157 | 0.6180740449296832 | 288 |
| 5259 | lstm | 71.3767548401122 | 41.604612110456806 | 0.5828874148685156 | 288 |
| 5259 | lstm_ensemble | 71.3767548401122 | 41.604612110456806 | 0.5828874148685156 | 288 |

### Worst contiguous 6-hour windows (test week)

`ratio_to_persistence` above 1 means the baseline would have been better over
exactly that stretch.

| square_id | model | start | mae | persistence_mae | ratio_to_persistence |
|---|---|---|---|---|---|
| 5161 | lstm | 2013-12-17T13:20:00.000000000 | 244.117 | 171.866 | 1.42 |
| 5259 | lightgbm | 2013-12-19T12:20:00.000000000 | 172.571 | 135.005 | 1.278 |
| 5161 | lightgbm | 2013-12-17T13:00:00.000000000 | 229.321 | 179.992 | 1.274 |
| 5161 | harmonic_arima | 2013-12-22T14:00:00.000000000 | 253.434 | 199.088 | 1.273 |
| 5161 | lightgbm | 2013-12-21T14:30:00.000000000 | 204.327 | 163.508 | 1.25 |
| 5259 | lstm | 2013-12-19T11:50:00.000000000 | 158.785 | 130.977 | 1.212 |
| 5059 | lstm | 2013-12-19T10:50:00.000000000 | 134.548 | 113.465 | 1.186 |
| 5259 | lightgbm | 2013-12-17T13:10:00.000000000 | 150.199 | 127.628 | 1.177 |
| 5161 | harmonic_arima | 2013-12-21T11:20:00.000000000 | 214.059 | 182.791 | 1.171 |
| 5161 | lightgbm | 2013-12-22T15:30:00.000000000 | 218.376 | 190.002 | 1.149 |
| 5161 | lstm | 2013-12-18T12:50:00.000000000 | 154.747 | 139.838 | 1.107 |
| 5059 | lstm | 2013-12-18T13:30:00.000000000 | 124.538 | 119.316 | 1.044 |

### Held-out stress split (23 Dec - 1 Jan), MASE by area

Never tuned on and never used for selection. Holds four of the eight Italian
public holidays in the study period.

| model | 5161 | 5059 | 5259 | mean |
|---|---:|---:|---:|---:|
| persistence | 0.19827 | 0.21246 | 0.064 | 0.15824 |
| harmonic_arima | 0.21609 | 0.24735 | 0.05935 | 0.17426 |
| lightgbm | 0.33421 | 0.32706 | 0.09575 | 0.25234 |
| lstm_ensemble | 0.34127 | 0.45169 | 0.13023 | 0.30773 |
| lstm | 0.35294 | 0.45538 | 0.13074 | 0.31302 |
| seasonal_naive | 1.26624 | 1.29529 | 0.63192 | 1.06448 |

