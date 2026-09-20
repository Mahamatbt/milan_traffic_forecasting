# Fact registry

Every figure the report may cite, computed from the artefacts and named.
Quote a fact by its key so that a change in measurement propagates by
regeneration rather than by recollection.

Regenerate with `python -m src.facts`.

## Dataset

| key | value | unit | source | description |
|---|---|---|---|---|
| `grid_cells_total` | 89,280,000 | cells | `derived` | Timestamps x squares |
| `matrix_mib` | 340.58 | MiB | `derived` | Assembled matrix as float32 |
| `n_days` | 62 | days | `config` | Observation period length |
| `n_squares` | 10,000 | cells | `config` | Grid cells in Milan |
| `n_timestamps` | 8,928 | intervals | `config` | 10-minute intervals in the period |
| `raw_bytes` | 20,804,803,507 | bytes | `Dataverse file listing` | Total published dataset size |
| `raw_gib` | 19.38 | GiB | `Dataverse file listing` | Total published dataset size |

## Areas

| key | value | unit | source | description |
|---|---|---|---|---|
| `forecast_squares` | 5161, 5059, 5259 |  | `selected_areas.json` | Cells modelled in Section 4 |
| `rank_5059` | 424 |  | `selected_areas.json` | Rank of square 5059 by total activity |
| `rank_5259` | 109 |  | `selected_areas.json` | Rank of square 5259 by total activity |
| `rank_5059` | 2 |  | `selected_areas.json` | Rank of square 5059 by total activity |
| `rank_5161` | 1 |  | `selected_areas.json` | Rank of square 5161 by total activity |
| `rank_5259` | 3 |  | `selected_areas.json` | Rank of square 5259 by total activity |
| `top_three_squares` | 5161, 5059, 5259 |  | `selected_areas.json` | Three highest-traffic cells |
| `top_traffic_square` | 5,161 |  | `selected_areas.json` | Highest-traffic cell |

## Models

| key | value | unit | source | description |
|---|---|---|---|---|
| `experiments_logged` | 101 | runs | `experiments.csv` | Candidates logged across all three models, each with a rationale |

## Results

| key | value | unit | source | description |
|---|---|---|---|---|
| `best_model_gain_over_persistence` | 15.2% | % | `final_metrics_all_test.csv` | How much the best model improves on persistence, mean MASE |
| `best_model_test` | harmonic_arima |  | `final_metrics_all_test.csv` | Model with the lowest mean MASE across areas on the test week |
| `mase_mean_harmonic_arima` | 0.202 | MASE | `final_metrics_all_test.csv` | Mean MASE for harmonic_arima across the three areas, test week |
| `mase_mean_lightgbm` | 0.209 | MASE | `final_metrics_all_test.csv` | Mean MASE for lightgbm across the three areas, test week |
| `mase_mean_lstm` | 0.217 | MASE | `final_metrics_all_test.csv` | Mean MASE for lstm across the three areas, test week |
| `mase_mean_lstm_ensemble` | 0.202 | MASE | `final_metrics_all_test.csv` | Mean MASE for lstm_ensemble across the three areas, test week |
| `mase_mean_persistence` | 0.238 | MASE | `final_metrics_all_test.csv` | Mean MASE for persistence across the three areas, test week |
| `mase_mean_seasonal_naive` | 0.836 | MASE | `final_metrics_all_test.csv` | Mean MASE for seasonal_naive across the three areas, test week |
| `models_beating_persistence` | 4 | models | `final_metrics_all_test.csv` | Models whose mean MASE beats persistence on the test week |

## Failure analysis

| key | value | unit | source | description |
|---|---|---|---|---|
| `stress_areas_persistence_wins` | 2 | areas | `final_metrics_all_stress.csv` | Areas on the holiday split where no model beats persistence |
| `stress_worst_ratio_harmonic_arima` | 1.16x | x persistence | `final_metrics_all_stress.csv` | Worst per-area MASE for harmonic_arima on the holiday split, relative to persistence on the same area |
| `stress_worst_ratio_lightgbm` | 1.69x | x persistence | `final_metrics_all_stress.csv` | Worst per-area MASE for lightgbm on the holiday split, relative to persistence on the same area |
| `stress_worst_ratio_lstm` | 2.14x | x persistence | `final_metrics_all_stress.csv` | Worst per-area MASE for lstm on the holiday split, relative to persistence on the same area |

## Diagnostics

| key | value | unit | source | description |
|---|---|---|---|---|
| `copy_ratio_min` | 0.45 |  | `copying_test.csv` | Closest any model comes to persistence; 0.00 would be a collapse |
| `models_collapsed_to_persistence` | 0 | models | `copying_test.csv` | Models judged to have collapsed to repeating their last input |

## Cost

| key | value | unit | source | description |
|---|---|---|---|---|
| `inference_ms_per_step_harmonic_arima` | 69.876 ms | ms | `timing_test.csv` | Wall-clock milliseconds per one-step forecast for harmonic_arima, square 5161 |
| `inference_ms_per_step_lightgbm` | 0.021 ms | ms | `timing_test.csv` | Wall-clock milliseconds per one-step forecast for lightgbm, square 5161 |
| `inference_ms_per_step_lstm` | 0.043 ms | ms | `timing_test.csv` | Wall-clock milliseconds per one-step forecast for lstm, square 5161 |
| `inference_ms_per_step_lstm_ensemble` | 0.130 ms | ms | `timing_test.csv` | Wall-clock milliseconds per one-step forecast for lstm_ensemble, square 5161 |
| `inference_ratio_harmonic_over_lstm` | 1,617x | x | `timing_test.csv` | Harmonic ARIMA inference cost per step relative to the LSTM; the cheapest model to fit is the most expensive to serve |

