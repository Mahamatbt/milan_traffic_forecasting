# Figure index

Each exported figure, the report section it supports, and the table
holding the numbers that accompany it.

All figures are 300 dpi PNG in `report/figures/`.

## Exploratory Analysis

| figure | shows | numbers in |
|---|---|---|
| `01_total_distribution.png` | Histogram of per-cell totals on a log axis, beside the CCDF on log-log axes. | `distribution_stats.json` |
| `02_spatial_totals.png` | log10 total activity over the 100x100 Milano grid, study areas marked by rank. | `selected_areas.json` |
| `03_series_first_fortnight.png` | Traffic for the five study areas, 1-14 November 2013, one panel each. | `area_summary.csv` |
| `04_series_overlay_normalised.png` | The same five series scaled to their own maxima, comparing shape not volume. | `area_summary.csv` |
| `05_mstl_decomposition.png` | MSTL of square 5161 into trend, daily and weekly seasonality, and residual. | `decomposition.json` |
| `06_acf_pacf.png` | ACF and PACF to lag 1100 with confidence bands; daily and weekly lags marked. | `autocorrelation.json` |
| `07_rolling_stats.png` | Rolling mean and standard deviation over a one-day window. | `stationarity.csv` |
| `08_periodogram.png` | Spectral power against period in hours, dominant cycles annotated. | `spectral_peaks.csv` |

## Results

| figure | shows | numbers in |
|---|---|---|
| `forecast_test_5161_harmonic_arima.png` | harmonic_arima against observed on square 5161 over the test week, with the busiest day enlarged beneath and persistence overlaid. | `final_metrics_area_5161_test.csv` |
| `forecast_test_5161_lightgbm.png` | lightgbm against observed on square 5161 over the test week, with the busiest day enlarged beneath and persistence overlaid. | `final_metrics_area_5161_test.csv` |
| `forecast_test_5161_lstm.png` | lstm against observed on square 5161 over the test week, with the busiest day enlarged beneath and persistence overlaid. | `final_metrics_area_5161_test.csv` |
| `forecast_test_5059_harmonic_arima.png` | harmonic_arima against observed on square 5059 over the test week, with the busiest day enlarged beneath and persistence overlaid. | `final_metrics_area_5059_test.csv` |
| `forecast_test_5059_lightgbm.png` | lightgbm against observed on square 5059 over the test week, with the busiest day enlarged beneath and persistence overlaid. | `final_metrics_area_5059_test.csv` |
| `forecast_test_5059_lstm.png` | lstm against observed on square 5059 over the test week, with the busiest day enlarged beneath and persistence overlaid. | `final_metrics_area_5059_test.csv` |
| `forecast_test_5259_harmonic_arima.png` | harmonic_arima against observed on square 5259 over the test week, with the busiest day enlarged beneath and persistence overlaid. | `final_metrics_area_5259_test.csv` |
| `forecast_test_5259_lightgbm.png` | lightgbm against observed on square 5259 over the test week, with the busiest day enlarged beneath and persistence overlaid. | `final_metrics_area_5259_test.csv` |
| `forecast_test_5259_lstm.png` | lstm against observed on square 5259 over the test week, with the busiest day enlarged beneath and persistence overlaid. | `final_metrics_area_5259_test.csv` |
| `cross_area_mase_test.png` | MASE per model grouped by area; the only metric comparable across areas that differ by an order of magnitude in volume. | `cross_area_mase_test.csv` |

## Discussion

| figure | shows | numbers in |
|---|---|---|
| `error_by_hour_test_5161.png` | Mean absolute error against hour of day on square 5161, one line per model. | `error_by_hour_test.csv` |
| `error_by_hour_test_5059.png` | Mean absolute error against hour of day on square 5059, one line per model. | `error_by_hour_test.csv` |
| `error_by_hour_test_5259.png` | Mean absolute error against hour of day on square 5259, one line per model. | `error_by_hour_test.csv` |
| `error_heatmap_test_5161_harmonic_arima.png` | harmonic_arima error over day-of-week by hour-of-day on square 5161. | `error_by_daytype_test.csv` |
| `error_heatmap_test_5161_lightgbm.png` | lightgbm error over day-of-week by hour-of-day on square 5161. | `error_by_daytype_test.csv` |
| `error_heatmap_test_5161_lstm.png` | lstm error over day-of-week by hour-of-day on square 5161. | `error_by_daytype_test.csv` |
| `error_heatmap_test_5059_harmonic_arima.png` | harmonic_arima error over day-of-week by hour-of-day on square 5059. | `error_by_daytype_test.csv` |
| `error_heatmap_test_5059_lightgbm.png` | lightgbm error over day-of-week by hour-of-day on square 5059. | `error_by_daytype_test.csv` |
| `error_heatmap_test_5059_lstm.png` | lstm error over day-of-week by hour-of-day on square 5059. | `error_by_daytype_test.csv` |
| `error_heatmap_test_5259_harmonic_arima.png` | harmonic_arima error over day-of-week by hour-of-day on square 5259. | `error_by_daytype_test.csv` |
| `error_heatmap_test_5259_lightgbm.png` | lightgbm error over day-of-week by hour-of-day on square 5259. | `error_by_daytype_test.csv` |
| `error_heatmap_test_5259_lstm.png` | lstm error over day-of-week by hour-of-day on square 5259. | `error_by_daytype_test.csv` |
| `residual_acf_test_5161.png` | Residual autocorrelation per model on square 5161; structure here is signal the model did not use. | `final_metrics_all_test.csv` |
| `residual_acf_test_5059.png` | Residual autocorrelation per model on square 5059; structure here is signal the model did not use. | `final_metrics_all_test.csv` |
| `residual_acf_test_5259.png` | Residual autocorrelation per model on square 5259; structure here is signal the model did not use. | `final_metrics_all_test.csv` |
| `cross_correlation_test_5161.png` | Forecast-to-observation cross-correlation on square 5161. A lag-1 peak is what a causal one-step forecast looks like, not evidence of copying. | `copying_test.csv` |
| `cross_correlation_test_5059.png` | Forecast-to-observation cross-correlation on square 5059. A lag-1 peak is what a causal one-step forecast looks like, not evidence of copying. | `copying_test.csv` |
| `cross_correlation_test_5259.png` | Forecast-to-observation cross-correlation on square 5259. A lag-1 peak is what a causal one-step forecast looks like, not evidence of copying. | `copying_test.csv` |

## Failure analysis

| figure | shows | numbers in |
|---|---|---|
| `forecast_stress_5161_harmonic_arima.png` | harmonic_arima on square 5161 over the holiday stress split (23 Dec - 1 Jan), never tuned on. | `final_metrics_area_5161_stress.csv` |
| `forecast_stress_5161_lightgbm.png` | lightgbm on square 5161 over the holiday stress split (23 Dec - 1 Jan), never tuned on. | `final_metrics_area_5161_stress.csv` |
| `forecast_stress_5161_lstm.png` | lstm on square 5161 over the holiday stress split (23 Dec - 1 Jan), never tuned on. | `final_metrics_area_5161_stress.csv` |
| `forecast_stress_5059_harmonic_arima.png` | harmonic_arima on square 5059 over the holiday stress split (23 Dec - 1 Jan), never tuned on. | `final_metrics_area_5059_stress.csv` |
| `forecast_stress_5059_lightgbm.png` | lightgbm on square 5059 over the holiday stress split (23 Dec - 1 Jan), never tuned on. | `final_metrics_area_5059_stress.csv` |
| `forecast_stress_5059_lstm.png` | lstm on square 5059 over the holiday stress split (23 Dec - 1 Jan), never tuned on. | `final_metrics_area_5059_stress.csv` |
| `forecast_stress_5259_harmonic_arima.png` | harmonic_arima on square 5259 over the holiday stress split (23 Dec - 1 Jan), never tuned on. | `final_metrics_area_5259_stress.csv` |
| `forecast_stress_5259_lightgbm.png` | lightgbm on square 5259 over the holiday stress split (23 Dec - 1 Jan), never tuned on. | `final_metrics_area_5259_stress.csv` |
| `forecast_stress_5259_lstm.png` | lstm on square 5259 over the holiday stress split (23 Dec - 1 Jan), never tuned on. | `final_metrics_area_5259_stress.csv` |
| `cross_area_mase_stress.png` | MASE per model on the stress split, where persistence wins two of three areas. | `cross_area_mase_stress.csv` |

## Pending

Phase 6 adds: 9 actual-vs-predicted plots (3 models x 3 areas), per-area
zoom panels, error-by-hour heatmaps, residual ACF per model, and the
stress-split failure figures.

