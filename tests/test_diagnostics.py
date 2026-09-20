"""Tests for the error-structure and copying diagnostics.

These are checked against forecasts whose answer is known by construction -- a
perfect forecast, an exact persistence copy, a forecast wrong only at night --
because a diagnostic that merely runs is worthless. The copying check in
particular must fire on a model that has collapsed and stay silent on one that
has not, and both cases are constructed here rather than hoped for.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.diagnostics import (
    copying_report,
    cross_area_table,
    cross_correlation,
    error_by_daytype,
    error_by_hour,
    error_heatmap,
    residual_acf,
    worst_windows,
)

DAILY = 144


@pytest.fixture
def series() -> tuple[np.ndarray, np.ndarray]:
    """Two weeks of 10-minute data with a daily cycle, from a Monday."""
    n = DAILY * 14
    start = np.datetime64("2013-12-16T00:00")  # a Monday
    times = start + np.arange(n) * np.timedelta64(10, "m")
    t = np.arange(n)
    values = 500 + 200 * np.sin(2 * np.pi * t / DAILY) + 20 * np.cos(2 * np.pi * t / 7)
    return values, times


# --------------------------------------------------------------------------
# Error by hour and day type
# --------------------------------------------------------------------------


def test_perfect_forecast_has_zero_error_every_hour(series) -> None:
    values, times = series
    result = error_by_hour(values, values.copy(), times)
    assert result["mae"].shape == (24,)
    np.testing.assert_allclose(result["mae"], 0.0, atol=1e-12)
    assert result["n"].sum() == values.size


def test_error_by_hour_isolates_the_hour_that_is_wrong(series) -> None:
    """An error confined to 03:00 must appear at 03:00 and nowhere else."""
    values, times = series
    hours = times.astype("datetime64[h]").astype(int) % 24
    predicted = values.copy()
    predicted[hours == 3] += 100.0

    result = error_by_hour(values, predicted, times)
    assert result["mae"][3] == pytest.approx(100.0)
    assert np.nanmax(np.delete(result["mae"], 3)) == pytest.approx(0.0, abs=1e-12)


def test_weekend_split_counts_and_penalty(series) -> None:
    values, times = series
    # Two full weeks from a Monday: 10 weekdays, 4 weekend days.
    result = error_by_daytype(values, values.copy(), times)
    assert result["n_weekday"] == 10 * DAILY
    assert result["n_weekend"] == 4 * DAILY


def test_weekend_penalty_detects_a_weekend_only_failure(series) -> None:
    from src.features import CALENDAR_FEATURE_NAMES, calendar_features

    values, times = series
    weekend = calendar_features(times)[:, CALENDAR_FEATURE_NAMES.index("is_weekend")].astype(bool)
    predicted = values.copy()
    predicted[weekend] += 50.0

    result = error_by_daytype(values, predicted, times)
    assert result["weekday_mae"] == pytest.approx(0.0, abs=1e-12)
    assert result["weekend_mae"] == pytest.approx(50.0)
    assert not np.isfinite(result["weekend_penalty"])  # zero weekday error


def test_heatmap_is_seven_by_twentyfour_and_starts_on_monday(series) -> None:
    values, times = series
    result = error_heatmap(values, values.copy(), times)
    assert result["grid"].shape == (7, 24)
    # The series starts on a Monday at 00:00, so that cell must be populated.
    assert result["counts"][0, 0] > 0
    assert result["counts"].sum() == values.size


def test_heatmap_leaves_unobserved_cells_as_nan() -> None:
    """An empty cell must not read as a perfect forecast."""
    times = np.datetime64("2013-12-16T00:00") + np.arange(6) * np.timedelta64(10, "m")
    values = np.arange(6, dtype=float)
    result = error_heatmap(values, values, times)
    assert np.isnan(result["grid"]).sum() > 0
    assert result["grid"][0, 0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Residual autocorrelation
# --------------------------------------------------------------------------


def test_residual_acf_starts_at_one(series) -> None:
    values, times = series
    rng = np.random.default_rng(0)
    predicted = values + rng.normal(0, 10, values.size)
    result = residual_acf(values, predicted, n_lags=50)
    assert result["acf"][0] == pytest.approx(1.0)
    assert result["confidence"] > 0


def test_white_noise_residuals_stay_inside_the_confidence_band(series) -> None:
    values, times = series
    rng = np.random.default_rng(1)
    predicted = values + rng.normal(0, 10, values.size)
    result = residual_acf(values, predicted, n_lags=40)

    outside = np.abs(result["acf"][1:]) > result["confidence"]
    assert outside.mean() < 0.15  # a few excursions are expected by chance


def test_seasonal_residuals_show_a_daily_spike(series) -> None:
    """Structure left in the residuals must be visible as structure."""
    values, times = series
    t = np.arange(values.size)
    predicted = values + 50 * np.sin(2 * np.pi * t / DAILY)
    result = residual_acf(values, predicted, n_lags=DAILY + 5)

    assert result["acf"][DAILY] > 0.8
    assert result["acf"][DAILY] > result["acf"][DAILY // 2]


def test_residual_acf_survives_a_perfect_forecast(series) -> None:
    values, times = series
    result = residual_acf(values, values.copy(), n_lags=10)
    assert np.isnan(result["acf"]).all()


# --------------------------------------------------------------------------
# Cross-correlation and the copying check
# --------------------------------------------------------------------------


def test_aligned_forecast_peaks_at_lag_zero(series) -> None:
    values, _ = series
    result = cross_correlation(values, values.copy())
    assert result["peak_lag"] == 0
    assert result["peak_correlation"] == pytest.approx(1.0)


def test_persistence_forecast_peaks_at_lag_one(series) -> None:
    """The signature the plan asks to test for explicitly."""
    values, _ = series
    observed = values[1:]
    persistence = values[:-1]  # last observation carried forward
    result = cross_correlation(observed, persistence)
    assert result["peak_lag"] == 1


def test_copying_report_flags_a_collapsed_model(series) -> None:
    values, _ = series
    observed = values[1:]
    persistence = values[:-1]
    report = copying_report("lazy", 5161, observed, persistence, persistence)

    assert report.verdict == "collapsed"
    assert report.copy_ratio == pytest.approx(0.0, abs=1e-12)
    assert report.peak_lag == 1


def test_copying_report_clears_an_independent_model(series) -> None:
    values, _ = series
    observed = values[1:]
    persistence = values[:-1]
    rng = np.random.default_rng(2)
    # A genuinely different forecast: the truth plus noise, not the last value.
    predicted = observed + rng.normal(0, 30, observed.size)

    report = copying_report("real", 5161, observed, predicted, persistence)
    assert report.verdict == "independent"
    assert report.copy_ratio > 0.6


def test_copying_report_row_is_serialisable(series) -> None:
    values, _ = series
    observed, persistence = values[1:], values[:-1]
    row = copying_report("lazy", 4159, observed, persistence, persistence).to_row()
    assert row["square_id"] == 4159
    assert row["verdict"] == "collapsed"
    assert isinstance(row["copy_ratio"], float)


# --------------------------------------------------------------------------
# Failure windows
# --------------------------------------------------------------------------


def test_worst_window_finds_the_injected_failure(series) -> None:
    values, times = series
    predicted = values.copy()
    bad = slice(500, 536)
    predicted[bad] += 400.0
    persistence = np.concatenate([[values[0]], values[:-1]])

    windows = worst_windows("m", 5161, values, predicted, persistence, times, window=36, top=1)
    assert len(windows) == 1
    assert windows[0].start_index == 500
    assert windows[0].mae == pytest.approx(400.0)


def test_worst_windows_do_not_overlap(series) -> None:
    """Without this the top three are one episode reported three times."""
    values, times = series
    predicted = values.copy()
    predicted[500:536] += 400.0
    predicted[900:936] += 300.0
    predicted[1200:1236] += 200.0
    persistence = np.concatenate([[values[0]], values[:-1]])

    windows = worst_windows("m", 5161, values, predicted, persistence, times, window=36, top=3)
    starts = sorted(w.start_index for w in windows)
    assert starts == [500, 900, 1200]
    for earlier, later in zip(starts[:-1], starts[1:], strict=True):
        assert later - earlier >= 36


def test_worst_windows_are_ordered_worst_first(series) -> None:
    values, times = series
    predicted = values.copy()
    predicted[500:536] += 100.0
    predicted[900:936] += 400.0
    persistence = np.concatenate([[values[0]], values[:-1]])

    windows = worst_windows("m", 5161, values, predicted, persistence, times, window=36, top=2)
    assert windows[0].start_index == 900
    assert windows[0].mae > windows[1].mae


def test_failure_window_ratio_compares_against_persistence(series) -> None:
    values, times = series
    predicted = values.copy()
    predicted[500:536] += 400.0
    persistence = np.concatenate([[values[0]], values[:-1]])

    window = worst_windows("m", 5161, values, predicted, persistence, times, window=36, top=1)[0]
    assert window.ratio > 1.0
    assert window.to_row()["ratio_to_persistence"] == pytest.approx(window.ratio, abs=1e-3)


def test_worst_windows_handles_a_series_shorter_than_the_window() -> None:
    times = np.datetime64("2013-12-16T00:00") + np.arange(10) * np.timedelta64(10, "m")
    values = np.arange(10, dtype=float)
    assert worst_windows("m", 1, values, values, values, times, window=36) == []


# --------------------------------------------------------------------------
# Cross-area comparison
# --------------------------------------------------------------------------


def test_cross_area_table_pivots_and_ranks() -> None:
    rows = [
        {"square_id": 5161, "model": "a", "mase": 0.20},
        {"square_id": 4159, "model": "a", "mase": 0.30},
        {"square_id": 5161, "model": "b", "mase": 0.10},
        {"square_id": 4159, "model": "b", "mase": 0.12},
    ]
    table = cross_area_table(rows)
    assert [r["model"] for r in table] == ["b", "a"]
    assert table[0]["mase_5161"] == pytest.approx(0.10)
    assert table[0]["mase_mean"] == pytest.approx(0.11)
    assert table[1]["mase_worst"] == pytest.approx(0.30)
    assert table[0]["n_areas"] == 2


def test_cross_area_table_marks_missing_areas() -> None:
    """A model evaluated on fewer areas must not look better for it."""
    rows = [
        {"square_id": 5161, "model": "a", "mase": 0.20},
        {"square_id": 4159, "model": "a", "mase": 0.30},
        {"square_id": 5161, "model": "partial", "mase": 0.05},
    ]
    table = cross_area_table(rows)
    partial = next(r for r in table if r["model"] == "partial")
    assert partial["mase_4159"] is None
    assert partial["n_areas"] == 1


def test_weekend_split_uses_saturdays_and_sundays_not_holidays() -> None:
    """Regression: the split once selected is_holiday by taking the last column.

    calendar_features ends with (..., is_weekend, is_holiday), so ``[:, -1]``
    marked Christmas Eve, Christmas and Boxing Day as the weekend and no
    Saturday at all -- and the resulting weekday/weekend comparison was a
    holiday comparison wearing the wrong label.
    """
    import datetime as dt

    n = DAILY * 14
    times = np.datetime64("2013-12-16T00:00") + np.arange(n) * np.timedelta64(10, "m")
    values = np.zeros(n)
    predicted = np.zeros(n)

    truth = np.array(
        [(dt.date(2013, 12, 16) + dt.timedelta(days=d)).weekday() >= 5 for d in range(14)]
    )
    expected_weekend_days = int(truth.sum())
    assert expected_weekend_days == 4  # Dec 21, 22, 28, 29

    result = error_by_daytype(values, predicted, times)
    assert result["n_weekend"] == expected_weekend_days * DAILY
    assert result["n_weekday"] == (14 - expected_weekend_days) * DAILY
    # Christmas Eve to Boxing Day are holidays, and are not the weekend.
    assert result["n_holiday"] > 0
    assert result["n_holiday"] != result["n_weekend"]


def test_a_causal_forecast_may_peak_at_lag_one_without_being_a_copy() -> None:
    """The verdict must not key on the cross-correlation peak.

    A one-step forecast is built from observations up to t-1, so it cannot
    contain the innovation at t and will correlate slightly more with the
    previous value than the current one. On this series -- lag-1
    autocorrelation 0.987 -- that is true of every honest model, so keying the
    verdict on the peak flagged all three real models while clearing seasonal
    naive, which peaks at lag 0 and is the worst forecaster in the study.

    Checked against the committed forecasts rather than a synthetic series,
    because the point is about what real models on this data actually do.
    """
    import polars as pl

    from src.config import PROJECT_ROOT

    parquet = PROJECT_ROOT / "results" / "predictions" / "test_area_5161.parquet"
    if not parquet.exists():
        pytest.skip("committed predictions not present")

    frame = pl.read_parquet(parquet)
    observed = frame["observed"].to_numpy()
    persistence = frame["persistence"].to_numpy()

    harmonic = copying_report(
        "harmonic_arima", 5161, observed, frame["harmonic_arima"].to_numpy(), persistence
    )
    seasonal = copying_report(
        "seasonal_naive", 5161, observed, frame["seasonal_naive"].to_numpy(), persistence
    )

    # The best model peaks at lag 1; the worst peaks at lag 0.
    assert harmonic.peak_lag == 1
    assert seasonal.peak_lag == 0
    # Yet the verdict must not call the good one a copy.
    assert harmonic.verdict != "collapsed"
    assert harmonic.copy_ratio > 0.5


def test_verdict_is_driven_by_distance_from_persistence() -> None:
    """Same peak lag, different verdicts, decided by the copy ratio alone."""
    rng = np.random.default_rng(4)
    values = np.cumsum(rng.normal(0, 1, 3000)) + 500
    observed, persistence = values[1:], values[:-1]
    movement = float(np.mean(np.abs(np.diff(values))))

    close = copying_report("close", 1, observed, persistence + 0.01 * movement, persistence)
    far = copying_report("far", 1, observed, persistence + 2.0 * movement, persistence)

    assert close.verdict == "collapsed"
    assert far.verdict == "independent"
    assert close.copy_ratio < far.copy_ratio
