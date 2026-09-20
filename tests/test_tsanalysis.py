"""Tests for the time-series characterisation.

Each analysis runs against a synthetic series whose structure is known exactly,
so the assertion is that the method recovers what was put in. A decomposition
that cannot find a seasonality it was handed is not evidence about the real
data either.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tsanalysis import (
    autocorrelation,
    decompose,
    periodogram_peaks,
    rolling_stats,
    seasonal_naive_anomalies,
    stationarity_table,
)

DAILY = 144
WEEKLY = 1008
INTERVAL_MINUTES = 10


@pytest.fixture(scope="module")
def synthetic() -> np.ndarray:
    """Four weeks with a strong daily cycle, weaker weekly cycle and mild trend."""
    n = WEEKLY * 4
    t = np.arange(n)
    rng = np.random.default_rng(0)
    return (
        100.0
        + 0.004 * t
        + 30 * np.sin(2 * np.pi * t / DAILY)
        + 12 * np.sin(2 * np.pi * t / WEEKLY)
        + rng.normal(0, 4, n)
    )


# --------------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------------


def test_decomposition_recovers_both_seasonalities(synthetic: np.ndarray) -> None:
    result = decompose(synthetic, periods=(DAILY, WEEKLY))

    assert result.periods == [DAILY, WEEKLY]
    assert set(result.seasonal) == {f"seasonal_{DAILY}", f"seasonal_{WEEKLY}"}
    # The daily term was injected with 2.5x the amplitude of the weekly one.
    assert result.variance_share[f"seasonal_{DAILY}"] > result.variance_share[f"seasonal_{WEEKLY}"]
    assert result.variance_share[f"seasonal_{DAILY}"] > 0.5
    assert result.variance_share["residual"] < 0.1


def test_seasonal_strength_is_high_for_an_injected_cycle(synthetic: np.ndarray) -> None:
    result = decompose(synthetic, periods=(DAILY, WEEKLY))
    assert result.strength[f"seasonal_{DAILY}"] > 0.9
    assert 0.0 <= result.strength[f"seasonal_{WEEKLY}"] <= 1.0


def test_components_reconstruct_the_observed_series(synthetic: np.ndarray) -> None:
    """MSTL is additive, so the parts must sum back to the whole."""
    result = decompose(synthetic, periods=(DAILY, WEEKLY))
    rebuilt = result.trend + result.residual + sum(result.seasonal.values())
    np.testing.assert_allclose(rebuilt, result.observed, rtol=1e-6, atol=1e-6)


def test_decomposition_rejects_a_series_shorter_than_two_cycles() -> None:
    with pytest.raises(ValueError, match="too short"):
        decompose(np.zeros(WEEKLY), periods=(DAILY, WEEKLY))


def test_decomposition_stats_are_json_friendly(synthetic: np.ndarray) -> None:
    import json

    json.dumps(decompose(synthetic, periods=(DAILY, WEEKLY)).stats())


# --------------------------------------------------------------------------
# Stationarity
# --------------------------------------------------------------------------


def test_stationarity_table_has_three_transforms(synthetic: np.ndarray) -> None:
    rows = stationarity_table(synthetic, seasonal_period=DAILY)
    assert [r.transform for r in rows] == [
        "raw",
        "first difference",
        f"seasonal difference (lag {DAILY})",
    ]


def test_random_walk_is_non_stationary_until_differenced() -> None:
    """The textbook case both tests should agree on."""
    rng = np.random.default_rng(3)
    walk = np.cumsum(rng.normal(0, 1, 3000))

    raw, differenced, _ = stationarity_table(walk, seasonal_period=DAILY)

    assert not raw.adf_rejects_unit_root
    assert raw.kpss_rejects_stationarity
    assert raw.verdict.startswith("non-stationary")

    assert differenced.adf_rejects_unit_root
    assert not differenced.kpss_rejects_stationarity
    assert differenced.verdict.startswith("stationary")


def test_white_noise_is_stationary() -> None:
    noise = np.random.default_rng(4).normal(0, 1, 3000)
    raw = stationarity_table(noise, seasonal_period=DAILY)[0]
    assert raw.verdict.startswith("stationary")


def test_stationarity_rows_serialise_with_a_verdict(synthetic: np.ndarray) -> None:
    row = stationarity_table(synthetic, seasonal_period=DAILY)[0].to_row()
    assert "verdict" in row
    assert {"adf_pvalue", "kpss_pvalue", "n_obs"} <= set(row)


# --------------------------------------------------------------------------
# Autocorrelation
# --------------------------------------------------------------------------


def test_acf_finds_the_daily_and_weekly_lags(synthetic: np.ndarray) -> None:
    analysis = autocorrelation(synthetic, nlags=1100)

    assert analysis["acf"].size == 1101
    assert analysis["lag_1"] > 0.9
    # Peaks should cluster at the daily period and land on the weekly one.
    near_daily = [lag for lag in analysis["notable_lags"] if abs(lag - DAILY) <= 2]
    assert near_daily, analysis["notable_lags"]
    assert any(abs(lag - WEEKLY) <= 2 for lag in analysis["notable_lags"])


def test_pacf_is_capped_below_the_acf_length(synthetic: np.ndarray) -> None:
    """PACF at large lags is slow and ill-conditioned, so it is bounded."""
    analysis = autocorrelation(synthetic, nlags=1100)
    assert analysis["pacf"].size <= analysis["acf"].size
    assert analysis["pacf_lags"] <= 400


def test_white_noise_has_no_notable_lags() -> None:
    noise = np.random.default_rng(5).normal(0, 1, 5000)
    assert autocorrelation(noise, nlags=300)["notable_lags"] == []


# --------------------------------------------------------------------------
# Spectrum
# --------------------------------------------------------------------------


def test_periodogram_finds_24_and_168_hour_cycles(synthetic: np.ndarray) -> None:
    peaks = periodogram_peaks(synthetic, interval_minutes=INTERVAL_MINUTES)["peaks"]
    periods = [p["period_hours"] for p in peaks]

    assert any(abs(p - 24.0) < 0.5 for p in periods), periods
    assert any(abs(p - 168.0) < 4.0 for p in periods), periods


def test_periodogram_peaks_are_ordered_by_power(synthetic: np.ndarray) -> None:
    peaks = periodogram_peaks(synthetic, interval_minutes=INTERVAL_MINUTES)["peaks"]
    powers = [p["power"] for p in peaks]
    assert powers == sorted(powers, reverse=True)


# --------------------------------------------------------------------------
# Anomalies
# --------------------------------------------------------------------------


def test_injected_spike_is_flagged(synthetic: np.ndarray) -> None:
    series = synthetic.copy()
    series[2000] += 400

    result = seasonal_naive_anomalies(series, seasonal_period=DAILY)
    assert 2000 in result["index"].tolist()


def test_clean_series_flags_almost_nothing(synthetic: np.ndarray) -> None:
    """A detector that fires constantly identifies nothing."""
    result = seasonal_naive_anomalies(synthetic, seasonal_period=DAILY, z_threshold=4.0)
    assert result["fraction_flagged"] < 0.01


def test_per_phase_scale_handles_heteroscedastic_errors() -> None:
    """A global scale is set by the quiet hours and flags every busy one.

    Reproduces the real failure: on square 5161 the seasonal-naive residual's
    spread varies 25x across the day, and a single global scale flagged 12.7%
    of all points at z > 4.
    """
    rng = np.random.default_rng(7)
    n = DAILY * 40
    phase = np.arange(n) % DAILY
    # Noise amplitude swings 20x between the quiet and busy parts of the cycle.
    amplitude = 1.0 + 19.0 * (np.sin(2 * np.pi * phase / DAILY) ** 2)
    series = np.cumsum(np.zeros(n)) + rng.normal(0, 1, n) * amplitude

    globally = seasonal_naive_anomalies(series, seasonal_period=DAILY, per_phase_scale=False)
    per_phase = seasonal_naive_anomalies(series, seasonal_period=DAILY, per_phase_scale=True)

    assert per_phase["fraction_flagged"] < globally["fraction_flagged"]
    assert per_phase["fraction_flagged"] < 0.01


def test_per_phase_scale_still_finds_a_real_spike() -> None:
    """Rescaling must not cost sensitivity to a genuine outlier."""
    rng = np.random.default_rng(8)
    n = DAILY * 40
    phase = np.arange(n) % DAILY
    amplitude = 1.0 + 19.0 * (np.sin(2 * np.pi * phase / DAILY) ** 2)
    series = rng.normal(0, 1, n) * amplitude
    # Inject into a quiet phase, where a global scale would hide it least.
    quiet = int(np.argmin(amplitude[:DAILY]))
    target = DAILY * 20 + quiet
    series[target] += 200

    result = seasonal_naive_anomalies(series, seasonal_period=DAILY, per_phase_scale=True)
    assert target in result["index"].tolist()


def test_scale_choice_is_recorded(synthetic: np.ndarray) -> None:
    """The report must be able to say which scaling produced the numbers."""
    result = seasonal_naive_anomalies(synthetic, seasonal_period=DAILY)
    assert result["scale_per_phase"] is True
    assert result["scale"] > 0


def test_series_shorter_than_the_period_is_rejected() -> None:
    with pytest.raises(ValueError, match="shorter than period"):
        seasonal_naive_anomalies(np.zeros(10), seasonal_period=DAILY)


def test_threshold_controls_sensitivity(synthetic: np.ndarray) -> None:
    loose = seasonal_naive_anomalies(synthetic, seasonal_period=DAILY, z_threshold=2.0)
    strict = seasonal_naive_anomalies(synthetic, seasonal_period=DAILY, z_threshold=6.0)
    assert loose["n_flagged"] >= strict["n_flagged"]


def test_anomaly_indices_refer_to_the_original_series(synthetic: np.ndarray) -> None:
    """Residuals start at `seasonal_period`, so indices must be shifted back."""
    result = seasonal_naive_anomalies(synthetic, seasonal_period=DAILY)
    assert (result["index"] >= DAILY).all()
    assert (result["index"] < synthetic.size).all()


# --------------------------------------------------------------------------
# Rolling statistics
# --------------------------------------------------------------------------


def test_rolling_mean_of_a_constant_series_is_constant() -> None:
    stats = rolling_stats(np.full(500, 7.0), window=144)
    np.testing.assert_allclose(stats["mean"], 7.0, rtol=1e-9)
    np.testing.assert_allclose(stats["std"], 0.0, atol=1e-6)


def test_rolling_stats_track_a_known_step() -> None:
    values = np.concatenate([np.zeros(200), np.full(200, 10.0)])
    stats = rolling_stats(values, window=50)
    assert stats["mean"][0] == pytest.approx(0.0)
    assert stats["mean"][-1] == pytest.approx(10.0)
    assert stats["std"].max() > 0


def test_rolling_stats_index_aligns_with_the_window_end() -> None:
    stats = rolling_stats(np.arange(100.0), window=10)
    assert stats["index"][0] == 9
    assert stats["index"][-1] == 99
    assert stats["mean"].size == stats["index"].size


@pytest.mark.parametrize("window", [0, 1, 501])
def test_rolling_stats_rejects_an_invalid_window(window: int) -> None:
    with pytest.raises(ValueError, match="window"):
        rolling_stats(np.zeros(500), window=window)
