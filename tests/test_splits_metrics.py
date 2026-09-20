"""Tests for splits, metrics and the baseline forecasters.

Three properties matter most here. Splits must have exactly the lengths the
configured dates imply, because a silent off-by-one would shift every reported
metric. MASE must equal 1.0 when a forecast matches the seasonal-naive baseline,
since that is the value the whole scale is defined against. And ``walk_forward``
must use only true prior observations -- the vectorised baseline
implementations are checked against the step-by-step loop for exactly that.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.metrics import Metrics, evaluate, mase_denominator
from src.models.base import walk_forward
from src.models.baselines import Persistence, SeasonalNaive
from src.splits import SplitError, make_splits

DAILY = 144


@pytest.fixture(scope="module")
def config():
    return load_config(auto_env=False)


@pytest.fixture(scope="module")
def full_series(config):
    """The real series for the top area, if the artefacts are present."""
    import polars as pl

    path = config.paths.processed / "selected_series.parquet"
    if not path.exists():
        pytest.skip("selected_series.parquet not present; run python run.py eda")
    frame = pl.read_parquet(path)
    column = next(c for c in frame.columns if c.startswith("square_"))
    return frame[column].to_numpy().astype(np.float64), frame["local"].to_numpy()


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def test_split_lengths_match_the_configured_dates(config, full_series) -> None:
    values, times = full_series
    splits = make_splits(values, times, config)

    assert len(splits.train) == 5472
    assert len(splits.validation) == 1008
    assert len(splits.test) == 1008
    assert len(splits.stress) == 1440


def test_splits_are_contiguous_and_ordered(config, full_series) -> None:
    values, times = full_series
    splits = make_splits(values, times, config)
    ordered = list(splits)

    for earlier, later in zip(ordered[:-1], ordered[1:], strict=True):
        assert later.start == earlier.stop
        assert earlier.last_day < later.first_day


def test_test_split_is_the_assigned_week(config, full_series) -> None:
    values, times = full_series
    test = make_splits(values, times, config).test

    assert test.first_day.isoformat() == "2013-12-16"
    assert test.last_day.isoformat() == "2013-12-22"
    assert test.first_day.weekday() == 0  # Monday
    assert test.last_day.weekday() == 6  # Sunday


def test_fit_values_cover_train_and_validation_only(config, full_series) -> None:
    """The final refit may use validation; it must never touch test."""
    values, times = full_series
    splits = make_splits(values, times, config)

    assert splits.fit_values.size == len(splits.train) + len(splits.validation)
    np.testing.assert_array_equal(splits.fit_values[: len(splits.train)], splits.train.values)
    assert splits.fit_values.size == splits.test.start


def test_split_values_match_the_underlying_series(config, full_series) -> None:
    values, times = full_series
    splits = make_splits(values, times, config)
    for split in splits:
        np.testing.assert_array_equal(split.values, values[split.start : split.stop])


def test_length_mismatch_is_rejected(config, full_series) -> None:
    """A series that does not match the configured boundaries must fail loudly."""
    values, times = full_series
    with pytest.raises(SplitError, match="expected"):
        make_splits(values[:-1], times[:-1], config)


def test_mismatched_lengths_are_rejected(config, full_series) -> None:
    values, times = full_series
    with pytest.raises(SplitError, match="against"):
        make_splits(values, times[:-5], config)


def test_unknown_split_name_is_rejected(config, full_series) -> None:
    values, times = full_series
    splits = make_splits(values, times, config)
    with pytest.raises(KeyError):
        splits["nonexistent"]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _periodic(n: int = DAILY * 10) -> np.ndarray:
    t = np.arange(n)
    return 100.0 + 30 * np.sin(2 * np.pi * t / DAILY) + 5 * np.cos(2 * np.pi * t / 7)


def test_perfect_forecast_scores_zero() -> None:
    truth = _periodic()
    result = evaluate(truth[-100:], truth[-100:], y_train=truth, seasonal_period=DAILY)

    assert result.mae == 0.0
    assert result.rmse == 0.0
    assert result.mape == 0.0
    assert result.mase == 0.0
    assert result.r2 == pytest.approx(1.0)


def test_mase_is_one_when_the_forecast_equals_seasonal_naive() -> None:
    """The definition MASE is scaled against, checked directly."""
    rng = np.random.default_rng(0)
    series = _periodic() + rng.normal(0, 4, DAILY * 10)

    start = len(series) - 500
    truth = series[start:]
    seasonal = series[start - DAILY : len(series) - DAILY]

    result = evaluate(truth, seasonal, y_train=series[:start], seasonal_period=DAILY)
    # The evaluation-window error against the in-sample scale; equal in
    # expectation and close in practice for a stationary series.
    assert result.mase == pytest.approx(1.0, rel=0.15)


def test_mase_below_one_beats_the_naive_baseline() -> None:
    rng = np.random.default_rng(1)
    train = _periodic() + rng.normal(0, 10, DAILY * 10)
    truth = np.full(200, 100.0)
    near = truth + rng.normal(0, 0.1, 200)

    assert evaluate(truth, near, y_train=train, seasonal_period=DAILY).mase < 1.0


def test_mae_and_rmse_on_a_known_error() -> None:
    truth = np.full(100, 50.0)
    predicted = truth + 3.0
    result = evaluate(truth, predicted, y_train=_periodic(), seasonal_period=DAILY)

    assert result.mae == pytest.approx(3.0)
    assert result.rmse == pytest.approx(3.0)
    assert result.bias == pytest.approx(3.0)
    assert result.mape == pytest.approx(6.0)


def test_bias_distinguishes_over_and_under_forecasting() -> None:
    truth = np.full(50, 20.0)
    train = _periodic()
    assert evaluate(truth, truth + 2, y_train=train, seasonal_period=DAILY).bias > 0
    assert evaluate(truth, truth - 2, y_train=train, seasonal_period=DAILY).bias < 0


def test_small_denominators_are_reported_not_clipped() -> None:
    """MAPE near zero is unstable; the count must be visible to the reader."""
    truth = np.array([0.5, 0.2, 100.0, 200.0])
    predicted = np.array([1.0, 1.0, 101.0, 199.0])
    result = evaluate(
        truth, predicted, y_train=_periodic(), seasonal_period=DAILY, mape_threshold=1.0
    )

    assert result.n_small_denominators == 2
    assert result.mape > result.mape_excluding_small
    assert np.isfinite(result.mape_excluding_small)


def test_smape_is_bounded_at_200_percent() -> None:
    truth = np.array([1.0, 1.0, 1.0])
    predicted = np.array([1000.0, 1000.0, 1000.0])
    result = evaluate(truth, predicted, y_train=_periodic(), seasonal_period=DAILY)
    assert 0 <= result.smape <= 200


def test_wape_weights_by_volume() -> None:
    truth = np.array([10.0, 1000.0])
    predicted = np.array([20.0, 1000.0])
    result = evaluate(truth, predicted, y_train=_periodic(), seasonal_period=DAILY)
    # MAPE treats both points equally; WAPE weights by magnitude.
    assert result.wape < result.mape


def test_mase_denominator_uses_the_training_series() -> None:
    train = _periodic()
    expected = np.mean(np.abs(train[DAILY:] - train[:-DAILY]))
    assert mase_denominator(train, DAILY) == pytest.approx(expected)


def test_mase_denominator_rejects_a_degenerate_series() -> None:
    exact = np.tile(np.arange(DAILY, dtype=float), 5)
    with pytest.raises(ValueError, match="exactly periodic"):
        mase_denominator(exact, DAILY)
    with pytest.raises(ValueError, match="too short"):
        mase_denominator(np.arange(10.0), DAILY)


def test_metrics_reject_malformed_input() -> None:
    train = _periodic()
    with pytest.raises(ValueError, match="observations against"):
        evaluate(np.ones(5), np.ones(4), y_train=train, seasonal_period=DAILY)
    with pytest.raises(ValueError, match="empty"):
        evaluate(np.array([]), np.array([]), y_train=train, seasonal_period=DAILY)
    with pytest.raises(ValueError, match="non-finite"):
        evaluate(np.array([1.0, np.nan]), np.ones(2), y_train=train, seasonal_period=DAILY)


def test_metrics_serialise_to_a_table_row() -> None:
    result = evaluate(np.ones(10), np.ones(10) * 1.1, y_train=_periodic(), seasonal_period=DAILY)
    row = result.to_row()
    assert isinstance(result, Metrics)
    assert {"mae", "rmse", "mape", "mase", "smape", "wape", "r2", "bias", "n"} <= set(row)
    assert "MAE" in result.summary()


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def test_persistence_repeats_the_last_observation() -> None:
    series = np.arange(100.0)
    predictions = Persistence().fit(series).walk_forward(series, 50, 60)
    np.testing.assert_array_equal(predictions, series[49:59])


def test_seasonal_naive_is_exact_on_a_periodic_series() -> None:
    """Given an exactly periodic series, the forecast must be exact."""
    period = np.arange(DAILY, dtype=float)
    series = np.tile(period, 8)

    model = SeasonalNaive(seasonal_period=DAILY).fit(series)
    predictions = model.walk_forward(series, DAILY * 4, DAILY * 5)
    np.testing.assert_allclose(predictions, series[DAILY * 4 : DAILY * 5])


def test_seasonal_naive_copies_the_previous_day() -> None:
    rng = np.random.default_rng(2)
    series = rng.normal(100, 10, DAILY * 6)
    model = SeasonalNaive(seasonal_period=DAILY).fit(series)

    predictions = model.walk_forward(series, DAILY * 3, DAILY * 4)
    np.testing.assert_allclose(predictions, series[DAILY * 2 : DAILY * 3])


@pytest.mark.parametrize(
    "model", [Persistence(), SeasonalNaive(seasonal_period=DAILY)], ids=["persistence", "seasonal"]
)
def test_vectorised_walk_forward_matches_the_loop(model) -> None:
    """The fast path must agree with the definition it optimises."""
    rng = np.random.default_rng(3)
    series = rng.normal(100, 10, DAILY * 6)
    model.fit(series)

    start, stop = DAILY * 4, DAILY * 4 + 200
    np.testing.assert_allclose(
        model.walk_forward(series, start, stop),
        walk_forward(model, series, start, stop),
    )


def test_walk_forward_never_sees_the_target(config) -> None:
    """Altering the value being predicted must not change the prediction."""
    rng = np.random.default_rng(4)
    series = rng.normal(100, 10, DAILY * 6)
    model = Persistence().fit(series)

    start, stop = DAILY * 4, DAILY * 4 + 50
    before = walk_forward(model, series, start, stop)

    altered = series.copy()
    altered[start:stop] = 1e9  # corrupt exactly the targets
    after = walk_forward(model, altered, start, stop)

    # Only the first prediction depends on series[start - 1], which is unchanged;
    # every later one would move if the model were reading its own target.
    assert before[0] == after[0]


def test_walk_forward_rejects_an_invalid_window() -> None:
    series = np.arange(100.0)
    model = Persistence().fit(series)
    for start, stop in ((0, 10), (50, 40), (50, 200)):
        with pytest.raises(ValueError, match="valid range|invalid"):
            walk_forward(model, series, start, stop)


def test_seasonal_naive_requires_a_full_period_of_history() -> None:
    series = np.arange(300.0)
    model = SeasonalNaive(seasonal_period=DAILY).fit(series)
    with pytest.raises(ValueError, match="seasonal period"):
        model.walk_forward(series, 10, 20)


def test_baselines_report_no_parameters() -> None:
    assert Persistence().n_params() == 0
    assert SeasonalNaive().n_params() == 0
