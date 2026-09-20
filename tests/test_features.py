"""Tests for scaling, windowing and causal feature construction.

Leakage is the failure this file exists to prevent. It cannot be detected from
metrics -- a model that has seen the future simply reports a better score -- so
causality is tested directly: the future is altered, and every feature computed
for an earlier target must be unchanged.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from src.features import (
    CALENDAR_FEATURE_NAMES,
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    LeakageError,
    LogStandardScaler,
    calendar_features,
    lag_features,
    make_windows,
)

DAILY = 144


def _times(n: int, start: str = "2013-11-04T00:00") -> np.ndarray:
    """Local timestamps at 10-minute spacing from a Monday midnight."""
    return np.datetime64(start) + np.arange(n) * np.timedelta64(10, "m")


@pytest.fixture
def series() -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(DAILY * 12)
    return 100.0 + 40 * np.sin(2 * np.pi * t / DAILY) + rng.normal(0, 3, t.size)


# --------------------------------------------------------------------------
# Scaler
# --------------------------------------------------------------------------


def test_scaler_round_trip(series: np.ndarray) -> None:
    scaler = LogStandardScaler().fit(series)
    np.testing.assert_allclose(
        scaler.inverse_transform(scaler.transform(series)), series, rtol=1e-6
    )


def test_scaler_standardises_the_training_split(series: np.ndarray) -> None:
    scaler = LogStandardScaler().fit(series)
    transformed = scaler.transform(series)
    assert transformed.mean() == pytest.approx(0.0, abs=1e-9)
    assert transformed.std() == pytest.approx(1.0, abs=1e-9)


def test_scaler_refuses_to_be_refitted(series: np.ndarray) -> None:
    """Refitting is how validation or test data leaks into the transform."""
    scaler = LogStandardScaler().fit(series)
    with pytest.raises(LeakageError, match="already fitted"):
        scaler.fit(series)


def test_scaler_must_be_fitted_before_use() -> None:
    with pytest.raises(LeakageError, match="not been fitted"):
        LogStandardScaler().transform(np.array([1.0]))
    with pytest.raises(LeakageError, match="not been fitted"):
        LogStandardScaler().inverse_transform(np.array([1.0]))


def test_scaler_statistics_come_only_from_what_it_was_given(series: np.ndarray) -> None:
    """Fitting on train alone must not depend on values that follow it."""
    train, rest = series[:1000], series[1000:]
    a = LogStandardScaler().fit(train)
    b = LogStandardScaler().fit(np.concatenate([train, rest * 10]))
    assert a.mean_ != pytest.approx(b.mean_)

    # The point: the train-only scaler is unaffected by the tail existing.
    c = LogStandardScaler().fit(train.copy())
    assert a.mean_ == pytest.approx(c.mean_)
    assert a.std_ == pytest.approx(c.std_)


def test_inverse_transform_never_returns_negative_traffic() -> None:
    scaler = LogStandardScaler().fit(np.array([1.0, 2.0, 3.0, 100.0]))
    assert (scaler.inverse_transform(np.array([-50.0, -10.0, 0.0])) >= 0).all()


def test_scaler_rejects_negative_and_degenerate_input() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LogStandardScaler().fit(np.array([-1.0, 2.0]))
    with pytest.raises(ValueError, match="constant"):
        LogStandardScaler().fit(np.full(10, 5.0))
    with pytest.raises(ValueError, match="empty"):
        LogStandardScaler().fit(np.array([]))


def test_scaler_state_is_recorded_for_the_experiment_log(series: np.ndarray) -> None:
    state = LogStandardScaler().fit(series).to_dict()
    assert set(state) == {"mean", "std", "n_fitted"}
    assert state["n_fitted"] == series.size


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------


def test_windows_have_the_expected_shapes() -> None:
    values = np.arange(20, dtype=np.float32)
    x, y = make_windows(values, sequence_length=5)
    assert x.shape == (15, 5, 1)
    assert y.shape == (15,)


def test_window_target_is_the_step_after_the_window() -> None:
    values = np.arange(20, dtype=np.float32)
    x, y = make_windows(values, sequence_length=5)
    for i in range(x.shape[0]):
        np.testing.assert_array_equal(x[i, :, 0], values[i : i + 5])
        assert y[i] == values[i + 5]


def test_no_window_contains_its_own_target() -> None:
    """The defining property of a causal window."""
    values = np.arange(50, dtype=np.float32)
    x, y = make_windows(values, sequence_length=8)
    for i in range(x.shape[0]):
        assert y[i] not in set(x[i, :, 0].tolist())


def test_windows_are_unaffected_by_later_values() -> None:
    """Altering the tail must not change windows that end before it."""
    values = np.arange(60, dtype=np.float32)
    x_before, y_before = make_windows(values, sequence_length=6)

    altered = values.copy()
    altered[40:] = -999.0
    x_after, y_after = make_windows(altered, sequence_length=6)

    unaffected = 40 - 6  # windows whose target index is below 40
    np.testing.assert_array_equal(x_before[:unaffected], x_after[:unaffected])
    np.testing.assert_array_equal(y_before[:unaffected], y_after[:unaffected])


def test_exogenous_features_are_taken_at_the_target_time() -> None:
    """Calendar values are known in advance, so using the target's is valid."""
    values = np.arange(20, dtype=np.float32)
    exog = np.arange(20, dtype=np.float32)[:, None] * 100
    x, _ = make_windows(values, sequence_length=4, exog=exog)

    assert x.shape == (16, 4, 2)
    for i in range(x.shape[0]):
        assert (x[i, :, 1] == (i + 4) * 100).all()


def test_windows_reject_impossible_arguments() -> None:
    with pytest.raises(ValueError, match="too short"):
        make_windows(np.arange(3.0), sequence_length=10)
    with pytest.raises(ValueError, match="sequence_length"):
        make_windows(np.arange(10.0), sequence_length=0)
    with pytest.raises(ValueError, match="horizon"):
        make_windows(np.arange(10.0), sequence_length=2, horizon=0)
    with pytest.raises(ValueError, match="1-D"):
        make_windows(np.zeros((4, 4)), sequence_length=2)


# --------------------------------------------------------------------------
# Calendar features
# --------------------------------------------------------------------------


def test_calendar_encodings_are_cyclical() -> None:
    """23:50 must sit next to 00:00, not at the opposite end of the range."""
    times = np.array(["2013-11-04T23:50", "2013-11-05T00:00"], dtype="datetime64[m]")
    features = calendar_features(times)
    distance = np.hypot(features[0, 0] - features[1, 0], features[0, 1] - features[1, 1])
    assert distance < 0.05


def test_calendar_midnight_and_noon() -> None:
    times = np.array(["2013-11-04T00:00", "2013-11-04T12:00"], dtype="datetime64[m]")
    features = calendar_features(times)
    assert features[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert features[0, 1] == pytest.approx(1.0, abs=1e-6)
    assert features[1, 0] == pytest.approx(0.0, abs=1e-6)
    assert features[1, 1] == pytest.approx(-1.0, abs=1e-6)


def test_weekend_flag_matches_the_standard_library() -> None:
    """Checked against datetime.weekday() rather than a hand-written list.

    The first version of this encoding was off by one day, because 1970-01-01
    was a Thursday and the offset assumed otherwise. Comparing against the
    calendar itself is what caught it.
    """
    times = np.array(
        [f"2013-{month:02d}-{day:02d}T09:00" for month in (11, 12) for day in range(1, 29)],
        dtype="datetime64[m]",
    )
    weekend = calendar_features(times)[:, 4]
    for index, stamp in enumerate(times):
        day = stamp.astype("datetime64[D]").astype(dt.date)
        assert weekend[index] == float(day.weekday() >= 5), day


def test_day_of_week_encoding_is_monday_zero() -> None:
    """sin/cos of day-of-week must place Monday at angle zero."""
    monday = np.array(["2013-11-04T00:00"], dtype="datetime64[m]")
    features = calendar_features(monday)
    assert features[0, 2] == pytest.approx(0.0, abs=1e-6)
    assert features[0, 3] == pytest.approx(1.0, abs=1e-6)


def test_holiday_flag_fires_on_a_known_holiday() -> None:
    times = np.array(
        ["2013-12-06T12:00", "2013-12-07T12:00", "2013-12-25T12:00"],
        dtype="datetime64[m]",
    )
    holiday = calendar_features(times)[:, 5]
    np.testing.assert_array_equal(holiday, [0, 1, 1])  # Sant'Ambrogio, Christmas


def test_calendar_shape_matches_its_names() -> None:
    features = calendar_features(_times(100))
    assert features.shape == (100, len(CALENDAR_FEATURE_NAMES))
    assert features.dtype == np.float32


# --------------------------------------------------------------------------
# Lag features: causality
# --------------------------------------------------------------------------


def test_lag_features_are_unaffected_by_the_future(series: np.ndarray) -> None:
    """The central leakage check.

    Every feature row is computed for a target at index ``target_index[i]``.
    Corrupting the series from some point onward must leave every row whose
    target precedes that point bit-for-bit identical.
    """
    times = _times(series.size)
    x_before, y_before, _, index_before = lag_features(series, times)

    cut = int(index_before[len(index_before) // 2])
    altered = series.copy()
    altered[cut:] = 1e6
    x_after, y_after, _, index_after = lag_features(altered, times)

    np.testing.assert_array_equal(index_before, index_after)
    unaffected = index_before < cut
    np.testing.assert_array_equal(x_before[unaffected], x_after[unaffected])
    np.testing.assert_array_equal(y_before[unaffected], y_after[unaffected])


def test_lag_one_is_the_immediately_preceding_observation(series: np.ndarray) -> None:
    times = _times(series.size)
    x, y, names, index = lag_features(series, times, lags=(1, 2), rolling_windows=(3,))

    lag_1 = x[:, names.index("lag_1")]
    np.testing.assert_allclose(lag_1, series[index - 1], rtol=1e-5)
    lag_2 = x[:, names.index("lag_2")]
    np.testing.assert_allclose(lag_2, series[index - 2], rtol=1e-5)


def test_rolling_statistics_end_before_the_target(series: np.ndarray) -> None:
    """A rolling window must not include the value being predicted."""
    times = _times(series.size)
    window = 6
    x, _, names, index = lag_features(
        series, times, lags=(1,), rolling_windows=(window,), include_calendar=False
    )

    column = x[:, names.index(f"roll_mean_{window}")]
    for row, target in enumerate(index[:50]):
        expected = series[target - window : target].mean()
        assert column[row] == pytest.approx(expected, rel=1e-5)


def test_target_index_aligns_predictions_to_timestamps(series: np.ndarray) -> None:
    times = _times(series.size)
    _, y, _, index = lag_features(series, times)
    np.testing.assert_allclose(y, series[index], rtol=1e-5)


def test_feature_names_match_the_matrix_width(series: np.ndarray) -> None:
    x, _, names, _ = lag_features(series, _times(series.size))
    assert x.shape[1] == len(names)
    assert len(set(names)) == len(names)


def test_default_lags_include_the_measured_acf_peaks() -> None:
    """The lag set is taken from the data, not from convention."""
    for lag in (1, 144, 288, 1008):
        assert lag in DEFAULT_LAGS


def test_lag_features_reject_a_series_shorter_than_its_history() -> None:
    short = np.arange(50.0)
    with pytest.raises(ValueError, match="too short"):
        lag_features(short, _times(short.size))


def test_calendar_can_be_omitted(series: np.ndarray) -> None:
    times = _times(series.size)
    with_cal, _, names_with, _ = lag_features(series, times)
    without, _, names_without, _ = lag_features(series, times, include_calendar=False)

    assert with_cal.shape[1] - without.shape[1] == len(CALENDAR_FEATURE_NAMES)
    assert not set(CALENDAR_FEATURE_NAMES) & set(names_without)
    assert set(CALENDAR_FEATURE_NAMES) <= set(names_with)


def test_default_windows_cover_short_and_daily_scales() -> None:
    assert 144 in DEFAULT_ROLLING_WINDOWS
    assert min(DEFAULT_ROLLING_WINDOWS) < 12
