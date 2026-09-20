"""Scaling, windowing and causal feature construction.

Everything here is built so that a value at time ``t`` is computed from
observations at ``t`` or earlier, and never later. Leakage is the failure mode
that produces excellent metrics and worthless models, and it is invisible in the
numbers: a model that has seen the future reports a low error and gives no sign
that anything is wrong. The tests in ``tests/test_features.py`` therefore check
causality directly, by perturbing the future and asserting that no feature moves.

The scaler refuses to be fitted twice for the same reason. Fitting on the test
split is the most common way to leak, and it looks like ordinary code.

Transform
---------
``log1p`` is applied before standardisation because the spread of this data
scales with its level: on square 5161 the correlation between daily mean and
daily standard deviation is +0.947 on the raw scale and +0.265 after the
transform. It also keeps predictions non-negative once inverted, which matters
for a quantity that cannot be negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.holidays_it import holiday_flags

__all__ = [
    "LogStandardScaler",
    "make_windows",
    "calendar_features",
    "CALENDAR_FEATURE_NAMES",
    "lag_features",
    "DEFAULT_LAGS",
    "DEFAULT_ROLLING_WINDOWS",
]

# Lags identified from the measured autocorrelation rather than by convention:
# the ACF of square 5161 has local maxima at every multiple of the daily period,
# and lag 1 is the single strongest predictor at 0.987.
DEFAULT_LAGS: tuple[int, ...] = (1, 2, 3, 6, 12, 144, 145, 288, 1008)

# Short, half-day and full-day windows, all computed over past values only.
DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (6, 36, 144)

CALENDAR_FEATURE_NAMES: tuple[str, ...] = (
    "sin_minute_of_day",
    "cos_minute_of_day",
    "sin_day_of_week",
    "cos_day_of_week",
    "is_weekend",
    "is_holiday",
)


class LeakageError(RuntimeError):
    """Raised when a transform is asked to learn from data it must not see."""


# --------------------------------------------------------------------------
# Scaling
# --------------------------------------------------------------------------


@dataclass
class LogStandardScaler:
    """``log1p`` followed by standardisation, fitted on training data only.

    The fitted statistics are the mean and standard deviation of ``log1p(x)``,
    not of ``x``. Standardising first and taking logs afterwards would be a
    different transform and is not invertible in the same way.
    """

    mean_: float = field(default=float("nan"), init=False)
    std_: float = field(default=float("nan"), init=False)
    n_fitted_: int = field(default=0, init=False)
    _fitted: bool = field(default=False, init=False)

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, train_values: np.ndarray) -> LogStandardScaler:
        """Learn the transform from the training split.

        Args:
            train_values: Training observations, in original units.

        Returns:
            Self, for chaining.

        Raises:
            LeakageError: If already fitted. Refitting is how a scaler ends up
                having seen validation or test data, so it is refused rather
                than allowed and documented.
            ValueError: On empty, negative or non-finite input.
        """
        if self._fitted:
            raise LeakageError(
                "this scaler is already fitted; create a new one rather than "
                "refitting, which is how test data leaks into the transform"
            )

        values = np.asarray(train_values, dtype=np.float64).ravel()
        if values.size == 0:
            raise ValueError("cannot fit on an empty series")
        if not np.isfinite(values).all():
            raise ValueError("training values contain non-finite entries")
        if (values < 0).any():
            raise ValueError("log1p requires non-negative values")

        logged = np.log1p(values)
        std = float(logged.std(ddof=0))
        if std == 0.0:
            raise ValueError("training values are constant; standardisation is undefined")

        self.mean_ = float(logged.mean())
        self.std_ = std
        self.n_fitted_ = int(values.size)
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise LeakageError("scaler has not been fitted")

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply the fitted transform."""
        self._require_fitted()
        array = np.asarray(values, dtype=np.float64)
        if (array < 0).any():
            raise ValueError("log1p requires non-negative values")
        return (np.log1p(array) - self.mean_) / self.std_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Undo the transform, returning original units.

        ``expm1`` can produce a small negative value where the model predicts
        slightly below zero on the log scale. Those are clipped to zero: the
        quantity is non-negative by definition, and reporting a negative traffic
        forecast would be meaningless.
        """
        self._require_fitted()
        array = np.asarray(values, dtype=np.float64)
        return np.clip(np.expm1(array * self.std_ + self.mean_), 0.0, None)

    def to_dict(self) -> dict[str, Any]:
        """Fitted state, for the experiment log."""
        return {"mean": self.mean_, "std": self.std_, "n_fitted": self.n_fitted_}


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------


def make_windows(
    series: np.ndarray,
    sequence_length: int,
    *,
    horizon: int = 1,
    exog: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over a series to build supervised training pairs.

    Window ``i`` covers ``series[i : i + sequence_length]`` and its target is
    ``series[i + sequence_length + horizon - 1]``. For one-step-ahead forecasting
    that is the observation immediately after the window, so no sample can
    contain its own target.

    Args:
        series: One area's series, already transformed.
        sequence_length: Observations per input window.
        horizon: Steps ahead to predict. 1 for this study.
        exog: Optional ``(n, n_features)`` exogenous array aligned to ``series``.
            The features taken are those *at the target timestamp*, which is
            legitimate because calendar values are known in advance.

    Returns:
        ``(X, y)``. Without ``exog``, ``X`` has shape
        ``(n_windows, sequence_length, 1)``; with it, the exogenous columns are
        broadcast across the window and appended on the feature axis.

    Raises:
        ValueError: If the series is too short, or the arguments are invalid.
    """
    values = np.asarray(series, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"expected a 1-D series, got shape {values.shape}")
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    n_windows = values.size - sequence_length - horizon + 1
    if n_windows < 1:
        raise ValueError(
            f"series of {values.size} is too short for sequence_length "
            f"{sequence_length} at horizon {horizon}"
        )

    # A strided view avoids materialising n_windows copies of the window.
    windows = np.lib.stride_tricks.sliding_window_view(values, sequence_length)[:n_windows]
    targets = values[sequence_length + horizon - 1 :]

    x = windows[..., None].astype(np.float32, copy=True)
    if exog is not None:
        extra = np.asarray(exog, dtype=np.float32)
        if extra.ndim != 2 or extra.shape[0] != values.size:
            raise ValueError(f"exog must be (n, features) aligned to the series; got {extra.shape}")
        target_index = np.arange(n_windows) + sequence_length + horizon - 1
        at_target = extra[target_index]  # (n_windows, n_features)
        broadcast = np.repeat(at_target[:, None, :], sequence_length, axis=1)
        x = np.concatenate([x, broadcast], axis=2)

    return x, targets.astype(np.float32, copy=True)


# --------------------------------------------------------------------------
# Calendar features
# --------------------------------------------------------------------------


def calendar_features(times: np.ndarray) -> np.ndarray:
    """Cyclical time-of-day and day-of-week encodings, plus two flags.

    Time of day and day of week are encoded as sine and cosine pairs so that
    23:50 and 00:00 are adjacent. An integer encoding would place them at
    opposite ends of the range and force the model to learn that the wrap is not
    a discontinuity.

    These are known for any future timestamp, so using the value at the target
    time is not leakage.

    Args:
        times: Local wall-clock ``datetime64`` values.

    Returns:
        A ``(n, 6)`` float32 array, columns as in :data:`CALENDAR_FEATURE_NAMES`.
    """
    stamps = np.asarray(times).astype("datetime64[m]")
    days = stamps.astype("datetime64[D]")

    minute_of_day = (stamps - days).astype("timedelta64[m]").astype(np.float64)
    # 1970-01-01 was a Thursday. With Monday as 0, Thursday is 3, so the
    # epoch day number shifts by 3 rather than 4.
    day_of_week = (days.astype("datetime64[D]").astype(np.int64) + 3) % 7

    angle_day = 2.0 * np.pi * minute_of_day / (24 * 60)
    angle_week = 2.0 * np.pi * day_of_week / 7.0

    return np.column_stack(
        [
            np.sin(angle_day),
            np.cos(angle_day),
            np.sin(angle_week),
            np.cos(angle_week),
            (day_of_week >= 5).astype(np.float64),
            holiday_flags(stamps).astype(np.float64),
        ]
    ).astype(np.float32)


# --------------------------------------------------------------------------
# Lag features
# --------------------------------------------------------------------------


def lag_features(
    series: np.ndarray,
    times: np.ndarray,
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    include_calendar: bool = True,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Build a causal design matrix of lags, rolling statistics and calendar terms.

    Every feature for target ``t`` uses observations at ``t - horizon`` or
    earlier. Rolling statistics end at ``t - horizon``, so the window never
    includes the value being predicted.

    Args:
        series: One area's series, in the units the model will learn on.
        times: Local timestamps aligned to ``series``.
        lags: Lags to include, counted back from the last observed point.
        rolling_windows: Window sizes for rolling mean and standard deviation.
        include_calendar: Append the calendar encodings for the target time.
        horizon: Steps ahead. 1 for this study.

    Returns:
        ``(X, y, feature_names, target_index)`` where ``target_index`` gives the
        position of each target in the original series, so predictions can be
        aligned back to timestamps.

    Raises:
        ValueError: If the series is too short for the requested history.
    """
    values = np.asarray(series, dtype=np.float64).ravel()
    n = values.size
    history = max(max(lags, default=1), max(rolling_windows, default=1))
    first_target = history + horizon - 1
    if first_target >= n:
        raise ValueError(
            f"series of {n} is too short for {history} points of history at horizon {horizon}"
        )

    target_index = np.arange(first_target, n)
    # Index of the most recent observation usable for each target.
    last_observed = target_index - horizon

    columns: list[np.ndarray] = []
    names: list[str] = []

    for lag in lags:
        columns.append(values[last_observed - lag + 1])
        names.append(f"lag_{lag}")

    # Cumulative sums make each rolling statistic O(1) per target and, more
    # importantly, make the window bounds explicit rather than implied by a
    # library's alignment convention.
    padded = np.concatenate([[0.0], np.cumsum(values)])
    padded_sq = np.concatenate([[0.0], np.cumsum(values**2)])
    for window in rolling_windows:
        stop = last_observed + 1
        start = stop - window
        if (start < 0).any():
            raise ValueError(f"rolling window {window} extends before the series start")
        total = padded[stop] - padded[start]
        total_sq = padded_sq[stop] - padded_sq[start]
        mean = total / window
        variance = np.maximum(total_sq / window - mean**2, 0.0)
        columns.append(mean)
        names.append(f"roll_mean_{window}")
        columns.append(np.sqrt(variance))
        names.append(f"roll_std_{window}")

    design = np.column_stack(columns)

    if include_calendar:
        calendar = calendar_features(np.asarray(times)[target_index])
        design = np.concatenate([design, calendar.astype(np.float64)], axis=1)
        names.extend(CALENDAR_FEATURE_NAMES)

    return (
        design.astype(np.float32),
        values[target_index].astype(np.float32),
        names,
        target_index,
    )
