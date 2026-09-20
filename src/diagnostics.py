"""Error structure, copying behaviour and failure windows.

Aggregate metrics say which model is better; they do not say *when* it is worse,
or whether it is doing anything a one-line baseline does not. These functions
answer the second kind of question, and they are kept apart from
:mod:`src.plots` so every number can be computed, tested and quoted without a
figure existing.

The copying check is the one that matters most here. The lag-1 autocorrelation
of the study series is 0.987, which means a model can reach a respectable MAE by
learning to repeat its most recent input, and would look successful while having
learned nothing. Two independent tests are applied: where the cross-correlation
between forecast and observation peaks, and how far the forecasts sit from the
persistence baseline relative to how much the series actually moves. A model
that has collapsed fails both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "error_by_hour",
    "error_by_daytype",
    "error_heatmap",
    "residual_acf",
    "cross_correlation",
    "copying_report",
    "CopyingReport",
    "worst_windows",
    "FailureWindow",
    "cross_area_table",
]


# --------------------------------------------------------------------------
# When the errors happen
# --------------------------------------------------------------------------


def error_by_hour(
    observed: np.ndarray, predicted: np.ndarray, times: np.ndarray
) -> dict[str, np.ndarray]:
    """Mean absolute error grouped by hour of day.

    Args:
        observed: True values.
        predicted: Forecasts aligned to ``observed``.
        times: Local timestamps aligned to ``observed``.

    Returns:
        ``hours`` 0-23, ``mae`` per hour, and ``n`` observations per hour.
    """
    hours = _hours(times)
    errors = np.abs(np.asarray(observed, dtype=np.float64) - np.asarray(predicted))

    mae = np.zeros(24)
    counts = np.zeros(24, dtype=int)
    for hour in range(24):
        mask = hours == hour
        counts[hour] = int(mask.sum())
        mae[hour] = float(errors[mask].mean()) if counts[hour] else np.nan
    return {"hours": np.arange(24), "mae": mae, "n": counts}


def error_by_daytype(
    observed: np.ndarray, predicted: np.ndarray, times: np.ndarray
) -> dict[str, float]:
    """Mean absolute error split by weekday/weekend and by holiday.

    The exploratory analysis found weekend traffic differs in level and shape,
    so a model that has only learned the weekday pattern shows the gap here.
    Holidays are reported separately because they are the failure mode the
    stress split exists to expose, and four of the eight fall inside it.

    Columns are selected by name. Taking the last column of the feature matrix
    silently selected ``is_holiday`` instead, which flagged Christmas week as
    the weekend and no Saturday at all.
    """
    from src.features import CALENDAR_FEATURE_NAMES, calendar_features  # local: avoids a cycle

    features = calendar_features(np.asarray(times))
    weekend = features[:, CALENDAR_FEATURE_NAMES.index("is_weekend")].astype(bool)
    holiday = features[:, CALENDAR_FEATURE_NAMES.index("is_holiday")].astype(bool)
    errors = np.abs(np.asarray(observed, dtype=np.float64) - np.asarray(predicted))

    def _mean(mask: np.ndarray) -> float:
        return float(errors[mask].mean()) if mask.any() else float("nan")

    weekday_mae = _mean(~weekend)
    weekend_mae = _mean(weekend)
    ordinary_mae = _mean(~holiday)
    holiday_mae = _mean(holiday)
    return {
        "weekday_mae": weekday_mae,
        "weekend_mae": weekend_mae,
        "weekend_penalty": weekend_mae / weekday_mae if weekday_mae else float("nan"),
        "n_weekday": int((~weekend).sum()),
        "n_weekend": int(weekend.sum()),
        "ordinary_mae": ordinary_mae,
        "holiday_mae": holiday_mae,
        "holiday_penalty": holiday_mae / ordinary_mae if ordinary_mae else float("nan"),
        "n_holiday": int(holiday.sum()),
    }


def error_heatmap(
    observed: np.ndarray, predicted: np.ndarray, times: np.ndarray
) -> dict[str, np.ndarray]:
    """Mean absolute error over a day-of-week by hour-of-day grid.

    Returns:
        ``grid`` shaped ``(7, 24)`` with Monday first, plus the axis labels and
        the per-cell observation counts. Empty cells are NaN rather than zero,
        so a gap is visibly a gap and not an apparently perfect forecast.
    """
    times = np.asarray(times)
    hours = _hours(times)
    days = _weekdays(times)
    errors = np.abs(np.asarray(observed, dtype=np.float64) - np.asarray(predicted))

    grid = np.full((7, 24), np.nan)
    counts = np.zeros((7, 24), dtype=int)
    for day in range(7):
        for hour in range(24):
            mask = (days == day) & (hours == hour)
            counts[day, hour] = int(mask.sum())
            if counts[day, hour]:
                grid[day, hour] = float(errors[mask].mean())
    return {"grid": grid, "counts": counts, "hours": np.arange(24), "days": np.arange(7)}


# --------------------------------------------------------------------------
# Residual structure
# --------------------------------------------------------------------------


def residual_acf(
    observed: np.ndarray, predicted: np.ndarray, *, n_lags: int = 288
) -> dict[str, np.ndarray]:
    """Autocorrelation of the forecast residuals.

    Structure left in the residuals is signal the model did not use. A daily
    spike at lag 144 means the seasonal shape is still not fully captured; a
    large lag-1 value means the errors are serially dependent and the one-step
    forecast could be improved by using its own recent errors.

    Args:
        observed: True values.
        predicted: Forecasts aligned to ``observed``.
        n_lags: Highest lag to compute. Two days by default.

    Returns:
        ``lags``, ``acf``, and the 95% white-noise confidence bound.
    """
    residuals = np.asarray(observed, dtype=np.float64) - np.asarray(predicted)
    residuals = residuals - residuals.mean()
    n = residuals.size
    n_lags = min(n_lags, n - 1)

    denominator = float(np.dot(residuals, residuals))
    if denominator == 0:
        return {
            "lags": np.arange(n_lags + 1),
            "acf": np.full(n_lags + 1, np.nan),
            "confidence": 0.0,
        }

    acf = np.array(
        [
            float(np.dot(residuals[lag:], residuals[: n - lag]) / denominator)
            for lag in range(n_lags + 1)
        ]
    )
    return {"lags": np.arange(n_lags + 1), "acf": acf, "confidence": 1.96 / np.sqrt(n)}


def cross_correlation(
    observed: np.ndarray, predicted: np.ndarray, *, max_lag: int = 12
) -> dict[str, Any]:
    """Cross-correlation between forecasts and observations.

    A forecast that tracks the series in step peaks at lag 0. One that has
    collapsed to repeating its last input peaks at lag 1, because it reproduces
    the observation one step late. ``peak_lag`` is the diagnostic; the plan asks
    for this test explicitly.

    Args:
        observed: True values.
        predicted: Forecasts aligned to ``observed``.
        max_lag: Largest shift to test in each direction.

    Returns:
        ``lags``, ``correlation``, ``peak_lag`` and ``peak_correlation``.
    """
    x = np.asarray(observed, dtype=np.float64)
    y = np.asarray(predicted, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    scale = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if scale == 0:
        lags = np.arange(-max_lag, max_lag + 1)
        return {
            "lags": lags,
            "correlation": np.full(lags.size, np.nan),
            "peak_lag": 0,
            "peak_correlation": float("nan"),
        }

    lags = np.arange(-max_lag, max_lag + 1)
    correlation = np.empty(lags.size)
    for index, lag in enumerate(lags):
        # Positive lag: the forecast reproduces an earlier observation.
        if lag >= 0:
            correlation[index] = float(np.dot(y[lag:], x[: y.size - lag]) / scale)
        else:
            correlation[index] = float(np.dot(y[:lag], x[-lag:]) / scale)

    peak = int(np.nanargmax(correlation))
    return {
        "lags": lags,
        "correlation": correlation,
        "peak_lag": int(lags[peak]),
        "peak_correlation": float(correlation[peak]),
    }


@dataclass(frozen=True)
class CopyingReport:
    """Whether a forecast is doing more than repeating its last input."""

    model: str
    square_id: int
    peak_lag: int
    peak_correlation: float
    lag0_correlation: float
    lag1_correlation: float
    lag_margin: float
    copy_ratio: float
    verdict: str

    def to_row(self) -> dict[str, Any]:
        return {
            "square_id": self.square_id,
            "model": self.model,
            "peak_lag": self.peak_lag,
            "peak_correlation": round(self.peak_correlation, 5),
            "lag0_correlation": round(self.lag0_correlation, 5),
            "lag1_correlation": round(self.lag1_correlation, 5),
            "lag_margin": round(self.lag_margin, 5),
            "copy_ratio": round(self.copy_ratio, 4),
            "verdict": self.verdict,
        }


def copying_report(
    model: str,
    square_id: int,
    observed: np.ndarray,
    predicted: np.ndarray,
    persistence: np.ndarray,
) -> CopyingReport:
    """Test one forecast for collapse to persistence.

    The verdict rests on ``copy_ratio``: the mean distance between the forecast
    and the persistence baseline, divided by how far the series itself moves
    between steps. Zero means the two are the same forecast. One means the
    forecast departs from persistence by as much as the series moves.

    **The cross-correlation peak is reported but deliberately not used as the
    test.** A one-step forecast is built only from observations up to ``t-1``,
    so it cannot contain the innovation at ``t`` and will correlate slightly
    more with the previous observation than the current one. A lag-1 peak is
    therefore what a *causal* forecast looks like, not evidence of copying; a
    forecast peaking at lag 0 on a series this persistent would be the thing
    worth investigating, because it would suggest access to the present value.
    The measured data makes the point: seasonal naive is the only model here
    peaking at lag 0 and it is comfortably the worst of them.

    ``lag_margin`` (lag-1 minus lag-0 correlation) is kept because its *size*
    is informative even though its sign is not: persistence maximises it by
    construction, so a model approaching persistence's margin is behaving like
    persistence.

    Args:
        model: Model name, for the report row.
        square_id: Area identifier.
        observed: True values over the evaluation window.
        predicted: The model's forecasts.
        persistence: The persistence forecast over the same window.

    Returns:
        A :class:`CopyingReport` whose ``verdict`` is one of ``"collapsed"``,
        ``"near-persistence"`` or ``"independent"``.
    """
    correlation = cross_correlation(observed, predicted)
    lags = correlation["lags"]
    values = correlation["correlation"]
    lag0 = float(values[lags == 0][0])
    lag1 = float(values[lags == 1][0])

    movement = float(np.mean(np.abs(np.diff(np.asarray(observed, dtype=np.float64)))))
    distance = float(np.mean(np.abs(np.asarray(predicted) - np.asarray(persistence))))
    copy_ratio = distance / movement if movement else float("nan")

    if copy_ratio < 0.25:
        verdict = "collapsed"
    elif copy_ratio < 0.6:
        verdict = "near-persistence"
    else:
        verdict = "independent"

    return CopyingReport(
        model=model,
        square_id=square_id,
        peak_lag=int(correlation["peak_lag"]),
        peak_correlation=float(correlation["peak_correlation"]),
        lag0_correlation=lag0,
        lag1_correlation=lag1,
        lag_margin=lag1 - lag0,
        copy_ratio=copy_ratio,
        verdict=verdict,
    )


# --------------------------------------------------------------------------
# Where the model fails
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureWindow:
    """One contiguous stretch where a model did unusually badly."""

    model: str
    square_id: int
    start_index: int
    stop_index: int
    start_time: Any
    stop_time: Any
    mae: float
    baseline_mae: float

    @property
    def ratio(self) -> float:
        """How much worse than the persistence baseline over the same window."""
        return self.mae / self.baseline_mae if self.baseline_mae else float("nan")

    def to_row(self) -> dict[str, Any]:
        return {
            "square_id": self.square_id,
            "model": self.model,
            "start": str(self.start_time),
            "stop": str(self.stop_time),
            "n_steps": self.stop_index - self.start_index,
            "mae": round(self.mae, 3),
            "persistence_mae": round(self.baseline_mae, 3),
            "ratio_to_persistence": round(self.ratio, 3),
        }


def worst_windows(
    model: str,
    square_id: int,
    observed: np.ndarray,
    predicted: np.ndarray,
    persistence: np.ndarray,
    times: np.ndarray,
    *,
    window: int = 36,
    top: int = 3,
) -> list[FailureWindow]:
    """The worst non-overlapping contiguous windows by mean absolute error.

    Aggregate error hides episodes. A model can post a respectable weekly MAE
    while being badly wrong for six hours, and that episode is what a failure
    analysis is for.

    Windows are selected greedily and are not allowed to overlap, because the
    top few windows of a naive scan are otherwise the same episode shifted by
    one step each time.

    Args:
        model: Model name.
        square_id: Area identifier.
        observed: True values.
        predicted: Forecasts aligned to ``observed``.
        persistence: Persistence forecasts over the same window, for context.
        times: Local timestamps aligned to ``observed``.
        window: Window length in steps. 36 is six hours at 10-minute spacing.
        top: How many windows to return.

    Returns:
        Up to ``top`` windows, worst first.
    """
    observed = np.asarray(observed, dtype=np.float64)
    errors = np.abs(observed - np.asarray(predicted))
    baseline_errors = np.abs(observed - np.asarray(persistence))
    n = errors.size
    if n < window:
        return []

    # Rolling means via a cumulative sum: exact, and linear in n.
    cumulative = np.concatenate([[0.0], np.cumsum(errors)])
    rolling = (cumulative[window:] - cumulative[:-window]) / window
    baseline_cumulative = np.concatenate([[0.0], np.cumsum(baseline_errors)])
    baseline_rolling = (baseline_cumulative[window:] - baseline_cumulative[:-window]) / window

    order = np.argsort(rolling)[::-1]
    chosen: list[int] = []
    for start in order:
        if len(chosen) >= top:
            break
        if all(abs(int(start) - other) >= window for other in chosen):
            chosen.append(int(start))

    return [
        FailureWindow(
            model=model,
            square_id=square_id,
            start_index=start,
            stop_index=start + window,
            start_time=times[start],
            stop_time=times[start + window - 1],
            mae=float(rolling[start]),
            baseline_mae=float(baseline_rolling[start]),
        )
        for start in chosen
    ]


# --------------------------------------------------------------------------
# Comparing across areas
# --------------------------------------------------------------------------


def cross_area_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pivot per-area metric rows into one MASE table per model.

    MASE is the only metric comparable across these areas: their traffic differs
    by an order of magnitude, so a raw MAE of 83 on the busiest square and 13 on
    the quietest say nothing about which was forecast better. Each MASE is
    already normalised by that area's own in-sample naive error.

    Args:
        rows: Metric rows carrying ``square_id``, ``model`` and ``mase``.

    Returns:
        One row per model with a column per area, plus the mean across areas,
        sorted best first.
    """
    areas = sorted({int(r["square_id"]) for r in rows})
    by_model: dict[str, dict[int, float]] = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[int(row["square_id"])] = float(row["mase"])

    table = []
    for model, values in by_model.items():
        present = [values[a] for a in areas if a in values]
        entry: dict[str, Any] = {"model": model}
        for area in areas:
            entry[f"mase_{area}"] = round(values[area], 5) if area in values else None
        entry["mase_mean"] = round(float(np.mean(present)), 5) if present else None
        entry["mase_worst"] = round(float(np.max(present)), 5) if present else None
        entry["n_areas"] = len(present)
        table.append(entry)

    return sorted(table, key=lambda r: (r["mase_mean"] is None, r["mase_mean"]))


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _hours(times: np.ndarray) -> np.ndarray:
    """Hour of day for each timestamp."""
    return np.asarray(times).astype("datetime64[h]").astype(int) % 24


def _weekdays(times: np.ndarray) -> np.ndarray:
    """Day of week, Monday 0. 1970-01-01 was a Thursday, hence the +3."""
    days = np.asarray(times).astype("datetime64[D]").astype("int64")
    return (days + 3) % 7
