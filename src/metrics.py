"""Forecast accuracy metrics, reported in original units.

Every metric here is computed after inverse-transforming, on the scale the data
was measured in. A model trained on ``log1p`` values that reports its error in
log space is reporting a number nobody can interpret and that cannot be compared
against another model trained differently.

Two choices need justification.

**MASE is the cross-area metric.** The three study areas differ roughly fivefold
in mean volume, so MAE and RMSE cannot be compared across them -- a model can
look worse on the busiest area purely because the numbers are larger. MASE
scales by the in-sample seasonal-naive error [10], making it dimensionless:
below 1 means the model beats that baseline, above 1 means it does not.

**MAPE is reported with its denominators declared.** It is undefined at zero and
unstable near it, and quietly clipping small denominators turns an unstable
metric into a wrong one. This implementation reports the count of points below a
threshold and the value computed with and without them, so the reader can see
whether the figure is trustworthy rather than being asked to assume it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = ["Metrics", "evaluate", "mase_denominator"]


@dataclass(frozen=True)
class Metrics:
    """Accuracy of one model on one split, in original units."""

    n: int
    mae: float
    rmse: float
    mape: float
    mape_excluding_small: float
    n_small_denominators: int
    smape: float
    wape: float
    r2: float
    mase: float
    bias: float

    def to_row(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"MAE {self.mae:,.2f}  RMSE {self.rmse:,.2f}  "
            f"MAPE {self.mape:.2f}%  MASE {self.mase:.3f}  R2 {self.r2:.4f}"
        )


def _as_float(array: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def mase_denominator(y_train: np.ndarray, seasonal_period: int) -> float:
    """Mean absolute in-sample seasonal-naive error, the MASE scale factor.

    Computed on the *training* series, never on the evaluation window. Scaling
    by the error on the data being evaluated would let a hard test week make a
    model look better than it is.

    Args:
        y_train: The training series, in original units.
        seasonal_period: Lag the seasonal-naive predictor copies from.

    Returns:
        The mean absolute difference at ``seasonal_period`` lag.

    Raises:
        ValueError: If the training series is too short, or is perfectly
            periodic so that the scale would be zero.
    """
    values = _as_float(y_train, "y_train")
    if values.size <= seasonal_period:
        raise ValueError(
            f"training series of {values.size} is too short for seasonal period "
            f"{seasonal_period}"
        )
    denominator = float(np.mean(np.abs(values[seasonal_period:] - values[:-seasonal_period])))
    if denominator == 0.0:
        raise ValueError(
            "in-sample seasonal-naive error is zero, so MASE is undefined; "
            "the training series is exactly periodic"
        )
    return denominator


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_train: np.ndarray,
    seasonal_period: int = 144,
    mape_threshold: float = 1.0,
) -> Metrics:
    """Score a forecast against observations, in original units.

    Args:
        y_true: Observed values over the evaluation window.
        y_pred: Forecasts for the same window, same length.
        y_train: Training series, used only for the MASE scale factor.
        seasonal_period: Seasonal lag for the MASE denominator.
        mape_threshold: Observations below this are treated as small
            denominators and reported separately rather than clipped.

    Returns:
        A :class:`Metrics` with every quantity the report tabulates.

    Raises:
        ValueError: On length mismatch, empty input or non-finite values.
    """
    observed = _as_float(y_true, "y_true")
    predicted = _as_float(y_pred, "y_pred")
    if observed.size != predicted.size:
        raise ValueError(f"{observed.size} observations against {predicted.size} predictions")

    error = predicted - observed
    absolute = np.abs(error)

    mae = float(absolute.mean())
    rmse = float(np.sqrt(np.mean(error**2)))
    bias = float(error.mean())

    # MAPE, with the unstable points identified rather than hidden.
    small = np.abs(observed) < mape_threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = absolute / np.abs(observed)
    mape = (
        float(100.0 * np.mean(ratios[np.isfinite(ratios)]))
        if np.isfinite(ratios).any()
        else float("nan")
    )
    kept = ~small & np.isfinite(ratios)
    mape_excluding = float(100.0 * np.mean(ratios[kept])) if kept.any() else float("nan")

    # sMAPE on the convention with a factor of 2 in the denominator, so the
    # result is bounded at 200% rather than being unbounded like MAPE.
    denominator = np.abs(observed) + np.abs(predicted)
    with np.errstate(divide="ignore", invalid="ignore"):
        symmetric = np.where(denominator > 0, 2.0 * absolute / denominator, 0.0)
    smape = float(100.0 * symmetric.mean())

    total = float(np.abs(observed).sum())
    wape = float(100.0 * absolute.sum() / total) if total > 0 else float("nan")

    variance = float(np.sum((observed - observed.mean()) ** 2))
    r2 = float(1.0 - np.sum(error**2) / variance) if variance > 0 else float("nan")

    mase = mae / mase_denominator(y_train, seasonal_period)

    return Metrics(
        n=int(observed.size),
        mae=mae,
        rmse=rmse,
        mape=mape,
        mape_excluding_small=mape_excluding,
        n_small_denominators=int(small.sum()),
        smape=smape,
        wape=wape,
        r2=r2,
        mase=float(mase),
        bias=bias,
    )
