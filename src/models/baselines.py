"""Reference baselines that appear in every results table.

These are not among the three models under comparison. They are the floor: a
model that does not beat them has learned nothing useful, and reporting it
without them would let a mediocre result look respectable.

Persistence is a demanding opponent on this data rather than a formality. The
lag-1 autocorrelation of the highest-traffic area is 0.987, so simply repeating
the last observation is already a strong one-step-ahead forecast. That is also
why it is the reference for the lag-1 copying check in the evaluation: a neural
model can reach a low error by learning to echo its input, and would look
successful while having learned nothing at all.

Seasonal naive is the second reference and the basis of MASE. It copies the
value from the same time on the previous day, so it captures the daily cycle
exactly and nothing else -- which makes it the natural test of whether a model
contributes anything beyond that cycle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models.base import Forecaster

__all__ = ["Persistence", "SeasonalNaive", "BASELINES"]


class Persistence(Forecaster):
    """Predict the next value as the most recent observation.

    ``x_hat(t + 1) = x(t)``
    """

    name = "persistence"

    def fit(
        self,
        train: np.ndarray,
        *,
        train_times: np.ndarray | None = None,
        valid: np.ndarray | None = None,
        valid_times: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Persistence:
        """No parameters to estimate; present so the interface is uniform."""
        return self

    def predict_one_step(self, history: np.ndarray, **kwargs: Any) -> float:
        values = np.asarray(history, dtype=np.float64)
        if values.size == 0:
            raise ValueError("persistence needs at least one prior observation")
        return float(values[-1])

    def walk_forward(
        self,
        series: np.ndarray,
        start: int,
        stop: int,
        *,
        times: np.ndarray | None = None,
    ) -> np.ndarray:
        """Vectorised equivalent of the step-by-step loop."""
        values = np.asarray(series, dtype=np.float64)
        if not 0 < start < stop <= values.size:
            raise ValueError(f"window [{start}, {stop}) is invalid for size {values.size}")
        return values[start - 1 : stop - 1].copy()


class SeasonalNaive(Forecaster):
    """Predict the value observed one seasonal period earlier.

    ``x_hat(t + 1) = x(t + 1 - m)``, with ``m`` the seasonal period.
    """

    name = "seasonal_naive"

    def __init__(self, seasonal_period: int = 144) -> None:
        if seasonal_period < 1:
            raise ValueError("seasonal_period must be >= 1")
        self.seasonal_period = seasonal_period

    def fit(
        self,
        train: np.ndarray,
        *,
        train_times: np.ndarray | None = None,
        valid: np.ndarray | None = None,
        valid_times: np.ndarray | None = None,
        **kwargs: Any,
    ) -> SeasonalNaive:
        """No parameters to estimate; the period is given, not learned."""
        return self

    def predict_one_step(self, history: np.ndarray, **kwargs: Any) -> float:
        values = np.asarray(history, dtype=np.float64)
        # Predicting index t from history[:t] means reaching back
        # seasonal_period from t, which is seasonal_period - 1 from the end.
        offset = self.seasonal_period - 1
        if values.size <= offset:
            raise ValueError(
                f"seasonal naive needs more than {offset} prior observations, " f"got {values.size}"
            )
        return float(values[values.size - 1 - offset])

    def walk_forward(
        self,
        series: np.ndarray,
        start: int,
        stop: int,
        *,
        times: np.ndarray | None = None,
    ) -> np.ndarray:
        """Vectorised equivalent of the step-by-step loop."""
        values = np.asarray(series, dtype=np.float64)
        if not 0 < start < stop <= values.size:
            raise ValueError(f"window [{start}, {stop}) is invalid for size {values.size}")
        if start < self.seasonal_period:
            raise ValueError(
                f"window starts at {start}, before a full seasonal period of "
                f"{self.seasonal_period} is available"
            )
        return values[start - self.seasonal_period : stop - self.seasonal_period].copy()


BASELINES: tuple[type[Forecaster], ...] = (Persistence, SeasonalNaive)
