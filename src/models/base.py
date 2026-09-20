"""The interface every forecaster implements, and the one inference path.

``walk_forward`` is the only route used for reported results. At each step the
model receives the *true* observed history up to time ``t`` and predicts
``t + 1``; the prediction it just made is discarded rather than fed back. That
is what one-step-ahead means, and it is not the same as a recursive rollout,
where errors compound and the reported figure answers a different question.

The distinction matters enough to be enforced here rather than left to each
model: a recursive implementation would produce plausible-looking numbers that
are not comparable with anything else in the study.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

__all__ = ["Forecaster", "walk_forward"]


class Forecaster(ABC):
    """Base class for every model in the comparison.

    Subclasses implement :meth:`fit` and :meth:`predict_one_step`. Models that
    can vectorise the evaluation window may override :meth:`walk_forward`, but
    must produce the same values a step-by-step loop would.
    """

    name: str = "forecaster"

    @abstractmethod
    def fit(
        self,
        train: np.ndarray,
        *,
        train_times: np.ndarray | None = None,
        valid: np.ndarray | None = None,
        valid_times: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Forecaster:
        """Fit on the training series, optionally using a validation split.

        Args:
            train: Training observations in original units.
            train_times: Local timestamps for ``train``.
            valid: Validation observations, for early stopping or selection.
            valid_times: Local timestamps for ``valid``.

        Returns:
            Self, for chaining.
        """

    @abstractmethod
    def predict_one_step(self, history: np.ndarray, **kwargs: Any) -> float:
        """Predict the next value given all observations up to and including now.

        Args:
            history: Every observation up to time ``t``, in original units. The
                model may use as much or as little of the tail as it needs.

        Returns:
            The forecast for ``t + 1``, in original units.
        """

    def n_params(self) -> int:
        """Number of fitted parameters, for the model-complexity comparison."""
        return 0

    def walk_forward(
        self,
        series: np.ndarray,
        start: int,
        stop: int,
        *,
        times: np.ndarray | None = None,
    ) -> np.ndarray:
        """Forecast each point in ``[start, stop)`` from true prior observations.

        Args:
            series: The full series, in original units.
            start: First index to forecast.
            stop: One past the last index to forecast.
            times: Local timestamps aligned to ``series``, for models that use
                calendar information.

        Returns:
            One prediction per index in ``[start, stop)``.
        """
        return walk_forward(self, series, start, stop, times=times)


def walk_forward(
    model: Forecaster,
    series: np.ndarray,
    start: int,
    stop: int,
    *,
    times: np.ndarray | None = None,
) -> np.ndarray:
    """Step through an evaluation window, predicting each point from real history.

    For target ``t`` the model is given ``series[:t]`` -- every observation
    strictly before ``t``, which is exactly what would be available in
    production at the moment the forecast is made.

    Args:
        model: A fitted forecaster.
        series: Full series in original units.
        start: First index to forecast.
        stop: One past the last index.
        times: Optional timestamps, passed through to the model.

    Returns:
        Predictions for ``[start, stop)``.

    Raises:
        ValueError: If the window is empty, out of range, or starts at zero
            where there would be no history at all.
    """
    values = np.asarray(series, dtype=np.float64)
    if not 0 < start < stop <= values.size:
        raise ValueError(
            f"window [{start}, {stop}) is not a valid range within a series of "
            f"{values.size}; start must be positive so that history exists"
        )

    predictions = np.empty(stop - start, dtype=np.float64)
    for offset, target in enumerate(range(start, stop)):
        history = values[:target]
        kwargs: dict[str, Any] = {}
        if times is not None:
            kwargs["history_times"] = np.asarray(times)[:target]
            kwargs["target_time"] = np.asarray(times)[target]
        predictions[offset] = model.predict_one_step(history, **kwargs)

    return predictions
