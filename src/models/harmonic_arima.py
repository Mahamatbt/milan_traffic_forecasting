"""Dynamic harmonic regression: Fourier terms with ARIMA errors.

Seasonal ARIMA is not usable on this data, for two reasons established in the
exploratory analysis. At 10-minute resolution the daily period is s = 144, where
seasonal differencing and parameter estimation become intractable; and SARIMA
accommodates one seasonal period where the decomposition found two, daily
carrying 82.9% of the variance and weekly a further 10.0%.

Dynamic harmonic regression is the standard answer [6]: represent each seasonal
period by a small number of Fourier terms entered as exogenous regressors, and
let an ARIMA model the remaining autocorrelation. Its cost is linear in the
number of terms rather than in the seasonal period, so two long periods are no
harder than one short one.

Two parameters come from measurement rather than from a default. The periodogram
shows a clear 12-hour peak alongside the 24-hour one, so the daily cycle is not
a single sinusoid and ``K1 >= 2`` is required. ADF and KPSS agree that the raw
series has no stochastic trend, so ``d = 0``.

Inference fits once and then appends. ``results.append(obs, refit=False)`` folds
each new observation into the state-space filter without re-estimating
parameters, which is what makes a 1,008-step walk-forward affordable and is also
the honest reading of one-step-ahead: the model is not permitted to retrain on
the evaluation window.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np

from src.features import LogStandardScaler
from src.models.base import Forecaster

__all__ = ["HarmonicARIMA", "fourier_terms"]


def fourier_terms(n: int, period: int, n_harmonics: int, *, phase: int = 0) -> np.ndarray:
    """Sine and cosine pairs for one seasonal period.

    Args:
        n: Number of time steps to generate.
        period: Seasonal period in samples.
        n_harmonics: Number of sine/cosine pairs. Harmonic ``k`` has period
            ``period / k``, so ``k = 2`` is the half-period component the
            periodogram identified.
        phase: Index offset, so terms generated for a later window continue the
            same cycle rather than restarting it.

    Returns:
        A ``(n, 2 * n_harmonics)`` array, columns ordered sin_1, cos_1, sin_2, ...

    Raises:
        ValueError: If more harmonics are requested than the period supports.
    """
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1")
    if 2 * n_harmonics > period:
        raise ValueError(
            f"{n_harmonics} harmonics need {2 * n_harmonics} terms, more than the "
            f"period {period} can distinguish"
        )

    t = np.arange(phase, phase + n, dtype=np.float64)
    columns = []
    for k in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * k * t / period
        columns.append(np.sin(angle))
        columns.append(np.cos(angle))
    return np.column_stack(columns)


class HarmonicARIMA(Forecaster):
    """SARIMAX with Fourier exogenous terms for two seasonal periods."""

    name = "harmonic_arima"

    def __init__(
        self,
        *,
        daily_period: int = 144,
        weekly_period: int = 1008,
        k_daily: int = 3,
        k_weekly: int = 2,
        order: tuple[int, int, int] = (2, 0, 1),
        use_log: bool = True,
    ) -> None:
        self.daily_period = daily_period
        self.weekly_period = weekly_period
        self.k_daily = k_daily
        self.k_weekly = k_weekly
        self.order = order
        self.use_log = use_log

        self.scaler: LogStandardScaler | None = None
        self.results_: Any = None
        self._train_size = 0
        self._appended = 0
        self.fit_wall_s = 0.0
        self.aic = float("nan")
        self.aicc = float("nan")
        self.bic = float("nan")

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _exog(self, n: int, phase: int = 0) -> np.ndarray:
        """Fourier design matrix for both periods."""
        daily = fourier_terms(n, self.daily_period, self.k_daily, phase=phase)
        weekly = fourier_terms(n, self.weekly_period, self.k_weekly, phase=phase)
        return np.column_stack([daily, weekly])

    def _to_model_scale(self, values: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return np.asarray(values, dtype=np.float64)
        return self.scaler.transform(values)

    def _to_original_scale(self, values: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return np.asarray(values, dtype=np.float64)
        return self.scaler.inverse_transform(values)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        train: np.ndarray,
        *,
        train_times: np.ndarray | None = None,
        valid: np.ndarray | None = None,
        valid_times: np.ndarray | None = None,
        **kwargs: Any,
    ) -> HarmonicARIMA:
        """Estimate the model once on the training series.

        Args:
            train: Training observations in original units. When a validation
                split is supplied it is concatenated, because the final fit is
                entitled to both; the caller controls what is passed.
            valid: Optional validation observations to include in the fit.

        Returns:
            Self.
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        values = np.asarray(train, dtype=np.float64)
        if valid is not None:
            values = np.concatenate([values, np.asarray(valid, dtype=np.float64)])

        self.scaler = LogStandardScaler().fit(values) if self.use_log else None
        endog = self._to_model_scale(values)
        exog = self._exog(endog.size)

        started = time.perf_counter()
        with warnings.catch_warnings():
            # Convergence chatter is expected while searching harmonic orders;
            # the selection criterion, not the optimiser's opinion, decides.
            warnings.simplefilter("ignore")
            model = SARIMAX(
                endog,
                exog=exog,
                order=self.order,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.results_ = model.fit(disp=False, maxiter=200)
        self.fit_wall_s = time.perf_counter() - started

        self._train_size = endog.size
        self._appended = 0
        self.aic = float(self.results_.aic)
        self.bic = float(self.results_.bic)
        self.aicc = self._compute_aicc(endog.size)
        return self

    def _compute_aicc(self, n: int) -> float:
        """AIC with the small-sample correction, used for harmonic selection.

        AIC alone tends to favour more Fourier terms than the data supports; the
        correction penalises additional parameters more sharply as the parameter
        count grows relative to the sample.
        """
        k = int(self.results_.params.size)
        if n - k - 1 <= 0:
            return float("inf")
        return float(self.aic + (2 * k * (k + 1)) / (n - k - 1))

    def n_params(self) -> int:
        return int(self.results_.params.size) if self.results_ is not None else 0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_one_step(self, history: np.ndarray, **kwargs: Any) -> float:
        """Forecast one step from the given history.

        Provided for interface completeness. :meth:`walk_forward` is what the
        evaluation uses, because appending observations one at a time is far
        cheaper than rebuilding the filter for every step.
        """
        values = np.asarray(history, dtype=np.float64)
        if self.results_ is None:
            raise RuntimeError("model has not been fitted")

        extra = values.size - self._train_size - self._appended
        if extra < 0:
            raise ValueError("history is shorter than the fitted series")

        state = self.results_
        if extra:
            tail = self._to_model_scale(values[-extra:])
            phase = self._train_size + self._appended
            state = state.append(tail, exog=self._exog(extra, phase=phase), refit=False)
            self._appended += extra
            self.results_ = state

        phase = self._train_size + self._appended
        forecast = state.forecast(steps=1, exog=self._exog(1, phase=phase))
        return float(self._to_original_scale(np.asarray(forecast))[0])

    def walk_forward(
        self,
        series: np.ndarray,
        start: int,
        stop: int,
        *,
        times: np.ndarray | None = None,
    ) -> np.ndarray:
        """One-step-ahead forecasts across a window, appending as it goes.

        The fitted parameters are held fixed. Each observed value is folded into
        the filter after its own forecast has been made, so no prediction is
        informed by its own target.
        """
        if self.results_ is None:
            raise RuntimeError("model has not been fitted")
        values = np.asarray(series, dtype=np.float64)
        if not 0 < start < stop <= values.size:
            raise ValueError(f"window [{start}, {stop}) is invalid for size {values.size}")

        # Bring the filter up to the step before the window.
        state = self.results_
        already = self._train_size + self._appended
        if start > already:
            gap = self._to_model_scale(values[already:start])
            state = state.append(gap, exog=self._exog(gap.size, phase=already), refit=False)
            already = start

        predictions = np.empty(stop - start, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for offset, target in enumerate(range(start, stop)):
                forecast = state.forecast(steps=1, exog=self._exog(1, phase=already))
                predictions[offset] = float(np.asarray(forecast)[0])
                # Fold in the true observation only after forecasting it.
                observed = self._to_model_scale(values[target : target + 1])
                state = state.append(observed, exog=self._exog(1, phase=already), refit=False)
                already += 1

        return self._to_original_scale(predictions)

    def describe(self) -> dict[str, Any]:
        """Configuration and fit diagnostics, for the experiment log."""
        return {
            "k_daily": self.k_daily,
            "k_weekly": self.k_weekly,
            "order": list(self.order),
            "use_log": self.use_log,
            "n_params": self.n_params(),
            "aic": self.aic,
            "aicc": self.aicc,
            "bic": self.bic,
            "fit_wall_s": round(self.fit_wall_s, 3),
        }
