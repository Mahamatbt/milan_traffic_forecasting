"""Gradient-boosted trees on causal lag features.

Not a filler third model. The M5 competition was won by gradient-boosted trees
with LightGBM the most-used model among the winning entries [8], [9], which
makes this the approach that most recently beat both statistical and neural
methods at scale. Santos *et al.* [4] evaluated Random Forest and Decision Tree
on this dataset but not boosting, so it is also the gap this study occupies.

Its feature set is determined by measurement rather than convention. The
autocorrelation of the study area has local maxima at 144, 288, 432, 720, 864
and 1008 -- every multiple of the daily period, with the weekly lag among them --
and lag 1 is the strongest single predictor at 0.987. Those lags are what
``DEFAULT_LAGS`` contains.

That gives a diagnostic the other two models cannot offer: if the fitted
importances do not rest on the lags the autocorrelation identified, then either
the feature construction or the exploratory analysis is wrong, and the
disagreement is visible rather than silent.

One limitation to state plainly: a tree ensemble cannot extrapolate beyond the
range of its training targets. That matters for the held-out stress split, which
contains Christmas and New Year, and is a specific prediction about where this
model should struggle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.features import (
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    LogStandardScaler,
    lag_features,
)
from src.models.base import Forecaster

__all__ = ["GBMForecaster", "GBMConfig"]


@dataclass
class GBMConfig:
    """Boosting and feature settings for one run."""

    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 2000
    min_child_samples: int = 20
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 1
    lambda_l2: float = 0.0
    early_stopping_rounds: int = 100
    lags: tuple[int, ...] = DEFAULT_LAGS
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS
    use_calendar: bool = True
    use_log: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_child_samples": self.min_child_samples,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "lambda_l2": self.lambda_l2,
            "n_lags": len(self.lags),
            "max_lag": max(self.lags),
            "use_calendar": self.use_calendar,
            "use_log": self.use_log,
        }


class GBMForecaster(Forecaster):
    """LightGBM over lagged values, rolling statistics and calendar terms."""

    name = "lightgbm"

    def __init__(self, config: GBMConfig | None = None, *, seed: int = 0) -> None:
        self.config = config or GBMConfig()
        self.seed = seed
        self.scaler: LogStandardScaler | None = None
        self.booster: Any = None
        self.feature_names: list[str] = field(default_factory=list)  # type: ignore[assignment]
        self.feature_names = []
        self.best_iteration = 0
        self.fit_wall_s = 0.0

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _design(
        self, values: np.ndarray, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
        """Causal design matrix on the model's working scale."""
        assert self.scaler is not None or not self.config.use_log
        working = self.scaler.transform(values) if self.scaler else values
        return lag_features(
            working,
            times,
            lags=self.config.lags,
            rolling_windows=self.config.rolling_windows,
            include_calendar=self.config.use_calendar,
        )

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
    ) -> GBMForecaster:
        """Fit with early stopping on the validation split.

        ``n_estimators`` is an upper bound rather than a choice: the number of
        trees actually used is decided by early stopping, so the parameter does
        not need tuning alongside the learning rate.

        Args:
            train: Training observations in original units.
            train_times: Local timestamps for ``train``, required for lags.
            valid: Validation observations for early stopping.
            valid_times: Local timestamps for ``valid``.

        Returns:
            Self.
        """
        import lightgbm as lgb

        if train_times is None:
            raise ValueError("lag features require timestamps")

        train_values = np.asarray(train, dtype=np.float64)
        self.scaler = LogStandardScaler().fit(train_values) if self.config.use_log else None

        x_train, y_train, names, _ = self._design(train_values, np.asarray(train_times))
        self.feature_names = names

        callbacks = [lgb.log_evaluation(period=0)]
        eval_set = None
        if valid is not None and valid_times is not None:
            joined = np.concatenate([train_values, np.asarray(valid, dtype=np.float64)])
            joined_times = np.concatenate([np.asarray(train_times), np.asarray(valid_times)])
            all_x, all_y, _, index = self._design(joined, joined_times)
            inside = index >= train_values.size
            eval_set = [(all_x[inside], all_y[inside])]
            callbacks.append(lgb.early_stopping(self.config.early_stopping_rounds, verbose=False))

        model = lgb.LGBMRegressor(
            objective="l1",  # MAE, matching the selection metric
            num_leaves=self.config.num_leaves,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            min_child_samples=self.config.min_child_samples,
            colsample_bytree=self.config.feature_fraction,
            subsample=self.config.bagging_fraction,
            subsample_freq=self.config.bagging_freq,
            reg_lambda=self.config.lambda_l2,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )

        started = time.perf_counter()
        model.fit(x_train, y_train, eval_set=eval_set, eval_metric="l1", callbacks=callbacks)
        self.fit_wall_s = time.perf_counter() - started

        self.booster = model
        self.best_iteration = int(getattr(model, "best_iteration_", 0) or model.n_estimators)
        return self

    def n_params(self) -> int:
        """Total leaves across the ensemble, as a complexity measure.

        A tree ensemble has no parameter count comparable to a neural network's;
        leaf count is the closest honest analogue and is reported as such.
        """
        if self.booster is None:
            return 0
        dumped = self.booster.booster_.dump_model()
        return int(sum(tree["num_leaves"] for tree in dumped["tree_info"]))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_one_step(self, history: np.ndarray, **kwargs: Any) -> float:
        """Forecast one step from the tail of ``history``."""
        if self.booster is None:
            raise RuntimeError("model has not been fitted")
        history_times = kwargs.get("history_times")
        target_time = kwargs.get("target_time")
        if history_times is None or target_time is None:
            raise ValueError("lag features require history_times and target_time")

        values = np.concatenate([np.asarray(history, dtype=np.float64), [0.0]])
        times = np.concatenate([np.asarray(history_times), [target_time]])
        x, _, _, index = self._design(values, times)

        # The final row is the one whose target is the value being predicted.
        row = x[index == values.size - 1]
        scaled = float(self.booster.predict(row)[0])
        return float(
            self.scaler.inverse_transform(np.array([scaled]))[0] if self.scaler else scaled
        )

    def walk_forward(
        self,
        series: np.ndarray,
        start: int,
        stop: int,
        *,
        times: np.ndarray | None = None,
    ) -> np.ndarray:
        """Batched one-step-ahead forecasts across a window.

        Every feature for target ``t`` is built from observations strictly before
        ``t``, so the whole window can be scored in one pass. The equivalence
        with the step-by-step loop is asserted in ``tests/test_models.py``.
        """
        if self.booster is None:
            raise RuntimeError("model has not been fitted")
        if times is None:
            raise ValueError("lag features require timestamps")

        values = np.asarray(series, dtype=np.float64)
        x, _, _, index = self._design(values, np.asarray(times))
        wanted = (index >= start) & (index < stop)
        if int(wanted.sum()) != stop - start:
            raise ValueError(
                f"window [{start}, {stop}) needs {max(self.config.lags)} points of "
                "history; it starts too early in the series"
            )

        scaled = self.booster.predict(x[wanted])
        return self.scaler.inverse_transform(scaled) if self.scaler else np.asarray(scaled)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def importances(self, *, top: int = 15) -> list[tuple[str, float]]:
        """Feature importances by gain, highest first.

        Used as a cross-check on the exploratory analysis: the lags the
        autocorrelation identified should be the ones the model relies on.
        """
        if self.booster is None:
            raise RuntimeError("model has not been fitted")
        gains = self.booster.booster_.feature_importance(importance_type="gain")
        total = float(gains.sum()) or 1.0
        ranked = sorted(
            zip(self.feature_names, gains / total, strict=True),
            key=lambda pair: -pair[1],
        )
        return [(name, float(share)) for name, share in ranked[:top]]

    def describe(self) -> dict[str, Any]:
        """Configuration and fit diagnostics, for the experiment log."""
        return {
            **self.config.to_dict(),
            "seed": self.seed,
            "best_iteration": self.best_iteration,
            "n_leaves": self.n_params(),
            "n_features": len(self.feature_names),
            "fit_wall_s": round(self.fit_wall_s, 3),
        }
