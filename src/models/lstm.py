"""LSTM forecaster.

The one model in the comparison that assumes nothing about seasonal form. The
harmonic regression imposes the cycle explicitly and LightGBM is told which lags
to look at; this learns whatever structure it finds in a raw window. That makes
it the only candidate for the behaviour the decomposition could not remove --
the residual is still 3.6% of variance and strongly heteroscedastic, with spread
varying 25-fold across the day.

Two things it needs protecting from.

**Overfitting.** 5,472 training points is modest for this capacity, so training
stops on validation MAE with patience and the best checkpoint is restored rather
than the last. Taking the final epoch would report a model that had already
started memorising.

**Collapsing to persistence.** The lag-1 autocorrelation is 0.987, so a network
can reach a low error by learning to echo its most recent input. It would look
successful while having learned nothing. The evaluation compares against the
persistence baseline for exactly this reason, and :meth:`lag1_copy_ratio`
measures how close the model's behaviour is to simply repeating its input.

Walk-forward is batched, not looped. The window for target ``t`` is
``series[t - L : t]`` -- entirely true observations -- so every window in the
evaluation period can be built up front and scored in one pass. That is the same
computation the step-by-step loop performs, and the tests check it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.features import LogStandardScaler, calendar_features, make_windows
from src.models.base import Forecaster

__all__ = ["LSTMForecaster", "LSTMConfig"]


@dataclass
class LSTMConfig:
    """Architecture and optimisation settings for one run."""

    sequence_length: int = 144
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 8
    weight_decay: float = 0.0
    use_calendar: bool = True
    grad_clip: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "weight_decay": self.weight_decay,
            "use_calendar": self.use_calendar,
        }


@dataclass
class TrainingHistory:
    """Per-epoch losses, so the stopping decision can be inspected."""

    train_loss: list[float] = field(default_factory=list)
    valid_mae: list[float] = field(default_factory=list)
    best_epoch: int = -1
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs_run": len(self.train_loss),
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
            "best_valid_mae_scaled": min(self.valid_mae) if self.valid_mae else float("nan"),
        }


def resolve_device(name: str = "auto") -> Any:
    """Pick the execution device, preferring a GPU when one is present.

    This is an execution detail rather than a hyperparameter, so it is not part
    of :class:`LSTMConfig` and does not enter the search. It is recorded in
    :meth:`LSTMForecaster.describe` because the timing column in the results
    table is meaningless without knowing what produced it.

    The recurrence here is 288 sequential steps over small matrices, which CPU
    threads cannot parallelise -- measured at 0.88 of 8 cores. A GPU is not an
    optimisation but the difference between hours and minutes per fit.
    """
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_network(input_size: int, config: LSTMConfig) -> Any:
    """Stacked LSTM with a linear head, on the scaled target."""
    import torch
    from torch import nn

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                # PyTorch applies dropout between layers only, so it does
                # nothing with a single layer and warns if asked.
                dropout=config.dropout if config.num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(config.dropout)
            self.head = nn.Linear(config.hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output, _ = self.lstm(x)
            # The final step's hidden state summarises the whole window.
            return self.head(self.dropout(output[:, -1, :])).squeeze(-1)

    return Network()


class LSTMForecaster(Forecaster):
    """Stacked LSTM over a window of recent observations."""

    name = "lstm"

    def __init__(
        self, config: LSTMConfig | None = None, *, seed: int = 0, device: str = "auto"
    ) -> None:
        self.config = config or LSTMConfig()
        self.seed = seed
        self.device_name = device
        self.scaler: LogStandardScaler | None = None
        self.network: Any = None
        self.history = TrainingHistory()
        self.fit_wall_s = 0.0
        self._input_size = 1
        self._device: Any = None

    @property
    def device(self) -> Any:
        """The resolved torch device, decided once and reused."""
        if self._device is None:
            self._device = resolve_device(self.device_name)
        return self._device

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _exog(self, times: np.ndarray | None, n: int) -> np.ndarray | None:
        """Calendar features aligned to a series, or None if disabled."""
        if not self.config.use_calendar or times is None:
            return None
        features = calendar_features(times)
        if features.shape[0] != n:
            raise ValueError(f"{features.shape[0]} calendar rows against {n} observations")
        return features

    def _windows(
        self, values: np.ndarray, times: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scaled sliding windows and their targets."""
        assert self.scaler is not None
        scaled = self.scaler.transform(values)
        return make_windows(
            scaled,
            self.config.sequence_length,
            horizon=1,
            exog=self._exog(times, values.size),
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
    ) -> LSTMForecaster:
        """Train with early stopping on validation MAE.

        The scaler is fitted on ``train`` alone, never on ``valid``. Validation
        windows are built from the concatenated series so that the first
        validation target has a full window of real history behind it rather
        than being dropped.

        Args:
            train: Training observations in original units.
            train_times: Local timestamps for ``train``.
            valid: Validation observations, used only to decide when to stop.
            valid_times: Local timestamps for ``valid``.

        Returns:
            Self, with the best-validation weights restored.
        """
        import torch
        from torch import nn

        from src.seeding import seed_all

        seed_all(self.seed)
        device = self.device

        train_values = np.asarray(train, dtype=np.float64)
        self.scaler = LogStandardScaler().fit(train_values)

        x_train, y_train = self._windows(train_values, train_times)
        self._input_size = x_train.shape[2]
        self.network = _build_network(self._input_size, self.config).to(device)

        x_valid = y_valid = None
        if valid is not None:
            joined = np.concatenate([train_values, np.asarray(valid, dtype=np.float64)])
            joined_times = (
                np.concatenate([train_times, valid_times])
                if train_times is not None and valid_times is not None
                else None
            )
            all_x, all_y = self._windows(joined, joined_times)
            # Keep only windows whose target falls inside the validation split.
            first = train_values.size - self.config.sequence_length
            x_valid, y_valid = all_x[first:], all_y[first:]

        optimiser = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.L1Loss()  # MAE, matching the selection metric

        # The whole training set is a few tens of MB, so it is moved to the
        # device once rather than transferred batch by batch every epoch.
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x_train).to(device), torch.from_numpy(y_train).to(device)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False
        )
        valid_tensor = None if x_valid is None else torch.from_numpy(x_valid).to(device)

        best_state: dict[str, Any] | None = None
        best_mae = float("inf")
        since_improved = 0
        self.history = TrainingHistory()

        started = time.perf_counter()
        for epoch in range(self.config.max_epochs):
            self.network.train()
            running = 0.0
            for batch_x, batch_y in loader:
                optimiser.zero_grad()
                loss = loss_fn(self.network(batch_x), batch_y)
                loss.backward()
                if self.config.grad_clip:
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip)
                optimiser.step()
                running += float(loss.item()) * batch_x.shape[0]
            self.history.train_loss.append(running / len(dataset))

            if x_valid is None:
                continue

            self.network.eval()
            with torch.no_grad():
                predicted = self.network(valid_tensor).cpu().numpy()
            mae = float(np.mean(np.abs(predicted - y_valid)))
            self.history.valid_mae.append(mae)

            if mae < best_mae - 1e-6:
                best_mae = mae
                best_state = {k: v.detach().clone() for k, v in self.network.state_dict().items()}
                self.history.best_epoch = epoch
                since_improved = 0
            else:
                since_improved += 1
                if since_improved >= self.config.patience:
                    self.history.stopped_early = True
                    break

        # CUDA queues work asynchronously, so the clock must wait for it or the
        # reported training time is the time spent issuing kernels, not running
        # them -- and the timing comparison in the results table would be wrong.
        if device.type == "cuda":
            torch.cuda.synchronize()
        self.fit_wall_s = time.perf_counter() - started

        # Restore the best checkpoint; the final epoch is usually not the best.
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()
        return self

    def n_params(self) -> int:
        if self.network is None:
            return 0
        return int(sum(p.numel() for p in self.network.parameters()))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_one_step(self, history: np.ndarray, **kwargs: Any) -> float:
        """Forecast one step from a window at the end of ``history``."""
        import torch

        if self.network is None or self.scaler is None:
            raise RuntimeError("model has not been fitted")

        values = np.asarray(history, dtype=np.float64)
        length = self.config.sequence_length
        if values.size < length:
            raise ValueError(f"need {length} observations, got {values.size}")

        window = self.scaler.transform(values[-length:]).astype(np.float32)[:, None]
        if self.config.use_calendar:
            target_time = kwargs.get("target_time")
            if target_time is None:
                raise ValueError("calendar features require target_time")
            extra = calendar_features(np.asarray([target_time]))
            window = np.concatenate([window, np.repeat(extra, length, axis=0)], axis=1)

        with torch.no_grad():
            tensor = torch.from_numpy(window[None, ...]).to(self.device)
            scaled = float(self.network(tensor).item())
        return float(self.scaler.inverse_transform(np.array([scaled]))[0])

    def walk_forward(
        self,
        series: np.ndarray,
        start: int,
        stop: int,
        *,
        times: np.ndarray | None = None,
    ) -> np.ndarray:
        """Batched one-step-ahead forecasts across a window.

        Each input window is ``series[t - L : t]``, all true observations, so the
        windows are independent and can be scored in one pass. This is the same
        computation the step-by-step loop performs; ``tests/test_models.py``
        asserts the two agree.
        """
        import torch

        if self.network is None or self.scaler is None:
            raise RuntimeError("model has not been fitted")
        values = np.asarray(series, dtype=np.float64)
        length = self.config.sequence_length
        if start < length:
            raise ValueError(f"window starts at {start}, before {length} points of history")
        if not start < stop <= values.size:
            raise ValueError(f"window [{start}, {stop}) is invalid for size {values.size}")

        scaled = self.scaler.transform(values).astype(np.float32)
        targets = np.arange(start, stop)
        offsets = np.arange(-length, 0)
        batch = scaled[targets[:, None] + offsets[None, :]][..., None]

        if self.config.use_calendar:
            if times is None:
                raise ValueError("calendar features require timestamps")
            extra = calendar_features(np.asarray(times)[targets])
            batch = np.concatenate([batch, np.repeat(extra[:, None, :], length, axis=1)], axis=2)

        with torch.no_grad():
            predicted = self.network(torch.from_numpy(batch).to(self.device)).cpu().numpy()
        return self.scaler.inverse_transform(predicted)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def lag1_copy_ratio(
        self, series: np.ndarray, start: int, stop: int, times: np.ndarray | None = None
    ) -> float:
        """How close the forecasts are to simply repeating the last observation.

        Returns the mean absolute difference between the model's predictions and
        the persistence forecast, divided by the mean absolute change in the
        series. A value near zero means the model has collapsed to persistence;
        near or above one means it is doing something else.
        """
        predictions = self.walk_forward(series, start, stop, times=times)
        values = np.asarray(series, dtype=np.float64)
        persistence = values[start - 1 : stop - 1]
        movement = float(np.mean(np.abs(np.diff(values[start - 1 : stop]))))
        if movement == 0:
            return float("nan")
        return float(np.mean(np.abs(predictions - persistence)) / movement)

    def describe(self) -> dict[str, Any]:
        """Configuration and training diagnostics, for the experiment log."""
        return {
            **self.config.to_dict(),
            "seed": self.seed,
            "n_params": self.n_params(),
            "input_size": self._input_size,
            # Recorded because fit_wall_s is not comparable across devices.
            "device": str(self._device) if self._device is not None else self.device_name,
            **self.history.to_dict(),
            "fit_wall_s": round(self.fit_wall_s, 3),
        }
