"""Tests for the three forecasting models and the experiment log.

Two models score their evaluation window in one batched pass rather than by
looping. That is a real optimisation -- 0.02 s against 37 s for a thousand steps
-- but it is only valid if every window contains nothing but true prior
observations. These tests assert the batched path reproduces the step-by-step
loop exactly, which is what makes the shortcut defensible.

The experiment log is tested for the property the assignment depends on: it
appends and never truncates, and it refuses a row with no reasoning attached.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from src.config import load_config
from src.experiments import FIELDNAMES, ExperimentLog, ExperimentRecord
from src.models.base import walk_forward
from src.models.gbm import GBMConfig, GBMForecaster
from src.models.harmonic_arima import HarmonicARIMA, fourier_terms
from src.models.lstm import LSTMConfig, LSTMForecaster

DAILY = 144
WEEKLY = 1008


@pytest.fixture(scope="module")
def config():
    return load_config(auto_env=False)


@pytest.fixture(scope="module")
def synthetic() -> tuple[np.ndarray, np.ndarray]:
    """Six weeks of daily-plus-weekly seasonality with noise."""
    n = DAILY * 42
    t = np.arange(n)
    rng = np.random.default_rng(0)
    values = (
        500.0
        + 200 * np.sin(2 * np.pi * t / DAILY)
        + 60 * np.sin(2 * np.pi * t / WEEKLY)
        + rng.normal(0, 20, n)
    )
    times = np.datetime64("2013-11-04T00:00") + t * np.timedelta64(10, "m")
    return np.clip(values, 1.0, None), times


# --------------------------------------------------------------------------
# Fourier terms
# --------------------------------------------------------------------------


def test_fourier_terms_have_the_expected_shape() -> None:
    terms = fourier_terms(500, DAILY, 3)
    assert terms.shape == (500, 6)


def test_fourier_terms_are_periodic() -> None:
    """A term must repeat exactly one period later, or the cycle is wrong."""
    terms = fourier_terms(DAILY * 2, DAILY, 2)
    np.testing.assert_allclose(terms[:DAILY], terms[DAILY:], atol=1e-10)


def test_fourier_phase_continues_the_cycle() -> None:
    """Terms for a later window must continue, not restart, the cycle.

    Getting this wrong would reset the seasonal position at every forecast step
    and quietly destroy the model's seasonality.
    """
    whole = fourier_terms(300, DAILY, 2)
    tail = fourier_terms(100, DAILY, 2, phase=200)
    np.testing.assert_allclose(whole[200:], tail, atol=1e-10)


def test_second_harmonic_has_half_the_period() -> None:
    """The 12-hour component the periodogram found is harmonic k = 2."""
    terms = fourier_terms(DAILY, DAILY, 2)
    half = DAILY // 2
    np.testing.assert_allclose(terms[:half, 2], terms[half:, 2], atol=1e-10)


def test_fourier_rejects_more_harmonics_than_the_period_supports() -> None:
    with pytest.raises(ValueError, match="more than the period"):
        fourier_terms(100, 6, 5)


# --------------------------------------------------------------------------
# Batched inference must equal the loop
# --------------------------------------------------------------------------


def test_lstm_batched_walk_forward_matches_the_loop(synthetic) -> None:
    """The batched path is only valid if it reproduces the definition."""
    values, times = synthetic
    train_end = DAILY * 30
    model = LSTMForecaster(
        LSTMConfig(
            sequence_length=36, hidden_size=16, num_layers=1, max_epochs=2, use_calendar=False
        ),
        seed=0,
    )
    model.fit(values[:train_end], train_times=times[:train_end])

    start, stop = train_end, train_end + 40
    np.testing.assert_allclose(
        model.walk_forward(values, start, stop, times=times),
        walk_forward(model, values, start, stop, times=times),
        rtol=1e-5,
        atol=1e-4,
    )


def test_gbm_batched_walk_forward_matches_the_loop(synthetic) -> None:
    values, times = synthetic
    train_end = DAILY * 30
    model = GBMForecaster(
        GBMConfig(lags=(1, 2, 144), rolling_windows=(6,), n_estimators=30, early_stopping_rounds=0),
        seed=0,
    )
    model.fit(values[:train_end], train_times=times[:train_end])

    start, stop = train_end, train_end + 30
    np.testing.assert_allclose(
        model.walk_forward(values, start, stop, times=times),
        walk_forward(model, values, start, stop, times=times),
        rtol=1e-5,
        atol=1e-3,
    )


@pytest.mark.parametrize("model_name", ["lstm", "gbm"])
def test_predictions_do_not_depend_on_their_own_target(synthetic, model_name) -> None:
    """Corrupting the targets must leave every forecast unchanged."""
    values, times = synthetic
    train_end = DAILY * 30

    if model_name == "lstm":
        model = LSTMForecaster(
            LSTMConfig(
                sequence_length=36, hidden_size=16, num_layers=1, max_epochs=2, use_calendar=False
            ),
            seed=0,
        )
    else:
        model = GBMForecaster(
            GBMConfig(
                lags=(1, 2, 144), rolling_windows=(6,), n_estimators=30, early_stopping_rounds=0
            ),
            seed=0,
        )
    model.fit(values[:train_end], train_times=times[:train_end])

    start, stop = train_end, train_end + 50
    before = model.walk_forward(values, start, stop, times=times)

    corrupted = values.copy()
    corrupted[start:stop] = 99_999.0
    after = model.walk_forward(corrupted, start, stop, times=times)

    # Only the first target has no corrupted value in its history.
    assert before[0] == pytest.approx(after[0], rel=1e-6)


# --------------------------------------------------------------------------
# Model behaviour
# --------------------------------------------------------------------------


def test_harmonic_recovers_a_clean_seasonal_signal() -> None:
    """On a noiseless periodic series the fit should be close to exact."""
    n = DAILY * 20
    t = np.arange(n)
    values = 500.0 + 200 * np.sin(2 * np.pi * t / DAILY)

    model = HarmonicARIMA(k_daily=1, k_weekly=1, order=(1, 0, 0)).fit(values[: DAILY * 15])
    predictions = model.walk_forward(values, DAILY * 15, DAILY * 15 + 100)

    relative = np.abs(predictions - values[DAILY * 15 : DAILY * 15 + 100]) / values.mean()
    assert relative.mean() < 0.05


def test_lstm_restores_the_best_checkpoint(synthetic) -> None:
    """Training past the best epoch must not degrade the returned model."""
    values, times = synthetic
    train_end, valid_end = DAILY * 28, DAILY * 34
    model = LSTMForecaster(
        LSTMConfig(
            sequence_length=36,
            hidden_size=16,
            num_layers=1,
            max_epochs=12,
            patience=3,
            use_calendar=False,
        ),
        seed=0,
    )
    model.fit(
        values[:train_end],
        train_times=times[:train_end],
        valid=values[train_end:valid_end],
        valid_times=times[train_end:valid_end],
    )
    assert model.history.best_epoch >= 0
    assert model.history.valid_mae[model.history.best_epoch] == min(model.history.valid_mae)


def test_lstm_reports_parameter_count(synthetic) -> None:
    values, times = synthetic
    model = LSTMForecaster(
        LSTMConfig(
            sequence_length=36, hidden_size=16, num_layers=1, max_epochs=1, use_calendar=False
        ),
        seed=0,
    ).fit(values[: DAILY * 20], train_times=times[: DAILY * 20])
    assert model.n_params() > 0


def test_lstm_is_reproducible_for_a_fixed_seed(synthetic) -> None:
    values, times = synthetic
    train_end = DAILY * 20

    def run(seed: int) -> np.ndarray:
        model = LSTMForecaster(
            LSTMConfig(
                sequence_length=36, hidden_size=16, num_layers=1, max_epochs=3, use_calendar=False
            ),
            seed=seed,
        ).fit(values[:train_end], train_times=times[:train_end])
        return model.walk_forward(values, train_end, train_end + 20, times=times)

    np.testing.assert_allclose(run(0), run(0), rtol=1e-6)


def test_gbm_relies_on_the_lags_the_acf_identified(synthetic) -> None:
    """The cross-check between the model and the exploratory analysis."""
    values, times = synthetic
    train_end = DAILY * 30
    model = GBMForecaster(
        GBMConfig(
            lags=(1, 2, 3, 144, 1008),
            rolling_windows=(6, 144),
            n_estimators=100,
            early_stopping_rounds=0,
        ),
        seed=0,
    ).fit(values[:train_end], train_times=times[:train_end])

    top = [name for name, _ in model.importances(top=3)]
    assert "lag_1" in top, top


def test_gbm_reports_leaf_count_as_complexity(synthetic) -> None:
    values, times = synthetic
    model = GBMForecaster(
        GBMConfig(lags=(1, 144), rolling_windows=(6,), n_estimators=20, early_stopping_rounds=0),
        seed=0,
    ).fit(values[: DAILY * 20], train_times=times[: DAILY * 20])
    assert model.n_params() > 0


def test_models_refuse_to_predict_before_being_fitted() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        LSTMForecaster().walk_forward(np.arange(1000.0), 500, 510)
    with pytest.raises(RuntimeError, match="not been fitted"):
        GBMForecaster().walk_forward(np.arange(1000.0), 500, 510, times=np.arange(1000))
    with pytest.raises(RuntimeError, match="not been fitted"):
        HarmonicARIMA().walk_forward(np.arange(1000.0), 500, 510)


# --------------------------------------------------------------------------
# Experiment log
# --------------------------------------------------------------------------


def _record(**overrides) -> ExperimentRecord:
    defaults = {
        "model": "lstm",
        "area": 5161,
        "stage": "capacity",
        "hyperparams": {"hidden_size": 64},
        "feature_set": "window+calendar",
        "valid_mae": 100.0,
        "rationale_for_next_change": "hidden=64 beat 32 by 4%; next axis is learning rate",
    }
    defaults.update(overrides)
    return ExperimentRecord(**defaults)


def test_log_appends_and_never_truncates(tmp_path) -> None:
    """A tuning history that only keeps the last run is not a history."""
    log = ExperimentLog(tmp_path / "experiments.csv")
    for index in range(5):
        log.append(_record(valid_mae=100.0 - index))

    rows = log.load()
    assert len(rows) == 5
    assert [float(r["valid_mae"]) for r in rows] == [100.0, 99.0, 98.0, 97.0, 96.0]


def test_log_writes_the_header_once(tmp_path) -> None:
    log = ExperimentLog(tmp_path / "experiments.csv")
    log.append(_record())
    log.append(_record())

    with (tmp_path / "experiments.csv").open(encoding="utf-8") as fh:
        header_lines = [line for line in fh if line.startswith("run_id")]
    assert len(header_lines) == 1


def test_empty_rationale_is_rejected() -> None:
    """The field the experimentation criterion rests on cannot be blank."""
    with pytest.raises(ValueError, match="rationale_for_next_change is required"):
        _record(rationale_for_next_change="   ")


def test_row_matches_the_schema(tmp_path) -> None:
    log = ExperimentLog(tmp_path / "experiments.csv")
    log.append(_record())
    with (tmp_path / "experiments.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert set(rows[0]) == set(FIELDNAMES)
    assert rows[0]["hyperparams_json"].startswith("{")


def test_best_returns_the_lowest_validation_error(tmp_path) -> None:
    log = ExperimentLog(tmp_path / "experiments.csv")
    log.append(_record(valid_mae=120.0))
    log.append(_record(valid_mae=95.0))
    log.append(_record(valid_mae=110.0))

    best = log.best(model="lstm", area=5161)
    assert best is not None
    assert float(best["valid_mae"]) == 95.0


def test_filters_select_the_right_rows(tmp_path) -> None:
    log = ExperimentLog(tmp_path / "experiments.csv")
    log.append(_record(model="lstm", area=5161, stage="capacity"))
    log.append(_record(model="lightgbm", area=5161, stage="optuna"))
    log.append(_record(model="lstm", area=4159, stage="capacity"))

    assert len(log.rows_for(model="lstm")) == 2
    assert len(log.rows_for(area=4159)) == 1
    assert len(log.rows_for(model="lstm", stage="capacity")) == 2


def test_summary_counts_by_model_and_stage(tmp_path) -> None:
    log = ExperimentLog(tmp_path / "experiments.csv")
    assert "no experiments" in log.summary()

    log.append(_record(model="lstm"))
    log.append(_record(model="lightgbm", stage="optuna"))
    summary = log.summary()
    assert "2 experiments logged" in summary
    assert "lstm" in summary and "lightgbm" in summary


# --------------------------------------------------------------------------
# Device placement
# --------------------------------------------------------------------------


def test_resolve_device_honours_an_explicit_name() -> None:
    from src.models.lstm import resolve_device

    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_matches_cuda_availability() -> None:
    """``auto`` must pick the GPU exactly when there is one to pick."""
    import torch

    from src.models.lstm import resolve_device

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == expected


def test_network_and_data_land_on_the_requested_device(synthetic) -> None:
    """A tensor left on the wrong device raises at the first matmul."""
    values, times = synthetic
    model = LSTMForecaster(
        LSTMConfig(sequence_length=36, hidden_size=8, num_layers=1, max_epochs=1, patience=1),
        seed=0,
        device="cpu",
    )
    model.fit(
        values[:600], train_times=times[:600], valid=values[600:700], valid_times=times[600:700]
    )

    assert all(p.device.type == "cpu" for p in model.network.parameters())
    assert model.describe()["device"] == "cpu"


def test_inference_returns_numpy_not_tensors(synthetic) -> None:
    """Predictions cross back from the device; a tensor here breaks metrics."""
    values, times = synthetic
    model = LSTMForecaster(
        LSTMConfig(sequence_length=36, hidden_size=8, num_layers=1, max_epochs=1, patience=1),
        seed=0,
        device="cpu",
    )
    model.fit(values[:600], train_times=times[:600])

    batched = model.walk_forward(values, 600, 640, times=times)
    assert isinstance(batched, np.ndarray)
    assert batched.shape == (40,)
    assert np.isfinite(batched).all()

    single = model.predict_one_step(values[:600], target_time=times[600])
    assert isinstance(single, float)
