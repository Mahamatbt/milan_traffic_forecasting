"""Tests for the final-fit stage.

The property that matters most here is what the final fit is allowed to see.
Hyperparameters were selected on validation, so the refit may use train plus
validation -- but never the test week, and never an early-stopping rule
evaluated against it. These tests pin that boundary, and the reporting of seed
spread that makes a single neural result interpretable.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.final_runs import FinalResult, _lag1_copy_ratio, build_model
from src.metrics import Metrics
from src.models.gbm import GBMForecaster
from src.models.harmonic_arima import HarmonicARIMA
from src.models.lstm import LSTMForecaster

DAILY = 144


@pytest.fixture(scope="module")
def config():
    return load_config(auto_env=False)


def _metrics(mae: float, mase: float) -> Metrics:
    return Metrics(
        n=1008,
        mae=mae,
        rmse=mae * 1.4,
        mape=5.0,
        mape_excluding_small=5.0,
        n_small_denominators=0,
        smape=5.0,
        wape=5.0,
        r2=0.98,
        mase=mase,
        bias=0.1,
    )


# --------------------------------------------------------------------------
# Model construction from stored configuration
# --------------------------------------------------------------------------


def test_builds_each_model_from_its_stored_block(config) -> None:
    harmonic = build_model(
        "harmonic_arima", {"k_daily": 6, "k_weekly": 2, "order": [2, 0, 1]}, config, seed=0
    )
    assert isinstance(harmonic, HarmonicARIMA)
    assert harmonic.k_daily == 6
    assert harmonic.order == (2, 0, 1)

    lstm = build_model("lstm", {"hidden_size": 128, "sequence_length": 288}, config, seed=1)
    assert isinstance(lstm, LSTMForecaster)
    assert lstm.config.hidden_size == 128
    assert lstm.seed == 1

    gbm = build_model("lightgbm", {"num_leaves": 63, "learning_rate": 0.03}, config, seed=2)
    assert isinstance(gbm, GBMForecaster)
    assert gbm.config.num_leaves == 63


def test_extra_keys_in_the_stored_block_are_ignored(config) -> None:
    """describe() records diagnostics alongside parameters; only the latter apply."""
    model = build_model(
        "lstm",
        {"hidden_size": 32, "best_epoch": 7, "epochs_run": 20, "fit_wall_s": 3.2},
        config,
        seed=0,
    )
    assert model.config.hidden_size == 32


def test_unknown_model_is_rejected(config) -> None:
    with pytest.raises(KeyError, match="unknown model"):
        build_model("transformer", {}, config, seed=0)


# --------------------------------------------------------------------------
# Seed spread
# --------------------------------------------------------------------------


def test_single_seed_reports_zero_spread() -> None:
    result = FinalResult(square_id=5161, model="lightgbm", split="test")
    result.metrics = [_metrics(100.0, 0.30)]
    result.predictions = [np.zeros(1008)]
    result.inference_wall_s = [0.1]

    row = result.to_row()
    assert row["n_seeds"] == 1
    assert row["mae_std"] == 0.0


def test_multiple_seeds_report_mean_and_standard_deviation() -> None:
    """A single neural result is one draw; the spread is what makes it readable."""
    result = FinalResult(square_id=5161, model="lstm", split="test")
    result.metrics = [_metrics(100.0, 0.30), _metrics(110.0, 0.33), _metrics(105.0, 0.315)]
    result.predictions = [np.full(1008, v) for v in (100.0, 110.0, 105.0)]
    result.inference_wall_s = [0.1, 0.1, 0.1]
    result.train_wall_s = [60.0, 62.0, 61.0]

    row = result.to_row()
    assert row["n_seeds"] == 3
    assert row["mae"] == pytest.approx(105.0)
    assert row["mae_std"] == pytest.approx(5.0)
    assert row["train_wall_s"] == pytest.approx(61.0)
    assert row["train_wall_std"] > 0


def test_mean_predictions_average_across_seeds() -> None:
    result = FinalResult(square_id=1, model="lstm", split="test")
    result.predictions = [np.full(10, 2.0), np.full(10, 4.0)]
    np.testing.assert_allclose(result.mean_predictions, np.full(10, 3.0))


def test_row_carries_timing_and_complexity() -> None:
    result = FinalResult(square_id=1, model="lstm", split="test")
    result.metrics = [_metrics(100.0, 0.3)]
    result.predictions = [np.zeros(1008)]
    result.inference_wall_s = [2.0]
    result.n_params = 52_033

    row = result.to_row()
    assert row["n_params"] == 52_033
    assert row["inference_ms_per_step"] == pytest.approx(1000 * 2.0 / 1008, rel=1e-3)


# --------------------------------------------------------------------------
# The persistence-collapse diagnostic
# --------------------------------------------------------------------------


def test_copy_ratio_is_zero_for_a_persistence_forecast() -> None:
    """A model that exactly repeats its input must be detected as doing so."""
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.normal(0, 1, 500)) + 100
    start, stop = 200, 300

    persistence = series[start - 1 : stop - 1]
    assert _lag1_copy_ratio(persistence, series, start, stop) == pytest.approx(0.0, abs=1e-12)


def test_copy_ratio_grows_as_forecasts_depart_from_persistence() -> None:
    rng = np.random.default_rng(1)
    series = np.cumsum(rng.normal(0, 1, 500)) + 100
    start, stop = 200, 300
    persistence = series[start - 1 : stop - 1]

    near = _lag1_copy_ratio(persistence + 0.01, series, start, stop)
    far = _lag1_copy_ratio(persistence + 5.0, series, start, stop)
    assert near < far


def test_copy_ratio_is_undefined_for_a_flat_series() -> None:
    flat = np.full(500, 50.0)
    assert np.isnan(_lag1_copy_ratio(np.full(100, 50.0), flat, 200, 300))


# --------------------------------------------------------------------------
# What the final fit is allowed to see
# --------------------------------------------------------------------------


def test_fit_values_stop_before_the_test_split(config) -> None:
    """The boundary the whole evaluation depends on."""
    from src.evaluate import load_area_series
    from src.splits import make_splits

    path = config.paths.processed / "selected_series.parquet"
    if not path.exists():
        pytest.skip("selected_series.parquet not present")

    values, times = load_area_series(config, 4159)
    splits = make_splits(values, times, config)

    assert splits.fit_values.size == splits.test.start
    assert splits.fit_values.size == len(splits.train) + len(splits.validation)
    np.testing.assert_array_equal(
        splits.fit_values[-len(splits.validation) :], splits.validation.values
    )


def test_final_fit_disables_early_stopping(config) -> None:
    """With no held-out set left, stopping must come from the tuned count.

    Leaving early stopping enabled would have nothing valid to stop on, and
    pointing it at the test week would be leakage.
    """
    from src.final_runs import _fit_final
    from src.splits import make_splits

    path = config.paths.processed / "selected_series.parquet"
    if not path.exists():
        pytest.skip("selected_series.parquet not present")

    from src.evaluate import load_area_series

    values, times = load_area_series(config, 4159)
    splits = make_splits(values, times, config)

    model = build_model(
        "lightgbm", {"num_leaves": 15, "n_estimators": 2000, "best_iteration": 40}, config, seed=0
    )
    _fit_final(model, splits, "lightgbm", {"best_iteration": 40})
    assert model.config.n_estimators == 40
    assert model.config.early_stopping_rounds == 0


def test_final_lstm_uses_the_tuned_epoch_count(config) -> None:
    from src.final_runs import _fit_final
    from src.splits import make_splits

    path = config.paths.processed / "selected_series.parquet"
    if not path.exists():
        pytest.skip("selected_series.parquet not present")

    from src.evaluate import load_area_series

    values, times = load_area_series(config, 4159)
    splits = make_splits(values, times, config)

    model = build_model(
        "lstm",
        {"hidden_size": 8, "num_layers": 1, "sequence_length": 36, "use_calendar": False},
        config,
        seed=0,
    )
    # best_epoch 2 means three epochs were run before validation stopped improving.
    _fit_final(model, splits, "lstm", {"best_epoch": 2})
    assert model.config.max_epochs == 3
    assert model.config.patience > model.config.max_epochs


def test_timing_row_records_the_device() -> None:
    """A training time without its device is not a comparison, it is a trap.

    The LSTM is tuned and fitted on a GPU while the other two run on CPU, so
    the timing table must say which produced each number.
    """
    result = FinalResult(square_id=5161, model="lstm", split="test", device="cuda")
    result.metrics = [_metrics(100.0, 0.3)]
    result.predictions = [np.zeros(1008)]
    result.inference_wall_s = [0.5]
    result.train_wall_s = [42.0]

    assert result.to_row()["device"] == "cuda"


def test_device_defaults_to_cpu_for_models_that_never_report_one() -> None:
    result = FinalResult(square_id=5161, model="persistence", split="test")
    result.metrics = [_metrics(100.0, 0.3)]
    result.predictions = [np.zeros(1008)]
    result.inference_wall_s = [0.001]

    assert result.to_row()["device"] == "cpu"


# --------------------------------------------------------------------------
# The seed ensemble
# --------------------------------------------------------------------------


def _member(predictions: list[np.ndarray], **kwargs) -> FinalResult:
    result = FinalResult(square_id=5161, model="lstm", split="test", **kwargs)
    result.predictions = predictions
    result.metrics = [_metrics(100.0 + i, 0.3) for i in range(len(predictions))]
    result.train_wall_s = [10.0] * len(predictions)
    result.inference_wall_s = [2.0] * len(predictions)
    result.n_params = 1000
    return result


def test_ensemble_row_is_named_for_its_members() -> None:
    from src.final_runs import ENSEMBLE_SUFFIX

    assert ENSEMBLE_SUFFIX == "_ensemble"


def test_ensemble_cost_is_summed_not_averaged() -> None:
    """An ensemble forecast needs every member fitted and every member run."""
    from src.config import load_config
    from src.evaluate import load_area_series
    from src.final_runs import build_ensemble
    from src.splits import make_splits

    config = load_config(auto_env=False)
    if not (config.paths.processed / "selected_series.parquet").exists():
        pytest.skip("selected_series.parquet not present")

    values, times = load_area_series(config, 5161)
    splits = make_splits(values, times, config)
    split = splits["test"]

    rng = np.random.default_rng(0)
    members = [split.values + rng.normal(0, 5, split.values.size) for _ in range(3)]
    member = _member(members)

    ensemble = build_ensemble(member, member.mean_predictions, splits, split, config)
    row = ensemble.to_row()

    assert ensemble.model == "lstm_ensemble"
    assert row["n_seeds"] == 3
    assert row["train_wall_s"] == pytest.approx(30.0)  # 3 x 10, not the mean
    assert row["inference_wall_s"] == pytest.approx(6.0)  # 3 x 2
    assert row["n_params"] == 3000  # three models are kept


def test_averaging_predictions_cannot_increase_error() -> None:
    """The property that makes the ensemble row necessary.

    The member row averages three errors; the ensemble row is the error of the
    averaged forecast. Jensen's inequality makes the second no larger, so the
    saved prediction series always looks at least as good as the model row
    beside it. That gap is why both rows must exist.
    """
    from src.config import load_config
    from src.evaluate import load_area_series
    from src.final_runs import build_ensemble
    from src.splits import make_splits

    config = load_config(auto_env=False)
    if not (config.paths.processed / "selected_series.parquet").exists():
        pytest.skip("selected_series.parquet not present")

    values, times = load_area_series(config, 5161)
    splits = make_splits(values, times, config)
    split = splits["test"]

    rng = np.random.default_rng(1)
    members = [split.values + rng.normal(0, 20, split.values.size) for _ in range(3)]
    mean_of_member_errors = float(np.mean([np.mean(np.abs(m - split.values)) for m in members]))

    member = _member(members)
    ensemble = build_ensemble(member, member.mean_predictions, splits, split, config)

    assert ensemble.metrics[0].mae <= mean_of_member_errors + 1e-9


def test_committed_tables_carry_an_ensemble_row_matching_the_predictions() -> None:
    """The figure and the table must describe the same series."""
    import csv as _csv

    import polars as pl

    from src.config import PROJECT_ROOT

    tables = PROJECT_ROOT / "results" / "tables"
    predictions = PROJECT_ROOT / "results" / "predictions"
    path = tables / "final_metrics_area_5161_test.csv"
    parquet = predictions / "test_area_5161.parquet"
    if not path.exists() or not parquet.exists():
        pytest.skip("final tables or predictions not present")

    with path.open(encoding="utf-8", newline="") as fh:
        rows = {r["model"]: r for r in _csv.DictReader(fh)}

    # Asserted, not skipped. Skipping when the table looked incomplete is what
    # let a rebuild that erased five of the six rows reach a commit looking
    # green: the file was there, the row simply was not.
    assert {"persistence", "seasonal_naive", "harmonic_arima", "lightgbm", "lstm"} <= set(rows), (
        f"per-area table is missing models; it holds only {sorted(rows)}. "
        "Regenerate with `python -m src.final_runs --rebuild-ensemble`."
    )
    assert int(rows["lstm"]["n_seeds"]) >= 2

    assert "lstm_ensemble" in rows, (
        "predictions hold the seed-averaged LSTM series but no lstm_ensemble row "
        "describes it. Run `python -m src.final_runs --rebuild-ensemble`."
    )

    frame = pl.read_parquet(parquet)
    recomputed = float((frame["observed"] - frame["lstm"]).abs().mean())
    assert recomputed == pytest.approx(float(rows["lstm_ensemble"]["mae"]), abs=0.01)


def test_rebuilding_preserves_every_row_in_the_per_area_table(tmp_path) -> None:
    """Regression: rows from CSV and from to_row() must group together.

    to_row() gives an int square_id and csv.DictReader gives the string. Keyed
    as they arrive the two form separate groups that format to the *same*
    per-area filename, so the group written second replaced the first -- and a
    rebuild that appended one ensemble row erased the five model rows beside
    it while leaving the combined table intact.
    """
    import csv as _csv
    import dataclasses

    from src.config import load_config
    from src.final_runs import _write_rows

    config = load_config(auto_env=False)
    staged = tmp_path / "results"
    (staged / "tables").mkdir(parents=True)
    config = dataclasses.replace(config, paths=dataclasses.replace(config.paths, results=staged))

    def _row(square_id, model, mase):
        return {
            "square_id": square_id,
            "model": model,
            "split": "test",
            "n_seeds": 1,
            "mae": 1.0,
            "mae_std": 0.0,
            "rmse": 1.0,
            "rmse_std": 0.0,
            "mape": 1.0,
            "smape": 1.0,
            "mase": mase,
            "mase_std": 0.0,
            "r2": 0.9,
            "bias": 0.0,
            "n_params": 0,
            "device": "cpu",
            "lag1_copy_ratio": 0.5,
            "train_wall_s": 0.0,
            "train_wall_std": 0.0,
            "inference_wall_s": 0.0,
            "inference_ms_per_step": 0.0,
        }

    # Strings as a CSV read would give them, ints as to_row() gives them.
    rows = [
        _row("5161", "persistence", "0.27"),
        _row("5161", "harmonic_arima", "0.24"),
        _row(5161, "lstm_ensemble", 0.23),
    ]
    _write_rows(rows, config, "test")

    with (staged / "tables" / "final_metrics_area_5161_test.csv").open(
        encoding="utf-8", newline=""
    ) as fh:
        written = [r["model"] for r in _csv.DictReader(fh)]

    assert set(written) == {"persistence", "harmonic_arima", "lstm_ensemble"}
    assert written[0] == "lstm_ensemble"  # sorted by MASE, best first
