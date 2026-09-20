"""Tests for the evaluation harness.

The harness decides what every reported number means, so the properties checked
here are structural: baselines must appear in every table whether or not a model
was supplied, MASE must be scaled by training-set error rather than by the
window being scored, and predictions must be produced by walk-forward inference
rather than any shortcut.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from src.config import load_config
from src.evaluate import AreaResult, evaluate_all, evaluate_area, load_area_series
from src.metrics import evaluate as score
from src.models.baselines import Persistence
from src.splits import make_splits

DAILY = 144


@pytest.fixture(scope="module")
def config():
    return load_config(auto_env=False)


@pytest.fixture(scope="module")
def splits(config):
    path = config.paths.processed / "selected_series.parquet"
    if not path.exists():
        pytest.skip("selected_series.parquet not present; run python run.py eda")
    values, times = load_area_series(config, 4159)
    return make_splits(values, times, config)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_load_area_series_returns_aligned_arrays(config, splits) -> None:
    values, times = load_area_series(config, 4159)
    assert values.shape == times.shape
    assert values.size == 8928
    assert np.isfinite(values).all()


def test_unknown_square_reports_what_is_available(config, splits) -> None:
    with pytest.raises(KeyError, match="available"):
        load_area_series(config, 99999)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def test_baselines_appear_even_with_no_models(config, splits) -> None:
    """A results table without its floor is not interpretable."""
    result = evaluate_area(4159, [], splits, config)
    names = {row["model"] for row in result.rows}
    assert names == {"persistence", "seasonal_naive"}


def test_supplied_models_are_added_to_the_baselines(config, splits) -> None:
    class Constant(Persistence):
        name = "constant"

        def predict_one_step(self, history, **kwargs):
            return 100.0

        def walk_forward(self, series, start, stop, *, times=None):
            return np.full(stop - start, 100.0)

    result = evaluate_area(4159, [Constant()], splits, config)
    names = {row["model"] for row in result.rows}
    assert names == {"persistence", "seasonal_naive", "constant"}


def test_evaluation_covers_exactly_the_split(config, splits) -> None:
    result = evaluate_area(4159, [], splits, config)
    for row in result.rows:
        assert row["n"] == len(splits.test) == 1008


def test_mase_is_scaled_by_training_error_not_the_test_window(config, splits) -> None:
    """Scaling by the window being scored would let a hard week flatter a model."""
    result = evaluate_area(4159, [], splits, config)
    persistence = next(r for r in result.rows if r["model"] == "persistence")

    predictions = Persistence().walk_forward(splits.series, splits.test.start, splits.test.stop)
    expected = score(
        splits.test.values,
        predictions,
        y_train=splits.train.values,
        seasonal_period=DAILY,
    )
    assert persistence["mase"] == pytest.approx(expected.mase)

    # Scaling against the test window instead would give a different number.
    against_test = score(
        splits.test.values, predictions, y_train=splits.test.values, seasonal_period=DAILY
    )
    assert against_test.mase != pytest.approx(expected.mase)


def test_rows_carry_timing_and_parameter_counts(config, splits) -> None:
    result = evaluate_area(4159, [], splits, config)
    for row in result.rows:
        assert row["inference_wall_s"] >= 0
        assert row["inference_ms_per_step"] >= 0
        assert row["n_params"] == 0  # baselines fit nothing


def test_each_split_can_be_evaluated(config, splits) -> None:
    for name in ("validation", "test", "stress"):
        result = evaluate_area(4159, [], splits, config, split_name=name)
        assert result.split == name
        assert all(row["n"] == len(splits[name]) for row in result.rows)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_table_orders_by_mase() -> None:
    result = AreaResult(square_id=1, split="test")
    for name, mase in (("worse", 0.9), ("better", 0.2)):
        result.rows.append(
            {
                "model": name,
                "mae": 1.0,
                "rmse": 1.0,
                "mape": 1.0,
                "smape": 1.0,
                "mase": mase,
                "r2": 0.5,
            }
        )
    rendered = result.table()
    assert rendered.index("better") < rendered.index("worse")


def test_evaluate_all_writes_a_table_per_area(config, splits, tmp_path) -> None:
    """Writes are redirected: the suite must not rewrite committed artefacts.

    Run against the real config this rewrote the tracked metrics tables, so
    running the tests dirtied the working tree and a reported table could be
    silently replaced by one a test produced.
    """
    import dataclasses
    import shutil

    staged = tmp_path / "results"
    (staged / "tables").mkdir(parents=True)
    shutil.copy2(
        config.paths.tables / "selected_areas.json", staged / "tables" / "selected_areas.json"
    )
    config = dataclasses.replace(config, paths=dataclasses.replace(config.paths, results=staged))

    results = evaluate_all(config, split_name="test")
    assert len(results) == 3

    for result in results:
        path = config.paths.tables / f"metrics_area_{result.square_id}_test.csv"
        assert path.exists()
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert {r["model"] for r in rows} >= {"persistence", "seasonal_naive"}

    combined = config.paths.tables / "metrics_all_areas_test.csv"
    assert combined.exists()


# --------------------------------------------------------------------------
# The finding this phase establishes
# --------------------------------------------------------------------------


def test_persistence_is_a_demanding_baseline(config, splits) -> None:
    """With lag-1 autocorrelation of 0.987, repeating the last value is strong.

    This is the bar every model in Phase 5 has to clear, and the reason a
    respectable-looking error can still be worse than doing nothing.
    """
    result = evaluate_area(4159, [], splits, config)
    persistence = next(r for r in result.rows if r["model"] == "persistence")
    seasonal = next(r for r in result.rows if r["model"] == "seasonal_naive")

    assert persistence["r2"] > 0.95
    assert persistence["mase"] < 0.5
    assert persistence["mae"] < seasonal["mae"]
