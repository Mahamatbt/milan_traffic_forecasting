"""Tests for the memory benchmark's bookkeeping.

The measurements themselves need real files and real processes, so what is
tested here is the reasoning applied to them: that float32 rounding is not
mistaken for a correctness bug, that a genuine disagreement is, and that the
naive strategy's projection scales with the dataset while the optimised ones
do not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.memory_report import (
    REPORT_COLUMNS,
    STRATEGY_NOTES,
    _cross_check,
    _final_artifact_bytes,
    _to_measurement,
)


@pytest.fixture(scope="module")
def config():
    return load_config(auto_env=False)


def _raw(strategy: str, **overrides):
    base = {
        "strategy": strategy,
        "peak_rss_delta": 500 * 1024**2,
        "baseline_rss": 50 * 1024**2,
        "inmem_bytes": 5_760_000,
        "wall_s": 1.25,
        "available_at_start": 4 * 1024**3,
        "total_internet": 82_479_246.57,
    }
    base.update(overrides)
    return base


@pytest.fixture
def day_file(tmp_path) -> Path:
    path = tmp_path / "sms-call-internet-mi-2013-11-01.txt"
    path.write_bytes(b"x" * 1024)
    return path


# --------------------------------------------------------------------------
# Cross-checking the two implementations
# --------------------------------------------------------------------------


def test_float32_rounding_is_not_treated_as_disagreement() -> None:
    """Summing 4.8M float32 values in different orders differs in the last digit."""
    _cross_check({"2013-11-01": {82_479_246.57, 82_479_246.58}})


def test_genuine_disagreement_is_rejected() -> None:
    """A real aggregation bug would move the total by far more than rounding."""
    with pytest.raises(RuntimeError, match="disagree"):
        _cross_check({"2013-11-01": {82_479_246.57, 82_479_999.00}})


def test_single_value_is_fine() -> None:
    _cross_check({"2013-11-01": {82_479_246.57}})
    _cross_check({})


def test_tolerance_is_relative_not_absolute() -> None:
    """A 0.01 spread is noise at 8e7 and a real error at 1.0."""
    _cross_check({"big": {82_479_246.57, 82_479_246.58}})
    with pytest.raises(RuntimeError):
        _cross_check({"small": {1.00, 1.01}})


# --------------------------------------------------------------------------
# Projection semantics
# --------------------------------------------------------------------------


def test_naive_projection_scales_with_the_dataset(config, day_file) -> None:
    """Holding every day the naive way needs 62x one day."""
    m = _to_measurement(_raw("naive_pandas", inmem_bytes=310_000_000), day_file, config)

    assert m.extrapolated is True
    assert m.projected_full_ram_bytes == 310_000_000 * config.dataset.n_days
    assert m.final_artifact_ram_bytes is None


def test_optimised_projection_is_flat_in_the_number_of_days(config, day_file) -> None:
    """Only one block is ever resident, so the working set does not grow."""
    m = _to_measurement(_raw("polars_lazy"), day_file, config)

    assert m.extrapolated is False
    assert m.projected_full_ram_bytes == 500 * 1024**2
    assert m.final_artifact_ram_bytes == _final_artifact_bytes(config)


def test_final_artifact_is_the_float32_matrix(config) -> None:
    assert _final_artifact_bytes(config) == 8928 * 10000 * 4


def test_measurement_row_matches_the_report_schema(config, day_file) -> None:
    row = _to_measurement(_raw("pandas_chunked"), day_file, config).to_row()
    assert set(row) == set(REPORT_COLUMNS)
    assert row["day"] == "2013-11-01"
    assert row["source_file_bytes"] == 1024


@pytest.mark.parametrize("strategy", ["naive_pandas", "pandas_chunked", "polars_lazy"])
def test_every_strategy_has_a_note(strategy: str) -> None:
    """The CSV is read by a human; an unexplained row is not evidence."""
    assert STRATEGY_NOTES[strategy].strip()
