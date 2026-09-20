"""Tests for matrix assembly and the missing-data policy.

Assembly is where a quiet error is most expensive: a day stacked in the wrong
order, an off-by-one in the timestamp grid or a silently filled gap would
corrupt every metric downstream while leaving the shape correct. These tests
therefore assert on values and ordering, not just dimensions.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from src.build_matrix import (
    MatrixError,
    MatrixReport,
    _runs,
    apply_missing_policy,
    assemble_matrix,
    load_matrix,
    save_matrix,
    to_local,
)
from src.config import load_config

N_SQUARES = 3
PER_DAY = 4  # 4 intervals/day -> 360-minute spacing
N_DAYS = 4  # train/validation/test/stress get one day each
START = dt.date(2013, 11, 1)
BASE_MS = 1_383_260_400_000  # 2013-10-31T23:00:00Z == 2013-11-01 00:00 Rome
STEP_MS = 360 * 60 * 1000


@pytest.fixture
def config(tmp_path):
    """A miniature 4-day study whose paths point into tmp_path.

    One day per split, so the split validator is satisfied and the test-week
    escalation path can be exercised against a real split boundary.
    """
    return load_config(
        auto_env=False,
        overrides={
            "paths": {
                "raw": str(tmp_path / "raw"),
                "interim": str(tmp_path / "interim"),
                "processed": str(tmp_path / "processed"),
                "results": str(tmp_path / "results"),
            },
            "dataset": {
                "start_date": "2013-11-01",
                "end_date": "2013-11-04",
                "n_squares": N_SQUARES,
                "daily_period": PER_DAY,
                "weekly_period": PER_DAY * 7,
                "interval_minutes": 360,
            },
            "splits": {
                "train": ["2013-11-01", "2013-11-01"],
                "validation": ["2013-11-02", "2013-11-02"],
                "test": ["2013-11-03", "2013-11-03"],
                "stress": ["2013-11-04", "2013-11-04"],
            },
        },
    )


def _write_blocks(config, *, absent_per_day=0, value_fn=None):
    """Write synthetic per-day blocks whose values encode their position."""
    interim = config.paths.interim
    interim.mkdir(parents=True, exist_ok=True)
    for day in range(N_DAYS):
        date = (START + dt.timedelta(days=day)).isoformat()
        values = np.zeros((PER_DAY, N_SQUARES), dtype=np.float32)
        for t in range(PER_DAY):
            for s in range(N_SQUARES):
                values[t, s] = value_fn(day, t, s) if value_fn else day * 1000 + t * 10 + s
        times = (
            np.array(
                [BASE_MS + (day * PER_DAY + t) * STEP_MS for t in range(PER_DAY)],
                dtype="int64",
            )
            .astype("datetime64[ms]")
            .astype("datetime64[ns]")
        )

        np.save(interim / f"{date}.npy", values)
        np.save(interim / f"{date}.times.npy", times)
        (interim / f"{date}.json").write_text(
            json.dumps({"date": date, "n_absent_cells": absent_per_day}), encoding="utf-8"
        )


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def test_assembles_days_in_chronological_order(config) -> None:
    """Row r of the matrix must be the r-th interval of the whole period."""
    _write_blocks(config)
    matrix, utc, squares, report = assemble_matrix(config)

    assert matrix.shape == (N_DAYS * PER_DAY, N_SQUARES)
    assert report.shape == (N_DAYS * PER_DAY, N_SQUARES)
    for day in range(N_DAYS):
        for t in range(PER_DAY):
            assert matrix[day * PER_DAY + t, 0] == pytest.approx(day * 1000 + t * 10)
    assert (np.diff(utc).astype("int64") > 0).all()
    assert squares.tolist() == [1, 2, 3]


def test_absent_cell_counts_are_carried_through(config) -> None:
    _write_blocks(config, absent_per_day=7)
    _, _, _, report = assemble_matrix(config)
    assert report.total_absent_cells == 7 * N_DAYS


def test_missing_day_is_rejected(config) -> None:
    """A gap in the ingest must stop assembly, not produce a short matrix."""
    _write_blocks(config)
    for suffix in (".npy", ".times.npy", ".json"):
        (config.paths.interim / f"2013-11-02{suffix}").unlink()

    with pytest.raises(MatrixError, match="not ingested"):
        assemble_matrix(config)


def test_day_with_wrong_interval_count_is_rejected(config) -> None:
    """A short day must fail loudly rather than shift every later row."""
    _write_blocks(config)
    date = "2013-11-02"
    np.save(config.paths.interim / f"{date}.npy", np.zeros((PER_DAY - 1, N_SQUARES), np.float32))
    times = np.load(config.paths.interim / f"{date}.times.npy")[: PER_DAY - 1]
    np.save(config.paths.interim / f"{date}.times.npy", times)

    with pytest.raises(MatrixError):
        assemble_matrix(config)


def test_wrong_square_count_is_rejected(config) -> None:
    _write_blocks(config)
    np.save(
        config.paths.interim / "2013-11-02.npy",
        np.zeros((PER_DAY, N_SQUARES + 1), np.float32),
    )
    with pytest.raises(MatrixError, match="squares"):
        assemble_matrix(config)


def test_total_internet_matches_the_sum(config) -> None:
    _write_blocks(config)
    matrix, _, _, report = assemble_matrix(config)
    assert report.total_internet == pytest.approx(float(matrix.sum(dtype=np.float64)))


# --------------------------------------------------------------------------
# Timezone
# --------------------------------------------------------------------------


def test_local_times_start_at_local_midnight(config) -> None:
    """The period opens at 00:00 Europe/Rome, which is 23:00 UTC the day before."""
    _write_blocks(config)
    _, _, _, report = assemble_matrix(config)

    assert report.first_timestamp_utc.startswith("2013-10-31T23:00")
    assert report.first_timestamp_local.startswith("2013-11-01T00:00")
    assert report.timezone == "Europe/Rome"


def test_to_local_applies_the_cet_offset() -> None:
    utc = np.array(["2013-11-01T12:00:00"], dtype="datetime64[ns]")
    local = to_local(utc, "Europe/Rome")
    assert str(local[0]).startswith("2013-11-01T13:00")  # CET = UTC+1


def test_every_local_day_has_the_expected_interval_count(config) -> None:
    """Guards against a DST transition or a malformed day slipping through."""
    _write_blocks(config)
    _, _, _, report = assemble_matrix(config)
    assert report.days_with_wrong_interval_count == []


# --------------------------------------------------------------------------
# Missing-interval policy
# --------------------------------------------------------------------------


def test_runs_groups_consecutive_indices() -> None:
    assert _runs(np.array([])) == []
    assert _runs(np.array([3])) == [(3, 1)]
    assert _runs(np.array([1, 2, 3, 7, 8, 20])) == [(1, 3), (7, 2), (20, 1)]


def _blank_report() -> MatrixReport:
    return MatrixReport(
        shape=(0, 0),
        n_days=0,
        first_timestamp_utc="",
        last_timestamp_utc="",
        first_timestamp_local="",
        last_timestamp_local="",
        timezone="Europe/Rome",
        total_absent_cells=0,
    )


def test_short_gap_is_linearly_interpolated(config) -> None:
    matrix = np.arange(40, dtype=np.float32).reshape(10, 4) * 10.0
    matrix[4] = np.nan
    times = np.array(
        [np.datetime64("2013-11-01T00:00") + np.timedelta64(6 * i, "h") for i in range(10)],
        dtype="datetime64[ns]",
    )
    report = _blank_report()

    out = apply_missing_policy(matrix, times, config, report)

    expected = (np.arange(12, 16) * 10.0 + np.arange(20, 24) * 10.0) / 2
    np.testing.assert_allclose(out[4], expected, rtol=1e-5)
    assert len(report.interpolated_intervals) == 1
    assert report.unfilled_intervals == []
    assert report.n_nan_after_policy == 0


def test_long_gap_is_left_as_nan_and_reported(config) -> None:
    """Beyond max_interpolate_gap, inventing data would be worse than a gap."""
    matrix = np.ones((12, 4), dtype=np.float32)
    matrix[3:9] = np.nan  # 6 intervals, over the limit of 3
    # Hourly spacing keeps the whole window inside 2013-11-01, so this exercises
    # the "leave as NaN" branch rather than the test-week escalation.
    times = np.array(
        [np.datetime64("2013-11-01T00:00") + np.timedelta64(i, "h") for i in range(12)],
        dtype="datetime64[ns]",
    )
    report = _blank_report()

    out = apply_missing_policy(matrix, times, config, report)

    assert np.isnan(out[3:9]).all()
    assert len(report.unfilled_intervals) == 6
    assert report.interpolated_intervals == []
    assert report.n_nan_after_policy == 24


def test_unfilled_gap_inside_the_test_week_escalates(config) -> None:
    """A hole in the evaluation window must stop the pipeline, not be imputed."""
    matrix = np.ones((12, 4), dtype=np.float32)
    matrix[4:10] = np.nan
    # config's test split is 2013-11-03; place the gap there.
    times = np.array(
        [np.datetime64("2013-11-03T00:00") + np.timedelta64(i, "h") for i in range(12)],
        dtype="datetime64[ns]",
    )
    report = _blank_report()

    with pytest.raises(MatrixError, match="inside the test week"):
        apply_missing_policy(matrix, times, config, report)


def test_no_gaps_is_a_no_op(config) -> None:
    matrix = np.ones((6, 4), dtype=np.float32)
    times = np.array(
        [np.datetime64("2013-11-01T00:00") + np.timedelta64(i, "h") for i in range(6)],
        dtype="datetime64[ns]",
    )
    report = _blank_report()

    out = apply_missing_policy(matrix, times, config, report)
    assert out is matrix
    assert report.interpolated_intervals == []
    assert report.n_nan_after_policy == 0


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_save_and_load_round_trip(config) -> None:
    _write_blocks(config)
    matrix, utc, squares, report = assemble_matrix(config)
    paths = save_matrix(matrix, utc, squares, report, config)

    for path in paths.values():
        assert path.exists(), path

    loaded, timestamps, loaded_squares = load_matrix(config)
    np.testing.assert_array_equal(np.asarray(loaded), matrix)
    np.testing.assert_array_equal(loaded_squares, squares)
    assert timestamps.height == matrix.shape[0]
    assert {"index", "utc", "local"} <= set(timestamps.columns)


def test_totals_are_per_square_column_sums(config) -> None:
    _write_blocks(config)
    matrix, utc, squares, report = assemble_matrix(config)
    paths = save_matrix(matrix, utc, squares, report, config)

    import polars as pl

    totals = pl.read_parquet(paths["totals"])
    assert totals.height == N_SQUARES
    np.testing.assert_allclose(
        totals["total_internet"].to_numpy(), matrix.sum(axis=0, dtype=np.float64), rtol=1e-5
    )
    np.testing.assert_allclose(
        totals["mean_internet"].to_numpy(),
        matrix.sum(axis=0, dtype=np.float64) / matrix.shape[0],
        rtol=1e-5,
    )


def test_report_json_is_written(config) -> None:
    _write_blocks(config)
    matrix, utc, squares, report = assemble_matrix(config)
    paths = save_matrix(matrix, utc, squares, report, config)

    payload = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert payload["shape"] == [N_DAYS * PER_DAY, N_SQUARES]
    assert payload["timezone"] == "Europe/Rome"
    assert "total_absent_cells" in payload
