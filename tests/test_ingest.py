"""Tests for the raw-to-block ingest.

These run on a synthetic file small enough to be exact, so the assertions can
check values rather than shapes. The properties worth protecting are the ones
that would silently corrupt every downstream result:

* country-split rows must be summed, not dropped or double counted
* a cell with no row at all must become 0.0, not NaN and not be skipped
* the two independent implementations must agree
* a day that is not exactly 144 intervals must fail loudly rather than be padded
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from src.config import load_config
from src.ingest import (
    COLUMN_NAMES,
    DayBlock,
    DayStats,
    IngestError,
    date_from_path,
    day_is_complete,
    ingest_day_pandas_chunked,
    ingest_day_polars,
    raw_day_files,
    write_day,
)

# A miniature grid keeps the synthetic fixtures exact and fast.
N_SQUARES = 4
N_INTERVALS = 6
BASE_MS = 1_383_260_400_000  # 2013-10-31T23:00:00Z == 2013-11-01 00:00 Rome


@pytest.fixture
def config():
    """Config shrunk to the miniature grid used by the synthetic fixtures."""
    return load_config(
        auto_env=False,
        overrides={
            "dataset": {
                "n_squares": N_SQUARES,
                "daily_period": N_INTERVALS,
                "weekly_period": N_INTERVALS * 7,
                "interval_minutes": 240,  # 6 intervals/day
            }
        },
    )


def _write_raw(path, rows: list[tuple]) -> None:
    """Write rows in the raw tab-separated layout, blanks for missing values."""
    lines = []
    for row in rows:
        lines.append("\t".join("" if v is None else str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_day(tmp_path, *, name="sms-call-internet-mi-2013-11-01.txt", omit=()):
    """Build a synthetic day where every cell's value equals square*100 + interval.

    Each cell is split across two country rows so the test exercises the
    aggregation, and ``omit`` drops cells entirely to simulate no recorded
    activity.
    """
    rows = []
    expected = np.zeros((N_INTERVALS, N_SQUARES), dtype=np.float32)
    for t in range(N_INTERVALS):
        for s in range(1, N_SQUARES + 1):
            if (s, t) in omit:
                continue
            value = s * 100 + t
            expected[t, s - 1] = value
            time_ms = BASE_MS + t * 240 * 60 * 1000
            # Split across two countries, plus a row whose internet is blank.
            rows.append((s, time_ms, 39, 0.1, 0.1, 0.1, 0.1, value * 0.25))
            rows.append((s, time_ms, 33, 0.1, 0.1, 0.1, 0.1, value * 0.75))
            rows.append((s, time_ms, 0, 0.1, 0.1, 0.1, 0.1, None))
    path = tmp_path / name
    _write_raw(path, rows)
    return path, expected


# --------------------------------------------------------------------------
# Filename parsing
# --------------------------------------------------------------------------


def test_date_from_path() -> None:
    from pathlib import Path

    assert date_from_path(Path("sms-call-internet-mi-2013-11-01.txt")) == "2013-11-01"
    assert date_from_path(Path("/a/b/sms-call-internet-mi-2014-01-01.txt")) == "2014-01-01"


def test_date_from_path_rejects_unparseable_name() -> None:
    from pathlib import Path

    with pytest.raises(IngestError, match="cannot parse a date"):
        date_from_path(Path("not-a-day-file.txt"))


# --------------------------------------------------------------------------
# Aggregation correctness
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ingest", [ingest_day_polars, ingest_day_pandas_chunked])
def test_country_split_rows_are_summed(tmp_path, config, ingest) -> None:
    """Each cell is split over three country rows; the block must hold the sum."""
    path, expected = _make_day(tmp_path)
    block = ingest(path, config)

    assert block.values.shape == (N_INTERVALS, N_SQUARES)
    assert block.values.dtype == np.float32
    np.testing.assert_allclose(block.values, expected, rtol=1e-5)


@pytest.mark.parametrize("ingest", [ingest_day_polars, ingest_day_pandas_chunked])
def test_raw_row_count_is_reported(tmp_path, config, ingest) -> None:
    path, _ = _make_day(tmp_path)
    block = ingest(path, config)
    assert block.stats.n_raw_rows == N_INTERVALS * N_SQUARES * 3


@pytest.mark.parametrize("ingest", [ingest_day_polars, ingest_day_pandas_chunked])
def test_absent_cells_become_zero_and_are_counted(tmp_path, config, ingest) -> None:
    """An absent (square, time) pair means no activity, so 0.0 -- but it is reported."""
    omit = {(2, 1), (3, 4)}
    path, expected = _make_day(tmp_path, omit=omit)
    block = ingest(path, config)

    assert block.stats.n_absent_cells == len(omit)
    for square, interval in omit:
        assert block.values[interval, square - 1] == 0.0
    assert not np.isnan(block.values).any()
    np.testing.assert_allclose(block.values, expected, rtol=1e-5)


@pytest.mark.parametrize("ingest", [ingest_day_polars, ingest_day_pandas_chunked])
def test_null_internet_rows_do_not_become_nan(tmp_path, config, ingest) -> None:
    """Every cell here has one blank-internet row; blanks must read as 0.0."""
    path, expected = _make_day(tmp_path)
    block = ingest(path, config)
    assert np.isfinite(block.values).all()
    assert block.values.sum() == pytest.approx(expected.sum(), rel=1e-5)


def test_both_strategies_agree(tmp_path, config) -> None:
    """Two independent implementations are only useful if they cross-check."""
    path, _ = _make_day(tmp_path, omit={(1, 0)})
    a = ingest_day_polars(path, config)
    b = ingest_day_pandas_chunked(path, config)

    np.testing.assert_allclose(a.values, b.values, rtol=1e-5)
    assert np.array_equal(a.timestamps, b.timestamps)
    assert a.stats.n_absent_cells == b.stats.n_absent_cells
    assert a.stats.n_raw_rows == b.stats.n_raw_rows


# --------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ingest", [ingest_day_polars, ingest_day_pandas_chunked])
def test_timestamps_are_utc_and_ordered(tmp_path, config, ingest) -> None:
    """Blocks store UTC; the conversion to Europe/Rome happens once, later."""
    path, _ = _make_day(tmp_path)
    block = ingest(path, config)

    assert block.timestamps.dtype == np.dtype("datetime64[ns]")
    assert (np.diff(block.timestamps).astype("int64") > 0).all()
    assert block.timestamps[0] == np.datetime64("2013-10-31T23:00:00.000000000")


def test_local_day_alignment_is_preserved(tmp_path, config) -> None:
    """The first record of a daily file is local midnight, not UTC midnight.

    2013-10-31T23:00Z is 2013-11-01 00:00 in Europe/Rome (CET, UTC+1). Losing
    this would shift every diurnal pattern by an hour.
    """
    import zoneinfo

    path, _ = _make_day(tmp_path)
    block = ingest_day_polars(path, config)

    first = block.timestamps[0].astype("datetime64[s]").astype(dt.datetime)
    local = first.replace(tzinfo=dt.UTC).astimezone(zoneinfo.ZoneInfo("Europe/Rome"))
    assert (local.hour, local.minute) == (0, 0)
    assert local.date() == dt.date(2013, 11, 1)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_short_day_is_rejected(tmp_path, config) -> None:
    """A day with the wrong number of intervals must fail, never be padded."""
    rows = [(1, BASE_MS, 39, 0.1, 0.1, 0.1, 0.1, 5.0)]
    path = tmp_path / "sms-call-internet-mi-2013-11-01.txt"
    _write_raw(path, rows)

    with pytest.raises(IngestError, match="expected 6 intervals"):
        ingest_day_polars(path, config)


def test_square_id_out_of_range_is_rejected(tmp_path, config) -> None:
    rows = []
    for t in range(N_INTERVALS):
        time_ms = BASE_MS + t * 240 * 60 * 1000
        rows.append((99, time_ms, 39, 0.1, 0.1, 0.1, 0.1, 1.0))
    path = tmp_path / "sms-call-internet-mi-2013-11-01.txt"
    _write_raw(path, rows)

    with pytest.raises(IngestError, match="outside 1"):
        ingest_day_polars(path, config)


def test_block_rejects_wrong_dtype() -> None:
    stats = DayStats(
        date="2013-11-01",
        source="x",
        n_raw_rows=0,
        n_pairs=0,
        n_absent_cells=0,
        n_timestamps=1,
        n_squares_seen=1,
        total_internet=0.0,
        strategy="test",
    )
    with pytest.raises(IngestError, match="float32"):
        DayBlock(
            values=np.zeros((1, 1), dtype=np.float64),
            timestamps=np.array(["2013-11-01"], dtype="datetime64[ns]"),
            stats=stats,
        )


# --------------------------------------------------------------------------
# Persistence and checkpointing
# --------------------------------------------------------------------------


def test_write_day_round_trip(tmp_path, config) -> None:
    path, expected = _make_day(tmp_path)
    block = ingest_day_polars(path, config)
    interim = tmp_path / "interim"

    written = write_day(block, interim)
    assert written.exists()
    np.testing.assert_allclose(np.load(written), expected, rtol=1e-5)

    stats = json.loads((interim / "2013-11-01.json").read_text(encoding="utf-8"))
    assert stats["date"] == "2013-11-01"
    assert stats["n_raw_rows"] == block.stats.n_raw_rows
    assert "absent_fraction" in stats


def test_day_is_complete_requires_every_artifact(tmp_path, config) -> None:
    """Checkpointing is what lets a timed-out Kaggle session resume."""
    path, _ = _make_day(tmp_path)
    interim = tmp_path / "interim"
    assert not day_is_complete("2013-11-01", interim)

    write_day(ingest_day_polars(path, config), interim)
    assert day_is_complete("2013-11-01", interim)

    (interim / "2013-11-01.json").unlink()
    assert not day_is_complete("2013-11-01", interim)


def test_raw_day_files_are_sorted(tmp_path) -> None:
    for name in ("2013-12-01", "2013-11-02", "2013-11-01"):
        (tmp_path / f"sms-call-internet-mi-{name}.txt").write_text("", encoding="utf-8")
    (tmp_path / "unrelated.txt").write_text("", encoding="utf-8")

    found = [p.name for p in raw_day_files(tmp_path)]
    assert found == [
        "sms-call-internet-mi-2013-11-01.txt",
        "sms-call-internet-mi-2013-11-02.txt",
        "sms-call-internet-mi-2013-12-01.txt",
    ]


def test_column_names_match_the_documented_layout() -> None:
    assert COLUMN_NAMES[0] == "square_id"
    assert COLUMN_NAMES[1] == "time_ms"
    assert COLUMN_NAMES[7] == "internet"
    assert len(COLUMN_NAMES) == 8
