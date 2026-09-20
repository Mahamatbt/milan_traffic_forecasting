"""Tests for the exploratory statistics.

These numbers go straight into the report, so each is checked against a case
where the answer is known analytically rather than against a previous run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.config import load_config
from src.eda import (
    SelectedAreas,
    area_summary,
    distribution_stats,
    extract_series,
    gini,
    grid_position,
    select_areas,
    totals_as_grid,
)


@pytest.fixture(scope="module")
def config():
    return load_config(auto_env=False)


# --------------------------------------------------------------------------
# Gini
# --------------------------------------------------------------------------


def test_gini_of_perfect_equality_is_zero() -> None:
    assert gini(np.ones(1000)) == pytest.approx(0.0, abs=1e-12)


def test_gini_approaches_one_when_one_cell_holds_everything() -> None:
    values = np.zeros(1000)
    values[-1] = 1.0
    assert gini(values) == pytest.approx(1 - 1 / 1000, abs=1e-9)


def test_gini_is_scale_invariant() -> None:
    """Doubling every cell changes no inequality, so the coefficient must hold."""
    rng = np.random.default_rng(0)
    values = rng.lognormal(size=500)
    assert gini(values) == pytest.approx(gini(values * 1000), rel=1e-12)


def test_gini_of_all_zeros_is_zero_not_nan() -> None:
    assert gini(np.zeros(10)) == 0.0


def test_gini_of_empty_is_zero() -> None:
    assert gini(np.array([])) == 0.0


def test_gini_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="negative"):
        gini(np.array([-1.0, 2.0]))


# --------------------------------------------------------------------------
# Distribution statistics
# --------------------------------------------------------------------------


def test_distribution_recovers_known_lognormal_parameters() -> None:
    """A fit that cannot recover parameters it generated is not evidence."""
    rng = np.random.default_rng(42)
    values = rng.lognormal(mean=8.0, sigma=1.5, size=20000)

    stats = distribution_stats(values)
    assert stats.lognormal_mu == pytest.approx(8.0, abs=0.05)
    assert stats.lognormal_sigma == pytest.approx(1.5, abs=0.05)
    assert stats.lognormal_ks_pvalue > 0.01


def test_tail_shares_are_ordered_and_bounded() -> None:
    rng = np.random.default_rng(1)
    stats = distribution_stats(rng.lognormal(mean=6, sigma=2, size=10000))

    assert 0 < stats.top_1pct_share <= stats.top_5pct_share <= stats.top_10pct_share <= 1
    assert 0 <= stats.bottom_50pct_share <= 0.5


@pytest.mark.filterwarnings("ignore:Precision loss occurred:RuntimeWarning")
def test_uniform_distribution_has_shares_matching_its_fractions() -> None:
    """With every cell equal, the top 10% must hold exactly 10% of the traffic.

    scipy warns about catastrophic cancellation computing moments of identical
    values; that is expected for this deliberately degenerate case.
    """
    stats = distribution_stats(np.full(1000, 5.0))
    assert stats.top_1pct_share == pytest.approx(0.01, abs=1e-9)
    assert stats.top_10pct_share == pytest.approx(0.10, abs=1e-9)
    assert stats.gini == pytest.approx(0.0, abs=1e-12)
    assert stats.max_over_median == pytest.approx(1.0)


def test_zero_cells_are_counted_and_excluded_from_the_fit() -> None:
    values = np.concatenate([np.zeros(50), np.full(950, 10.0)])
    stats = distribution_stats(values)
    assert stats.n_zero_cells == 50
    assert stats.n_cells == 1000


def test_stats_serialise_to_json() -> None:
    stats = distribution_stats(np.random.default_rng(0).lognormal(size=500))
    json.dumps(stats.to_dict())
    assert len(stats.summary_lines()) >= 8


# --------------------------------------------------------------------------
# Grid geometry
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("square_id", "expected"),
    [(1, (0, 0)), (100, (0, 99)), (101, (1, 0)), (10000, (99, 99))],
)
def test_grid_position(square_id: int, expected: tuple[int, int]) -> None:
    assert grid_position(square_id) == expected


@pytest.mark.parametrize("square_id", [0, -1, 10001])
def test_grid_position_rejects_ids_outside_the_grid(square_id: int) -> None:
    with pytest.raises(ValueError, match="outside"):
        grid_position(square_id)


def test_totals_as_grid_is_row_major() -> None:
    """Reshaping must agree with grid_position or the map is transposed."""
    totals = np.arange(10000, dtype=np.float64)
    grid = totals_as_grid(totals)
    for square in (1, 100, 101, 5161, 10000):
        row, column = grid_position(square)
        assert grid[row, column] == totals[square - 1]


def test_totals_as_grid_rejects_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="expected"):
        totals_as_grid(np.zeros(99))


# --------------------------------------------------------------------------
# Area selection
# --------------------------------------------------------------------------


def test_selects_the_three_highest_and_keeps_the_brief_ids(config) -> None:
    totals = np.zeros(10000)
    square_ids = np.arange(1, 10001)
    totals[[6, 41, 99]] = [300.0, 200.0, 100.0]  # squares 7, 42, 100
    totals[4158] = 50.0  # square 4159
    totals[4555] = 75.0  # square 4556

    areas = select_areas(totals, square_ids, config)

    assert areas.top_three == [7, 42, 100]
    assert areas.top_traffic == 7
    assert areas.fixed == [4159, 4556]
    assert areas.forecast == [7, 42, 100]


def test_ranks_and_totals_are_recorded(config) -> None:
    totals = np.zeros(10000)
    square_ids = np.arange(1, 10001)
    totals[[6, 41, 99]] = [300.0, 200.0, 100.0]
    totals[4158], totals[4555] = 50.0, 75.0

    areas = select_areas(totals, square_ids, config)
    assert areas.ranks["7"] == 1
    assert areas.ranks["4556"] == 4
    assert areas.ranks["4159"] == 5
    assert areas.totals["4556"] == 75.0


def test_all_plotted_is_five_areas_without_duplicates(config) -> None:
    totals = np.zeros(10000)
    square_ids = np.arange(1, 10001)
    totals[[6, 41, 99]] = [300.0, 200.0, 100.0]
    areas = select_areas(totals, square_ids, config)
    assert areas.all_plotted == [7, 42, 100, 4159, 4556]


def test_all_plotted_deduplicates_when_a_brief_id_is_also_top(config) -> None:
    """If square 4556 turns out to be the busiest, it must appear once."""
    totals = np.zeros(10000)
    square_ids = np.arange(1, 10001)
    totals[4555] = 500.0  # square 4556 is top
    totals[[6, 41]] = [300.0, 200.0]

    areas = select_areas(totals, square_ids, config)
    assert areas.top_three == [4556, 7, 42]
    assert areas.all_plotted == [4556, 7, 42, 4159]
    assert len(areas.all_plotted) == len(set(areas.all_plotted))


def test_selected_areas_round_trip(tmp_path, config) -> None:
    totals = np.zeros(10000)
    square_ids = np.arange(1, 10001)
    totals[[6, 41, 99]] = [300.0, 200.0, 100.0]
    areas = select_areas(totals, square_ids, config)

    path = areas.save(tmp_path / "selected_areas.json")
    assert SelectedAreas.load(path).forecast == areas.forecast


# --------------------------------------------------------------------------
# Series extraction
# --------------------------------------------------------------------------


def test_extract_series_returns_columns_in_the_requested_order() -> None:
    matrix = np.arange(50, dtype=np.float32).reshape(10, 5)
    square_ids = np.array([11, 22, 33, 44, 55])

    extracted = extract_series(matrix, square_ids, [33, 11])
    np.testing.assert_array_equal(extracted[:, 0], matrix[:, 2])
    np.testing.assert_array_equal(extracted[:, 1], matrix[:, 0])


def test_extract_series_rejects_an_unknown_square() -> None:
    matrix = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(KeyError, match="not in the matrix"):
        extract_series(matrix, np.array([1, 2, 3]), [99])


def test_extract_series_materialises_from_a_memmap(tmp_path) -> None:
    """Callers memory-map the 341 MiB matrix; extraction must copy, not view."""
    path = tmp_path / "m.npy"
    np.save(path, np.arange(50, dtype=np.float32).reshape(10, 5))
    mapped = np.load(path, mmap_mode="r")

    extracted = extract_series(mapped, np.arange(1, 6), [2])
    assert isinstance(extracted, np.ndarray)
    assert extracted.dtype == np.float32
    assert extracted.shape == (10, 1)


# --------------------------------------------------------------------------
# Per-area summary
# --------------------------------------------------------------------------


def _times(n: int) -> np.ndarray:
    """Local timestamps at 10-minute spacing from a Monday midnight."""
    start = np.datetime64("2013-11-04T00:00")  # a Monday
    return start + np.arange(n) * np.timedelta64(10, "m")


def test_area_summary_computes_known_quantities() -> None:
    n = 144 * 7
    times = _times(n)
    flat = np.full((n, 1), 10.0)

    row = area_summary(flat, [1], times)[0]
    assert row["square_id"] == 1
    assert row["mean"] == pytest.approx(10.0)
    assert row["cv"] == pytest.approx(0.0, abs=1e-12)
    assert row["peak_to_trough"] == pytest.approx(1.0)
    assert row["weekend_over_weekday"] == pytest.approx(1.0)
    assert row["n_zero"] == 0


def test_area_summary_detects_a_weekend_difference() -> None:
    n = 144 * 7
    times = _times(n)
    weekday = np.array([t.astype("datetime64[D]").astype(object).weekday() for t in times])
    values = np.where(weekday >= 5, 5.0, 10.0).reshape(-1, 1)

    row = area_summary(values, [1], times)[0]
    assert row["weekday_mean"] == pytest.approx(10.0)
    assert row["weekend_mean"] == pytest.approx(5.0)
    assert row["weekend_over_weekday"] == pytest.approx(0.5)


def test_area_summary_handles_multiple_areas() -> None:
    n = 144 * 7
    values = np.column_stack([np.full(n, 10.0), np.full(n, 100.0)])
    rows = area_summary(values, [7, 42], _times(n))

    assert [r["square_id"] for r in rows] == [7, 42]
    assert rows[1]["mean"] == pytest.approx(100.0)
