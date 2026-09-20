"""Tests for locating grid cells on the ground.

These underpin a claim in the report about what is physically at each study
area, so the checks are against independently known facts: Milan's bounding
coordinates, the published 235 m cell size, and the requirement that the grid
index agrees with compass direction.
"""

from __future__ import annotations

import pytest

from src.config import load_config
from src.eda import grid_position
from src.geo import (
    LANDMARKS,
    _haversine_m,
    cell_centroid,
    describe_cell,
    grid_path,
    nearest_landmarks,
)

# Milan sits near 45.46 N, 9.19 E; the grid extends well beyond the centre.
MILAN_LAT = (45.30, 45.60)
MILAN_LON = (8.95, 9.35)


@pytest.fixture(scope="module")
def config():
    return load_config(auto_env=False)


@pytest.fixture(scope="module", autouse=True)
def _require_grid(config):
    if not grid_path(config).exists():
        pytest.skip("milano-grid.geojson not present")


# --------------------------------------------------------------------------
# Distance
# --------------------------------------------------------------------------


def test_haversine_zero_for_identical_points() -> None:
    assert _haversine_m(45.4642, 9.19, 45.4642, 9.19) == pytest.approx(0.0, abs=1e-9)


def test_haversine_matches_a_known_separation() -> None:
    """One degree of latitude is close to 111.2 km anywhere on the globe."""
    assert _haversine_m(45.0, 9.0, 46.0, 9.0) == pytest.approx(111_200, rel=0.01)


def test_haversine_is_symmetric() -> None:
    a = _haversine_m(45.4642, 9.1900, 45.4520, 9.1750)
    b = _haversine_m(45.4520, 9.1750, 45.4642, 9.1900)
    assert a == pytest.approx(b)


# --------------------------------------------------------------------------
# Centroids
# --------------------------------------------------------------------------


def test_every_study_cell_lies_within_milan(config) -> None:
    for square in (5161, 5059, 5259, 4159, 4556):
        lat, lon = cell_centroid(square, config)
        assert MILAN_LAT[0] < lat < MILAN_LAT[1], (square, lat)
        assert MILAN_LON[0] < lon < MILAN_LON[1], (square, lon)


def test_adjacent_cells_are_one_cell_width_apart(config) -> None:
    """Published cell size is 235 m, so neighbours should be about that far."""
    a = cell_centroid(5161, config)
    b = cell_centroid(5162, config)  # next column
    assert _haversine_m(*a, *b) == pytest.approx(235, abs=30)

    c = cell_centroid(5261, config)  # next row
    assert _haversine_m(*a, *c) == pytest.approx(235, abs=30)


def test_grid_index_agrees_with_compass_direction(config) -> None:
    """Row must increase northward and column eastward.

    If this is inverted, the spatial map in the exploratory analysis is
    reflected and every statement about where traffic concentrates is wrong.
    """
    lower_lat, lower_lon = cell_centroid(5059, config)
    upper_lat, _ = cell_centroid(5259, config)  # two rows north
    _, east_lon = cell_centroid(5161, config)

    assert grid_position(5259)[0] > grid_position(5059)[0]
    assert upper_lat > lower_lat, "higher row index must be further north"

    assert grid_position(5161)[1] > grid_position(5059)[1]
    assert east_lon > lower_lon, "higher column index must be further east"


def test_unknown_square_is_rejected(config) -> None:
    with pytest.raises(KeyError, match="not in the Milano Grid"):
        cell_centroid(99_999, config)


# --------------------------------------------------------------------------
# Landmarks
# --------------------------------------------------------------------------


def test_landmarks_are_inside_the_bounding_box() -> None:
    for landmark in LANDMARKS:
        assert MILAN_LAT[0] < landmark.latitude < MILAN_LAT[1], landmark.name
        assert MILAN_LON[0] < landmark.longitude < MILAN_LON[1], landmark.name


def test_nearest_landmarks_are_ordered_by_distance(config) -> None:
    near = nearest_landmarks(5161, config, count=5)
    distances = [d for _, d in near]
    assert distances == sorted(distances)


def test_the_three_busiest_cells_are_all_in_the_historic_centre(config) -> None:
    """The finding the report rests on: the top three are all central.

    Each sits within 500 m of the Duomo, the Galleria or La Scala, which is why
    forecasting all three would have sampled one traffic regime three times.
    """
    for square in (5161, 5059, 5259):
        landmark, distance = nearest_landmarks(square, config, count=1)[0]
        assert landmark.kind == "historic centre", (square, landmark.name)
        assert distance < 500, (square, landmark.name, distance)


def test_the_two_brief_specified_cells_are_not_central(config) -> None:
    """4159 and 4556 sit elsewhere, which is what makes them useful contrasts."""
    for square, expected in ((4159, "Universita Bocconi"), (4556, "Navigli")):
        landmark, distance = nearest_landmarks(square, config, count=1)[0]
        assert landmark.name == expected, (square, landmark.name)
        assert distance < 500, (square, distance)


def test_describe_cell_returns_a_table_row(config) -> None:
    row = describe_cell(4556, config)
    assert row["square_id"] == 4556
    assert row["nearest"] == "Navigli"
    assert row["nearest_m"] < 500
    assert "m" in row["context"]
    assert set(row) == {
        "square_id",
        "latitude",
        "longitude",
        "nearest",
        "nearest_m",
        "nearest_kind",
        "context",
    }
