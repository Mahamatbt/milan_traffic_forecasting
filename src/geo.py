"""Locate grid cells on the ground, so claims about land use can be checked.

The exploratory analysis reads square 5259's weekend-to-weekday ratio of 0.425
as a business district and square 5161's 1.384 as a leisure area. Those are
inferences from a number, and a report should not assert what is physically at a
location without looking. This module does the lookup: it converts a square id
to the centroid of its cell using the published Milano Grid geometry, and
reports the nearest known landmarks.

The landmark list is deliberately small and confined to places whose coordinates
are unambiguous and checkable. A nearest-landmark distance is evidence about
where a cell is; it is not proof of what drives its traffic, and the report
should say so.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.config import Config

__all__ = ["Landmark", "LANDMARKS", "cell_centroid", "nearest_landmarks", "describe_cell"]

# Mean Earth radius, for the great-circle distances below.
EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class Landmark:
    """A reference point in Milan with a decimal-degree coordinate."""

    name: str
    latitude: float
    longitude: float
    kind: str


# Reference points spread across the city, chosen so that a cell anywhere in the
# central grid has something recognisable nearby. Coordinates are decimal
# degrees (WGS84) and are stated to four places, roughly 10 m precision.
LANDMARKS: tuple[Landmark, ...] = (
    Landmark("Duomo", 45.4642, 9.1900, "historic centre"),
    Landmark("Galleria Vittorio Emanuele II", 45.4659, 9.1899, "historic centre"),
    Landmark("Teatro alla Scala", 45.4674, 9.1895, "historic centre"),
    Landmark("Brera", 45.4719, 9.1881, "historic centre"),
    Landmark("Milano Centrale station", 45.4857, 9.2040, "transport hub"),
    Landmark("Porta Garibaldi station", 45.4848, 9.1875, "transport hub"),
    Landmark("Porta Venezia", 45.4749, 9.2049, "inner ring"),
    Landmark("Corso Buenos Aires", 45.4790, 9.2100, "retail"),
    Landmark("Navigli", 45.4520, 9.1750, "nightlife"),
    Landmark("Universita Bocconi", 45.4470, 9.1900, "university"),
    Landmark("Politecnico, Citta Studi", 45.4780, 9.2270, "university"),
    Landmark("Fiera Milano City / CityLife", 45.4780, 9.1560, "business / exhibition"),
    Landmark("San Siro stadium", 45.4781, 9.1240, "stadium"),
    Landmark("Porta Romana", 45.4497, 9.2043, "inner ring"),
    Landmark("Lambrate", 45.4850, 9.2380, "outer district"),
    Landmark("Bicocca", 45.5140, 9.2110, "university / outer"),
)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two decimal-degree points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


@lru_cache(maxsize=1)
def _load_grid(path_str: str) -> dict[int, tuple[float, float]]:
    """Map every cell id to its centroid, as ``(latitude, longitude)``.

    Cells are small enough (235 m) that averaging the polygon's vertices is
    within a few metres of the true centroid, so the ring is averaged directly
    rather than pulling in a geometry library for the difference.
    """
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    centroids: dict[int, tuple[float, float]] = {}

    for feature in payload["features"]:
        cell_id = int(feature["properties"]["cellId"])
        ring = feature["geometry"]["coordinates"][0]
        # GeoJSON polygons repeat the first vertex last; drop it before averaging.
        points = ring[:-1] if ring[0] == ring[-1] else ring
        longitude = sum(p[0] for p in points) / len(points)
        latitude = sum(p[1] for p in points) / len(points)
        centroids[cell_id] = (latitude, longitude)

    return centroids


def grid_path(config: Config) -> Path:
    """Location of the Milano Grid GeoJSON."""
    return config.paths.raw / "grid" / "milano-grid.geojson"


def cell_centroid(square_id: int, config: Config) -> tuple[float, float]:
    """Centroid of one grid cell as ``(latitude, longitude)``.

    Raises:
        FileNotFoundError: If the grid GeoJSON is absent.
        KeyError: If the square id is not in the grid.
    """
    path = grid_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It is committed to the repository; if missing, "
            "re-fetch it with: python run.py download --limit 0"
        )
    centroids = _load_grid(str(path))
    if square_id not in centroids:
        raise KeyError(f"square {square_id} is not in the Milano Grid")
    return centroids[square_id]


def nearest_landmarks(
    square_id: int, config: Config, *, count: int = 3
) -> list[tuple[Landmark, float]]:
    """The closest reference points to a cell, with distances in metres."""
    latitude, longitude = cell_centroid(square_id, config)
    ranked = sorted(
        (
            (landmark, _haversine_m(latitude, longitude, landmark.latitude, landmark.longitude))
            for landmark in LANDMARKS
        ),
        key=lambda pair: pair[1],
    )
    return ranked[:count]


def describe_cell(square_id: int, config: Config, *, count: int = 3) -> dict[str, object]:
    """Location summary for one cell, suitable for a table row."""
    latitude, longitude = cell_centroid(square_id, config)
    near = nearest_landmarks(square_id, config, count=count)
    return {
        "square_id": square_id,
        "latitude": round(latitude, 5),
        "longitude": round(longitude, 5),
        "nearest": near[0][0].name,
        "nearest_m": round(near[0][1]),
        "nearest_kind": near[0][0].kind,
        "context": "; ".join(f"{lm.name} {dist:.0f} m" for lm, dist in near),
    }
