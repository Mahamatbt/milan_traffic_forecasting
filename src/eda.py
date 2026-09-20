"""Exploratory analysis of the spatial traffic distribution and study areas.

Every function here is pure: it takes arrays and configuration and returns a
figure together with a ``stats`` dictionary. Notebooks call them and render;
nothing in this module reads a notebook's state or writes narrative. The
``stats`` dictionaries exist so that no figure in the report is unaccompanied by
numbers -- a plot that cannot be quoted is decoration.

The distribution work is deliberately on log axes. Cell totals span several
orders of magnitude, so a linear histogram collapses 99% of the grid into the
first bin and shows nothing. The CCDF on log-log axes is the companion view: it
makes the tail's shape legible where a histogram's tail is mostly empty bins.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Config

__all__ = [
    "DistributionStats",
    "SelectedAreas",
    "distribution_stats",
    "gini",
    "select_areas",
    "extract_series",
    "area_summary",
    "grid_position",
]


# --------------------------------------------------------------------------
# Distribution across the grid
# --------------------------------------------------------------------------


@dataclass
class DistributionStats:
    """Summary of how total traffic is distributed over the 10,000 cells."""

    n_cells: int
    total: float
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    skewness: float
    kurtosis: float
    gini: float
    top_1pct_share: float
    top_5pct_share: float
    top_10pct_share: float
    bottom_50pct_share: float
    n_zero_cells: int
    max_over_median: float
    lognormal_mu: float
    lognormal_sigma: float
    lognormal_ks_stat: float
    lognormal_ks_pvalue: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_lines(self) -> list[str]:
        """Human-readable lines for a notebook cell."""
        return [
            f"cells               : {self.n_cells:,}",
            f"total activity      : {self.total:,.0f}",
            f"mean / median       : {self.mean:,.0f} / {self.median:,.0f}",
            f"max / median        : {self.max_over_median:,.1f}x",
            f"skewness / kurtosis : {self.skewness:.2f} / {self.kurtosis:.2f}",
            f"Gini                : {self.gini:.3f}",
            f"top 1% / 5% / 10%   : {self.top_1pct_share:.1%} / "
            f"{self.top_5pct_share:.1%} / {self.top_10pct_share:.1%} of all traffic",
            f"bottom 50%          : {self.bottom_50pct_share:.1%}",
            f"lognormal fit       : mu={self.lognormal_mu:.3f} "
            f"sigma={self.lognormal_sigma:.3f} "
            f"(KS={self.lognormal_ks_stat:.4f}, p={self.lognormal_ks_pvalue:.3g})",
        ]


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative distribution.

    0 means every cell carries identical traffic; 1 means one cell carries all
    of it. Reported because it summarises spatial concentration in a single
    number that is comparable across datasets, where a skewness figure is not.

    Args:
        values: Non-negative totals, one per cell.

    Returns:
        The Gini coefficient, or 0.0 when every value is zero.

    Raises:
        ValueError: If any value is negative.
    """
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.size == 0:
        return 0.0
    if array[0] < 0:
        raise ValueError("Gini is undefined for negative values")
    total = array.sum()
    if total == 0:
        return 0.0
    index = np.arange(1, array.size + 1)
    return float((2 * index - array.size - 1).dot(array) / (array.size * total))


def _tail_share(sorted_desc: np.ndarray, fraction: float) -> float:
    """Share of the total held by the top ``fraction`` of cells."""
    cutoff = max(1, int(round(sorted_desc.size * fraction)))
    return float(sorted_desc[:cutoff].sum() / sorted_desc.sum())


def distribution_stats(totals: np.ndarray) -> DistributionStats:
    """Characterise the per-cell total distribution.

    Args:
        totals: Full-period total activity per cell, one entry per square.

    Returns:
        A :class:`DistributionStats` the report can quote directly.
    """
    from scipy import stats as sps

    values = np.asarray(totals, dtype=np.float64)
    descending = np.sort(values)[::-1]
    positive = values[values > 0]

    # Fit a lognormal to the positive cells: shape is sigma of log(x), and the
    # location is pinned at 0 so the fit stays a genuine two-parameter lognormal
    # rather than absorbing a shift that would make the fit meaningless.
    sigma, _, scale = sps.lognorm.fit(positive, floc=0)
    mu = float(np.log(scale))
    ks = sps.kstest(positive, "lognorm", args=(sigma, 0, scale))

    median = float(np.median(values))
    return DistributionStats(
        n_cells=int(values.size),
        total=float(values.sum()),
        mean=float(values.mean()),
        median=median,
        std=float(values.std(ddof=1)),
        minimum=float(values.min()),
        maximum=float(values.max()),
        skewness=float(sps.skew(values)),
        kurtosis=float(sps.kurtosis(values)),
        gini=gini(values),
        top_1pct_share=_tail_share(descending, 0.01),
        top_5pct_share=_tail_share(descending, 0.05),
        top_10pct_share=_tail_share(descending, 0.10),
        bottom_50pct_share=float(np.sort(values)[: values.size // 2].sum() / values.sum()),
        n_zero_cells=int((values == 0).sum()),
        max_over_median=float(values.max() / median) if median else float("inf"),
        lognormal_mu=mu,
        lognormal_sigma=float(sigma),
        lognormal_ks_stat=float(ks.statistic),
        lognormal_ks_pvalue=float(ks.pvalue),
    )


def grid_position(square_id: int, grid_side: int = 100) -> tuple[int, int]:
    """Convert a square id to its ``(row, column)`` in the Milano grid.

    Square ids run 1..10000 in row-major order over a 100x100 grid of 235 m
    cells, so id 1 is one corner and 10000 the opposite one.

    Args:
        square_id: Identifier in 1..grid_side**2.
        grid_side: Cells per side.

    Returns:
        Zero-based ``(row, column)``.

    Raises:
        ValueError: If the id falls outside the grid.
    """
    if not 1 <= square_id <= grid_side * grid_side:
        raise ValueError(f"square_id {square_id} outside 1..{grid_side**2}")
    zero_based = square_id - 1
    return zero_based // grid_side, zero_based % grid_side


def totals_as_grid(totals: np.ndarray, grid_side: int = 100) -> np.ndarray:
    """Reshape per-cell totals into the 2-D grid for spatial plotting."""
    values = np.asarray(totals, dtype=np.float64)
    if values.size != grid_side * grid_side:
        raise ValueError(f"expected {grid_side**2} totals, got {values.size}")
    return values.reshape(grid_side, grid_side)


# --------------------------------------------------------------------------
# Area selection
# --------------------------------------------------------------------------


@dataclass
class SelectedAreas:
    """The areas this study analyses, resolved from the data.

    ``forecast`` is the three highest-traffic cells, which Section 4 models.
    The ``fixed`` set (5059, 5259) is retained for exploratory comparison but
    is not forecast.
    """

    top_traffic: int
    top_three: list[int]
    fixed: list[int]
    forecast: list[int]
    ranks: dict[str, int] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=dict)

    @property
    def all_plotted(self) -> list[int]:
        """The five series the exploratory figures show, without duplicates."""
        ordered = [*self.top_three, *self.fixed]
        seen: list[int] = []
        for square in ordered:
            if square not in seen:
                seen.append(square)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> Path:
        """Persist so downstream stages read the resolved ids, not a guess."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> SelectedAreas:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


def select_areas(totals: np.ndarray, square_ids: np.ndarray, config: Config) -> SelectedAreas:
    """Resolve the study areas from measured totals.

    Args:
        totals: Full-period total per cell.
        square_ids: Square id for each entry of ``totals``.
        config: Study configuration, supplying the ids fixed by the brief.

    Returns:
        A :class:`SelectedAreas` recording ids, ranks and totals.
    """
    values = np.asarray(totals, dtype=np.float64)
    ids = np.asarray(square_ids)
    order = np.argsort(values)[::-1]
    top_three = [int(ids[i]) for i in order[:3]]
    fixed = [int(s) for s in config.areas.fixed]

    rank_of = {int(ids[position]): index + 1 for index, position in enumerate(order)}
    interesting = [*top_three, *fixed]

    return SelectedAreas(
        top_traffic=top_three[0],
        top_three=top_three,
        fixed=fixed,
        forecast=list(top_three),
        ranks={str(s): rank_of[s] for s in interesting},
        totals={str(s): float(values[ids == s][0]) for s in interesting},
    )


# --------------------------------------------------------------------------
# Series extraction
# --------------------------------------------------------------------------


def extract_series(
    matrix: np.ndarray,
    square_ids: np.ndarray,
    wanted: list[int],
) -> np.ndarray:
    """Pull the columns for ``wanted`` out of the traffic matrix.

    The matrix is memory-mapped by callers, so this is the one place that
    materialises anything: a handful of columns, a few hundred KB, rather than
    the full 341 MiB.

    Args:
        matrix: ``(n_timestamps, n_squares)`` array or memmap.
        square_ids: Column ordering of ``matrix``.
        wanted: Square ids to extract, in the order required.

    Returns:
        A ``(n_timestamps, len(wanted))`` float32 array.

    Raises:
        KeyError: If a requested square is not present.
    """
    lookup = {int(value): index for index, value in enumerate(np.asarray(square_ids))}
    missing = [s for s in wanted if int(s) not in lookup]
    if missing:
        raise KeyError(f"square id(s) not in the matrix: {missing}")
    columns = [lookup[int(s)] for s in wanted]
    return np.asarray(matrix[:, columns], dtype=np.float32)


def area_summary(
    series: np.ndarray,
    square_ids: list[int],
    local_times: np.ndarray,
) -> list[dict[str, Any]]:
    """Per-area descriptive statistics relevant to forecasting difficulty.

    The coefficient of variation and the peak-to-trough ratio are the ones that
    matter for the comparison: a series with a high night floor relative to its
    peak is easier to forecast than one that collapses to near zero, regardless
    of its absolute volume.

    Args:
        series: ``(n_timestamps, n_areas)`` traffic values.
        square_ids: Ids matching the columns of ``series``.
        local_times: Local wall-clock timestamps, one per row.

    Returns:
        One dict per area, ready to become a table row.
    """
    values = np.asarray(series, dtype=np.float64)
    hours = local_times.astype("datetime64[h]").astype(object)
    weekday = np.array([t.weekday() for t in hours])
    hour_of_day = np.array([t.hour for t in hours])
    is_weekend = weekday >= 5
    is_night = (hour_of_day >= 2) & (hour_of_day < 5)

    rows = []
    for index, square in enumerate(square_ids):
        column = values[:, index]
        mean = float(column.mean())
        weekday_mean = float(column[~is_weekend].mean())
        weekend_mean = float(column[is_weekend].mean())
        night_floor = float(np.median(column[is_night])) if is_night.any() else float("nan")
        trough = float(np.percentile(column, 1))
        rows.append(
            {
                "square_id": int(square),
                "mean": mean,
                "median": float(np.median(column)),
                "std": float(column.std(ddof=1)),
                "cv": float(column.std(ddof=1) / mean) if mean else float("nan"),
                "min": float(column.min()),
                "max": float(column.max()),
                "peak_to_trough": float(column.max() / trough) if trough else float("inf"),
                "night_floor": night_floor,
                "night_floor_over_mean": night_floor / mean if mean else float("nan"),
                "weekday_mean": weekday_mean,
                "weekend_mean": weekend_mean,
                "weekend_over_weekday": weekend_mean / weekday_mean
                if weekday_mean
                else float("nan"),
                "n_zero": int((column == 0).sum()),
            }
        )
    return rows
