"""Figures for the exploratory analysis.

Kept separate from the statistics so that the numbers can be computed, tested
and quoted without importing matplotlib, and so a figure never becomes the only
place a result exists. Every function takes already-computed values and returns
a ``Figure``; none of them calculates anything the report might cite.

Axis choices are deliberate rather than cosmetic. Cell totals span several
orders of magnitude, so distribution views use log axes; a linear histogram puts
99% of the grid in the first bin. Spatial maps use ``log10`` for the same
reason: on a linear colour scale the city centre saturates and everything else
reads as empty.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "FIGURE_DPI",
    "plot_total_distribution",
    "plot_spatial_totals",
    "plot_series_panel",
    "plot_series_overlay",
    "plot_decomposition",
    "plot_acf_pacf",
    "plot_rolling_stats",
    "plot_periodogram",
    "plot_forecast",
    "plot_error_heatmap",
    "plot_error_by_hour",
    "plot_residual_acf",
    "plot_cross_correlation",
    "plot_cross_area_mase",
]

# Print resolution for every exported figure. Defined once: the exploratory and
# results stages previously set this independently and drifted to 300 and 150,
# so half the report's figures were at half the intended resolution.
FIGURE_DPI = 300

# Colour-blind safe, distinguishable in greyscale print.
_SERIES_COLOURS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def _figure(*args: Any, **kwargs: Any):
    """Create a figure without requiring matplotlib at import time."""
    import matplotlib.pyplot as plt

    return plt.subplots(*args, **kwargs)


# --------------------------------------------------------------------------
# Spatial distribution
# --------------------------------------------------------------------------


def plot_total_distribution(totals: np.ndarray, stats: Any):
    """Histogram on a log axis beside the complementary CDF on log-log axes.

    The two views answer different questions. The histogram shows where the mass
    of cells sits; the CCDF shows how the tail decays, which is what decides
    whether a handful of cells dominate the city's traffic.

    Args:
        totals: Full-period total per cell.
        stats: A :class:`~src.eda.DistributionStats` for the annotation.

    Returns:
        The figure.
    """
    values = np.asarray(totals, dtype=np.float64)
    positive = values[values > 0]

    fig, (left, right) = _figure(1, 2, figsize=(12, 4.5))

    bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 60)
    left.hist(positive, bins=bins, color=_SERIES_COLOURS[0], alpha=0.85, edgecolor="none")
    left.set_xscale("log")
    left.axvline(stats.median, color=_SERIES_COLOURS[1], ls="--", lw=1.2, label="median")
    left.axvline(stats.mean, color=_SERIES_COLOURS[2], ls=":", lw=1.4, label="mean")
    left.set_xlabel("total internet activity over the period (log scale)")
    left.set_ylabel("number of cells")
    left.set_title("Distribution of total traffic across 10,000 cells")
    left.legend()

    ordered = np.sort(positive)[::-1]
    ccdf = np.arange(1, ordered.size + 1) / ordered.size
    right.loglog(ordered, ccdf, color=_SERIES_COLOURS[0], lw=1.6)
    right.set_xlabel("total internet activity")
    right.set_ylabel("P(X > x)")
    right.set_title("Complementary CDF")
    right.annotate(
        f"Gini {stats.gini:.3f}\ntop 1% hold {stats.top_1pct_share:.0%}\n"
        f"max/median {stats.max_over_median:,.0f}x",
        xy=(0.04, 0.06),
        xycoords="axes fraction",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout()
    return fig


def plot_spatial_totals(grid: np.ndarray, highlight: dict[int, str] | None = None):
    """Map ``log10`` total activity over the 100x100 grid.

    Args:
        grid: 2-D totals from :func:`~src.eda.totals_as_grid`.
        highlight: Optional ``{square_id: label}`` to mark the study areas.

    Returns:
        The figure.
    """
    from src.eda import grid_position

    values = np.asarray(grid, dtype=np.float64)
    with np.errstate(divide="ignore"):
        shown = np.log10(np.where(values > 0, values, np.nan))

    fig, axis = _figure(figsize=(7.5, 6.5))
    image = axis.imshow(shown, origin="lower", cmap="magma", interpolation="nearest")
    fig.colorbar(image, ax=axis, label="log10(total internet activity)")

    for square, label in (highlight or {}).items():
        row, column = grid_position(int(square))
        axis.plot(column, row, marker="o", ms=9, mfc="none", mec="#00E5FF", mew=1.8)
        axis.annotate(
            label,
            xy=(column, row),
            xytext=(6, 6),
            textcoords="offset points",
            color="#00E5FF",
            fontsize=9,
            fontweight="bold",
        )

    axis.set_xlabel("grid column")
    axis.set_ylabel("grid row")
    axis.set_title("Spatial distribution of total traffic (Milano grid)")
    axis.grid(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Series views
# --------------------------------------------------------------------------


def plot_series_panel(
    series: np.ndarray,
    square_ids: list[int],
    times: np.ndarray,
    *,
    title: str = "",
    labels: dict[int, str] | None = None,
):
    """Stacked panels, one per area, sharing a time axis.

    Each panel keeps its own y-scale: the areas differ by an order of magnitude
    in volume, and a shared scale would flatten the smaller ones into a line
    while saying nothing the totals table does not already say.

    Args:
        series: ``(n_timestamps, n_areas)`` values.
        square_ids: Ids matching the columns.
        times: Local timestamps for the x-axis.
        title: Figure title.
        labels: Optional extra description per square id.

    Returns:
        The figure.
    """
    values = np.asarray(series)
    count = len(square_ids)
    fig, axes = _figure(count, 1, figsize=(12, 1.9 * count), sharex=True)
    axes = np.atleast_1d(axes)

    for index, (axis, square) in enumerate(zip(axes, square_ids, strict=False)):
        axis.plot(
            times,
            values[:, index],
            color=_SERIES_COLOURS[index % len(_SERIES_COLOURS)],
            lw=0.9,
        )
        note = (labels or {}).get(square, "")
        axis.set_ylabel(f"square {square}", fontsize=9)
        if note:
            axis.annotate(
                note,
                xy=(0.995, 0.88),
                xycoords="axes fraction",
                ha="right",
                fontsize=8,
                alpha=0.75,
            )
    axes[-1].set_xlabel("local time (Europe/Rome)")
    if title:
        axes[0].set_title(title)
    fig.tight_layout()
    return fig


def plot_series_overlay(
    series: np.ndarray,
    square_ids: list[int],
    times: np.ndarray,
    *,
    title: str = "",
):
    """All areas on one axis, each scaled to its own maximum.

    Normalising is what makes the comparison about *shape* rather than volume:
    whether the areas share a diurnal profile, and where they diverge.

    Args:
        series: ``(n_timestamps, n_areas)`` values.
        square_ids: Ids matching the columns.
        times: Local timestamps.
        title: Figure title.

    Returns:
        The figure.
    """
    values = np.asarray(series, dtype=np.float64)
    fig, axis = _figure(figsize=(12, 4.5))

    for index, square in enumerate(square_ids):
        column = values[:, index]
        peak = column.max()
        axis.plot(
            times,
            column / peak if peak else column,
            lw=1.0,
            alpha=0.85,
            color=_SERIES_COLOURS[index % len(_SERIES_COLOURS)],
            label=f"square {square}",
        )

    axis.set_ylabel("activity / area maximum")
    axis.set_xlabel("local time (Europe/Rome)")
    axis.set_title(title or "Normalised traffic, all study areas")
    axis.legend(ncol=min(5, len(square_ids)), fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Time-series diagnostics
# --------------------------------------------------------------------------


def plot_decomposition(result: Any, times: np.ndarray | None = None):
    """Observed series with its MSTL trend, seasonal terms and residual.

    Args:
        result: A :class:`~src.tsanalysis.DecompositionResult`.
        times: Optional x-axis values; defaults to sample index.

    Returns:
        The figure.
    """
    components = [
        ("observed", result.observed),
        ("trend", result.trend),
        *result.seasonal.items(),
        ("residual", result.residual),
    ]
    x = times if times is not None else np.arange(result.observed.size)

    fig, axes = _figure(len(components), 1, figsize=(12, 1.75 * len(components)), sharex=True)
    for index, (axis, (name, values)) in enumerate(
        zip(np.atleast_1d(axes), components, strict=False)
    ):
        axis.plot(x, values, lw=0.8, color=_SERIES_COLOURS[index % len(_SERIES_COLOURS)])
        share = result.variance_share.get(name)
        label = name if share is None else f"{name}\n{share:.1%} of variance"
        axis.set_ylabel(label, fontsize=8)
    axes[0].set_title(f"MSTL decomposition, periods {result.periods}")
    axes[-1].set_xlabel("local time (Europe/Rome)" if times is not None else "sample")
    fig.tight_layout()
    return fig


def plot_acf_pacf(analysis: dict[str, Any], *, daily_period: int, weekly_period: int):
    """ACF and PACF with confidence bands and the seasonal lags marked."""
    fig, (top, bottom) = _figure(2, 1, figsize=(12, 6))

    acf_values = analysis["acf"]
    lags = np.arange(acf_values.size)
    top.vlines(lags, 0, acf_values, color=_SERIES_COLOURS[0], lw=0.6)
    band = analysis["acf_confint"] - acf_values[:, None]
    top.fill_between(lags, band[:, 0], band[:, 1], color="grey", alpha=0.25, lw=0)
    for period, style in ((daily_period, "--"), (weekly_period, ":")):
        for multiple in range(period, acf_values.size, period):
            top.axvline(multiple, color=_SERIES_COLOURS[1], ls=style, lw=0.8, alpha=0.6)
    top.set_ylabel("ACF")
    top.set_title(
        f"Autocorrelation to lag {analysis['nlags']} "
        f"(dashed = daily {daily_period}, dotted = weekly {weekly_period})"
    )

    pacf_values = analysis["pacf"]
    plags = np.arange(pacf_values.size)
    bottom.vlines(plags, 0, pacf_values, color=_SERIES_COLOURS[2], lw=0.6)
    pband = analysis["pacf_confint"] - pacf_values[:, None]
    bottom.fill_between(plags, pband[:, 0], pband[:, 1], color="grey", alpha=0.25, lw=0)
    bottom.set_ylabel("PACF")
    bottom.set_xlabel("lag (10-minute intervals)")

    fig.tight_layout()
    return fig


def plot_rolling_stats(stats: dict[str, np.ndarray], times: np.ndarray | None = None):
    """Rolling mean and standard deviation, for a visual stationarity check."""
    fig, axis = _figure(figsize=(12, 4))
    x = times[stats["index"]] if times is not None else stats["index"]

    axis.plot(x, stats["mean"], color=_SERIES_COLOURS[0], lw=1.2, label="rolling mean")
    axis.plot(x, stats["std"], color=_SERIES_COLOURS[1], lw=1.2, label="rolling std")
    axis.set_xlabel("local time (Europe/Rome)" if times is not None else "sample")
    axis.set_ylabel("activity")
    axis.set_title("Rolling mean and standard deviation (1-day window)")
    axis.legend()
    fig.tight_layout()
    return fig


def plot_periodogram(spectrum: dict[str, Any], *, max_period_hours: float = 200.0):
    """Spectral power against period in hours, with the top peaks annotated."""
    fig, axis = _figure(figsize=(12, 4))

    frequencies = spectrum["frequencies"]
    power = spectrum["power"]
    keep = frequencies > 1.0 / max_period_hours
    periods = 1.0 / frequencies[keep]

    axis.semilogx(periods, power[keep], color=_SERIES_COLOURS[0], lw=1.0)
    for peak in spectrum["peaks"][:4]:
        if peak["period_hours"] <= max_period_hours:
            axis.axvline(peak["period_hours"], color=_SERIES_COLOURS[1], ls="--", lw=0.9)
            axis.annotate(
                f"{peak['period_hours']:.1f} h",
                xy=(peak["period_hours"], peak["power"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9,
            )
    axis.set_xlabel("period (hours, log scale)")
    axis.set_ylabel("spectral power")
    axis.set_title("Periodogram: which cycles carry the variance")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Forecast evaluation
# --------------------------------------------------------------------------


def plot_forecast(
    times: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    model: str,
    square_id: int,
    persistence: np.ndarray | None = None,
    zoom_days: int = 1,
    daily_period: int = 144,
):
    """A forecast over the full window, with one day enlarged beneath it.

    The full week shows whether the level and seasonal shape are right. At that
    width a one-step lag is invisible, because a week of 10-minute data is 1,008
    points drawn across a few hundred pixels, and every model looks like the
    observed series. The zoomed panel is where copying becomes visible to the
    eye, which is why it is part of the same figure rather than an optional
    extra.

    Args:
        times: Local timestamps for the evaluation window.
        observed: True values.
        predicted: Forecasts aligned to ``observed``.
        model: Model name, for the title.
        square_id: Area identifier, for the title.
        persistence: Baseline forecasts to overlay on the zoom panel.
        zoom_days: How many days the lower panel covers.
        daily_period: Steps per day.

    Returns:
        The figure.
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    fig, (top, bottom) = _figure(2, 1, figsize=(12, 6.5))

    top.plot(times, observed, lw=1.0, color="#333333", label="observed")
    top.plot(times, predicted, lw=1.0, color=_SERIES_COLOURS[1], alpha=0.85, label=model)
    top.set_ylabel("internet activity")
    top.set_title(f"{model}, square {square_id} - full evaluation window")
    top.legend(fontsize=9, ncol=2)

    # The busiest day is the informative one: absolute errors scale with level,
    # so a quiet day would make every model look equally good.
    n_days = max(1, observed.size // daily_period)
    totals = [observed[d * daily_period : (d + 1) * daily_period].sum() for d in range(n_days)]
    peak_day = min(int(np.argmax(totals)), max(0, n_days - zoom_days))
    window = slice(peak_day * daily_period, (peak_day + zoom_days) * daily_period)

    bottom.plot(times[window], observed[window], lw=1.4, color="#333333", label="observed")
    bottom.plot(times[window], predicted[window], lw=1.4, color=_SERIES_COLOURS[1], label=model)
    if persistence is not None:
        bottom.plot(
            times[window],
            np.asarray(persistence)[window],
            lw=1.0,
            ls="--",
            color=_SERIES_COLOURS[2],
            alpha=0.9,
            label="persistence",
        )
    bottom.set_ylabel("internet activity")
    bottom.set_xlabel("local time (Europe/Rome)")
    bottom.set_title("Busiest day enlarged: a one-step lag is only visible at this scale")
    bottom.legend(fontsize=9, ncol=3)

    fig.tight_layout()
    return fig


def plot_error_heatmap(heatmap: dict[str, Any], *, model: str, square_id: int):
    """Mean absolute error over day-of-week by hour-of-day.

    Args:
        heatmap: Output of :func:`src.diagnostics.error_heatmap`.
        model: Model name, for the title.
        square_id: Area identifier.

    Returns:
        The figure.
    """
    grid = np.asarray(heatmap["grid"], dtype=np.float64)
    fig, axis = _figure(figsize=(11, 3.6))

    image = axis.imshow(grid, aspect="auto", origin="upper", cmap="magma")
    axis.set_yticks(range(7))
    axis.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axis.set_xticks(range(0, 24, 2))
    axis.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
    axis.set_xlabel("hour of day (Europe/Rome)")
    axis.set_title(f"{model}, square {square_id} - mean absolute error")

    bar = fig.colorbar(image, ax=axis, pad=0.02)
    bar.set_label("MAE")
    fig.tight_layout()
    return fig


def plot_error_by_hour(by_hour: dict[str, dict[str, Any]], *, square_id: int):
    """Error against hour of day, one line per model.

    Args:
        by_hour: Model name to the output of :func:`src.diagnostics.error_by_hour`.
        square_id: Area identifier.

    Returns:
        The figure.
    """
    fig, axis = _figure(figsize=(11, 4.0))

    for index, (model, result) in enumerate(by_hour.items()):
        axis.plot(
            result["hours"],
            result["mae"],
            marker="o",
            ms=3,
            lw=1.3,
            color=_SERIES_COLOURS[index % len(_SERIES_COLOURS)],
            label=model,
        )

    axis.set_xticks(range(0, 24, 2))
    axis.set_xlabel("hour of day (Europe/Rome)")
    axis.set_ylabel("MAE")
    axis.set_title(f"Error by hour of day, square {square_id}")
    axis.legend(fontsize=9, ncol=min(3, max(1, len(by_hour))))
    axis.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_residual_acf(acfs: dict[str, dict[str, Any]], *, square_id: int, daily_period: int = 144):
    """Residual autocorrelation, one panel per model.

    Args:
        acfs: Model name to the output of :func:`src.diagnostics.residual_acf`.
        square_id: Area identifier.
        daily_period: Steps per day, marked to show leftover seasonality.

    Returns:
        The figure.
    """
    models = list(acfs)
    fig, axes = _figure(len(models), 1, figsize=(11, 2.6 * len(models)), sharex=True)
    if len(models) == 1:
        axes = [axes]

    for axis, model in zip(axes, models, strict=True):
        result = acfs[model]
        lags, values = result["lags"][1:], result["acf"][1:]
        axis.bar(lags, values, width=1.0, color=_SERIES_COLOURS[0])
        bound = result["confidence"]
        axis.axhline(bound, color="grey", ls="--", lw=0.8)
        axis.axhline(-bound, color="grey", ls="--", lw=0.8)
        if lags.size and lags.max() >= daily_period:
            axis.axvline(daily_period, color=_SERIES_COLOURS[1], lw=1.0, alpha=0.8)
        axis.set_ylabel(model, fontsize=9)

    axes[-1].set_xlabel("lag (10-minute steps)")
    axes[0].set_title(
        f"Residual autocorrelation, square {square_id} "
        "(structure here is signal the model did not use)"
    )
    fig.tight_layout()
    return fig


def plot_cross_correlation(correlations: dict[str, dict[str, Any]], *, square_id: int):
    """Forecast-to-observation cross-correlation, one line per model.

    A peak at lag 0 means the forecast tracks the series. A peak at lag 1 means
    it reproduces the previous observation, which is the collapse this figure
    exists to make visible.

    Args:
        correlations: Model name to the output of
            :func:`src.diagnostics.cross_correlation`.
        square_id: Area identifier.

    Returns:
        The figure.
    """
    fig, axis = _figure(figsize=(9.5, 4.0))

    for index, (model, result) in enumerate(correlations.items()):
        axis.plot(
            result["lags"],
            result["correlation"],
            marker="o",
            ms=3,
            lw=1.2,
            color=_SERIES_COLOURS[index % len(_SERIES_COLOURS)],
            label=f"{model} (peak at {result['peak_lag']:+d})",
        )

    axis.axvline(0, color="grey", lw=0.8)
    axis.axvline(1, color=_SERIES_COLOURS[1], lw=0.8, ls="--", alpha=0.7)
    axis.set_xlabel("lag (10-minute steps; +1 means the forecast repeats the last observation)")
    axis.set_ylabel("correlation")
    axis.set_title(f"Forecast vs observation cross-correlation, square {square_id}")
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_cross_area_mase(table: list[dict[str, Any]], *, areas: list[int]):
    """Grouped MASE bars, one group per model.

    MASE is used because the areas differ by an order of magnitude in volume,
    so raw error is not comparable between them.

    Args:
        table: Output of :func:`src.diagnostics.cross_area_table`.
        areas: Area identifiers, in the order to plot them.

    Returns:
        The figure.
    """
    models = [row["model"] for row in table]
    fig, axis = _figure(figsize=(11, 4.4))

    width = 0.8 / max(len(areas), 1)
    positions = np.arange(len(models))
    for index, area in enumerate(areas):
        values = [
            row[f"mase_{area}"] if row.get(f"mase_{area}") is not None else np.nan for row in table
        ]
        axis.bar(
            positions + index * width,
            values,
            width=width,
            color=_SERIES_COLOURS[index % len(_SERIES_COLOURS)],
            label=f"square {area}",
        )

    axis.axhline(1.0, color="grey", ls="--", lw=0.9)
    axis.text(0.01, 1.02, "MASE 1.0 = in-sample naive", fontsize=8, color="grey")
    axis.set_xticks(positions + width * (len(areas) - 1) / 2)
    axis.set_xticklabels(models, rotation=20, ha="right")
    axis.set_ylabel("MASE")
    axis.set_title("Cross-area comparison (lower is better)")
    axis.legend(fontsize=9, ncol=max(1, len(areas)))
    fig.tight_layout()
    return fig
