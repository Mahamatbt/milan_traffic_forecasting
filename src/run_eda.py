"""Produce every exploratory artefact: figures, tables and the extracted series.

Runs headlessly so the analysis is reproducible from a clean clone with one
command, rather than depending on a notebook being executed in the right order.
The notebook imports the same functions and renders the same figures; nothing
is computed in two places.

Outputs
-------
``data/processed/selected_series.parquet``
    The five study series. Committed to the repository, because it is what lets
    the modelling stages reproduce without the 19.4 GiB download.
``results/tables/``
    ``selected_areas.json``, ``distribution_stats.json``, ``area_summary.csv``,
    ``stationarity.csv``, ``decomposition.json``, ``spectral_peaks.csv``,
    ``anomalies.csv``.
``results/figures/eda/``
    Every figure at 300 dpi.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Config, load_config
from src.eda import (
    area_summary,
    distribution_stats,
    extract_series,
    grid_position,
    select_areas,
    totals_as_grid,
)
from src.geo import describe_cell, grid_path
from src.holidays_it import describe, holiday_flags
from src.tsanalysis import (
    autocorrelation,
    decompose,
    periodogram_peaks,
    rolling_stats,
    seasonal_naive_anomalies,
    stationarity_table,
)

__all__ = ["run_eda", "load_inputs", "window_mask"]

from src.plots import FIGURE_DPI


def load_inputs(config: Config) -> dict[str, Any]:
    """Load the matrix, index and totals produced by the ingest stage."""
    import polars as pl

    processed = config.paths.processed
    matrix_path = processed / "traffic_matrix.npy"
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"{matrix_path} not found. Run the ingest and matrix stages first, or "
            "attach the processed dataset."
        )

    timestamps = pl.read_parquet(processed / "timestamps.parquet")
    return {
        # Memory-mapped: only the handful of columns we extract is materialised.
        "matrix": np.load(matrix_path, mmap_mode="r"),
        "square_ids": np.load(processed / "square_ids.npy"),
        "totals": pl.read_parquet(processed / "totals.parquet"),
        "local_times": timestamps["local"].to_numpy(),
        "utc_times": timestamps["utc"].to_numpy(),
    }


def _save(fig: Any, path: Path) -> Path:
    """Write a figure at report resolution and release it."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write a list of flat dicts as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def window_mask(local_times: np.ndarray, start: dt.date, days: int) -> np.ndarray:
    """Boolean mask selecting ``days`` local days from ``start``."""
    as_days = local_times.astype("datetime64[D]").astype(dt.date)
    end = start + dt.timedelta(days=days - 1)
    return np.array([start <= d <= end for d in as_days], dtype=bool)


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def _stage_distribution(data: dict[str, Any], config: Config) -> dict[str, Any]:
    """Spatial distribution of total traffic across the grid."""
    from src.plots import plot_total_distribution

    totals = data["totals"]["total_internet"].to_numpy()
    stats = distribution_stats(totals)
    _write_json(stats.to_dict(), config.paths.tables / "distribution_stats.json")

    figures = config.paths.figures / "eda"
    _save(plot_total_distribution(totals, stats), figures / "01_total_distribution.png")
    return {"stats": stats, "totals": totals}


def _stage_areas(data: dict[str, Any], config: Config, totals: np.ndarray) -> Any:
    """Resolve the study areas and map them."""
    from src.plots import plot_spatial_totals

    areas = select_areas(totals, data["square_ids"], config)
    areas.save(config.paths.tables / "selected_areas.json")

    labels = {square: f"#{areas.ranks[str(square)]}" for square in areas.all_plotted}
    _save(
        plot_spatial_totals(totals_as_grid(totals), highlight=labels),
        config.paths.figures / "eda" / "02_spatial_totals.png",
    )
    return areas


def _stage_series(data: dict[str, Any], config: Config, areas: Any) -> tuple[np.ndarray, list[int]]:
    """Extract the five study series, persist them, and plot the first fortnight."""
    import polars as pl

    from src.plots import plot_series_overlay, plot_series_panel

    wanted = areas.all_plotted
    series = extract_series(data["matrix"], data["square_ids"], wanted)

    frame = pl.DataFrame(
        {
            "utc": data["utc_times"],
            "local": data["local_times"],
            **{f"square_{s}": series[:, i] for i, s in enumerate(wanted)},
        }
    )
    frame.write_parquet(config.paths.processed / "selected_series.parquet", compression="zstd")

    # The brief asks for the first two weeks specifically.
    mask = window_mask(data["local_times"], config.dataset.start_date, 14)
    window_times = data["local_times"][mask]
    window = series[mask]
    notes = {s: f"rank {areas.ranks[str(s)]}" for s in wanted}

    figures = config.paths.figures / "eda"
    _save(
        plot_series_panel(
            window,
            wanted,
            window_times,
            title="Internet traffic, 1-14 November 2013",
            labels=notes,
        ),
        figures / "03_series_first_fortnight.png",
    )
    _save(
        plot_series_overlay(
            window,
            wanted,
            window_times,
            title="Normalised traffic, 1-14 November 2013",
        ),
        figures / "04_series_overlay_normalised.png",
    )

    rows = area_summary(series, wanted, data["local_times"])
    for row in rows:
        square = row["square_id"]
        row["rank"] = areas.ranks[str(square)]
        row["grid_row"], row["grid_col"] = grid_position(square)
        # Where the cell actually is, so the report can state land use rather
        # than infer it from the weekday/weekend ratio alone.
        if grid_path(config).exists():
            row.update({k: v for k, v in describe_cell(square, config).items() if k != "square_id"})
    _write_csv(rows, config.paths.tables / "area_summary.csv")
    return series, wanted


def _stage_timeseries(
    data: dict[str, Any], config: Config, series: np.ndarray, wanted: list[int], areas: Any
) -> dict[str, Any]:
    """The two required deeper analyses, on the highest-traffic area."""
    from src.plots import (
        plot_acf_pacf,
        plot_decomposition,
        plot_periodogram,
        plot_rolling_stats,
    )

    column = wanted.index(areas.top_traffic)
    y = series[:, column].astype(np.float64)
    figures = config.paths.figures / "eda"
    daily, weekly = config.dataset.daily_period, config.dataset.weekly_period

    # Analysis 1: multi-seasonal decomposition.
    started = time.perf_counter()
    decomposition = decompose(y, periods=(daily, weekly))
    decomposition_s = time.perf_counter() - started
    _write_json(
        {**decomposition.stats(), "square_id": areas.top_traffic, "wall_s": decomposition_s},
        config.paths.tables / "decomposition.json",
    )
    _save(
        plot_decomposition(decomposition, data["local_times"]),
        figures / "05_mstl_decomposition.png",
    )

    # Analysis 2: autocorrelation and stationarity.
    stationarity = stationarity_table(y, seasonal_period=daily)
    _write_csv([r.to_row() for r in stationarity], config.paths.tables / "stationarity.csv")

    acf = autocorrelation(y, nlags=weekly + 92)
    _save(
        plot_acf_pacf(acf, daily_period=daily, weekly_period=weekly),
        figures / "06_acf_pacf.png",
    )
    _write_json(
        {
            "square_id": areas.top_traffic,
            "lag_1": acf["lag_1"],
            "acf_at_daily": float(acf["acf"][daily]),
            "acf_at_weekly": float(acf["acf"][weekly]),
            "notable_lags": acf["notable_lags"],
            "notable_lag_values": acf["notable_lag_values"],
        },
        config.paths.tables / "autocorrelation.json",
    )

    rolling = rolling_stats(y, window=daily)
    _save(
        plot_rolling_stats(rolling, data["local_times"]),
        figures / "07_rolling_stats.png",
    )

    # Supporting evidence: which cycles actually carry power.
    spectrum = periodogram_peaks(y, interval_minutes=config.dataset.interval_minutes, top_n=8)
    _write_csv(spectrum["peaks"], config.paths.tables / "spectral_peaks.csv")
    _save(plot_periodogram(spectrum), figures / "08_periodogram.png")

    anomalies = _stage_anomalies(y, data, config)
    return {
        "decomposition": decomposition,
        "stationarity": stationarity,
        "acf": acf,
        "spectrum": spectrum,
        "anomalies": anomalies,
        "series": y,
    }


def _stage_anomalies(y: np.ndarray, data: dict[str, Any], config: Config) -> dict[str, Any]:
    """Flag where a seasonal-naive predictor fails, against the holiday calendar."""
    result = seasonal_naive_anomalies(y, seasonal_period=config.dataset.daily_period)
    local = data["local_times"]
    flags = holiday_flags(local)

    rows = []
    for index in result["index"]:
        stamp = local[index]
        day = stamp.astype("datetime64[D]").astype(dt.date)
        holiday = describe(day)
        rows.append(
            {
                "index": int(index),
                "local_time": str(stamp),
                "date": day.isoformat(),
                "weekday": day.strftime("%A"),
                "value": float(y[index]),
                "seasonal_naive_residual": float(
                    result["residual"][index - config.dataset.daily_period]
                ),
                "z_score": float(result["z_scores"][index - config.dataset.daily_period]),
                "is_holiday": bool(flags[index]),
                "holiday": holiday.name if holiday else "",
            }
        )

    if rows:
        _write_csv(rows, config.paths.tables / "anomalies.csv")
    on_holiday = sum(1 for r in rows if r["is_holiday"])
    return {
        "n_flagged": result["n_flagged"],
        "fraction_flagged": result["fraction_flagged"],
        "n_on_holidays": on_holiday,
        "share_on_holidays": on_holiday / len(rows) if rows else 0.0,
        "rows": rows,
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run_eda(config: Config) -> dict[str, Any]:
    """Run every exploratory stage and write all artefacts.

    Args:
        config: Study configuration.

    Returns:
        The computed objects, so a notebook can display them without recomputing.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed for the CLI path
    from src.notebook import apply_plot_style

    apply_plot_style()
    config.paths.mkdirs()

    data = load_inputs(config)
    print(f"matrix {data['matrix'].shape}, {data['square_ids'].size:,} squares")

    distribution = _stage_distribution(data, config)
    print("\n=== distribution across the grid ===")
    for line in distribution["stats"].summary_lines():
        print(" ", line)

    areas = _stage_areas(data, config, distribution["totals"])
    print("\n=== study areas ===")
    for square in areas.all_plotted:
        row, col = grid_position(square)
        print(
            f"  square {square:>5}  rank {areas.ranks[str(square)]:>5}/10000  "
            f"total {areas.totals[str(square)]:>13,.0f}  grid ({row}, {col})"
        )
    print(f"  forecasting: {areas.forecast}")

    series, wanted = _stage_series(data, config, areas)
    print(f"\nwrote selected_series.parquet with {len(wanted)} series")

    analysis = _stage_timeseries(data, config, series, wanted, areas)

    print(f"\n=== decomposition, square {areas.top_traffic} ===")
    for line in analysis["decomposition"].summary_lines():
        print(" ", line)

    print("\n=== stationarity ===")
    for row in analysis["stationarity"]:
        print(
            f"  {row.transform:<34} ADF p={row.adf_pvalue:<8.4f} "
            f"KPSS p={row.kpss_pvalue:<6.3f} -> {row.verdict}"
        )

    print("\n=== dominant cycles ===")
    for peak in analysis["spectrum"]["peaks"][:4]:
        print(f"  {peak['period_hours']:>9.2f} h   power {peak['power']:.3e}")

    anomalies = analysis["anomalies"]
    print(
        f"\n=== anomalies vs seasonal naive ===\n"
        f"  flagged {anomalies['n_flagged']} points "
        f"({anomalies['fraction_flagged']:.3%}), "
        f"{anomalies['n_on_holidays']} on holidays "
        f"({anomalies['share_on_holidays']:.0%})"
    )
    for row in anomalies["rows"][:8]:
        mark = f"  [{row['holiday']}]" if row["holiday"] else ""
        print(f"    {row['local_time']}  z={row['z_score']:6.1f}{mark}")

    figures = sorted((config.paths.figures / "eda").glob("*.png"))
    print(f"\n{len(figures)} figures at {FIGURE_DPI} dpi in {config.paths.figures / 'eda'}")
    for path in figures:
        print(f"  {path.name:<36} {path.stat().st_size / 1024:>7.0f} KB")

    return {"areas": areas, "distribution": distribution, "analysis": analysis, "data": data}


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="Path to a config YAML.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the exploratory analysis end to end."""
    args = build_parser().parse_args(argv)
    run_eda(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
