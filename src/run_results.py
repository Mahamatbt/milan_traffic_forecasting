"""Phase 6: evaluation figures, diagnostics and failure analysis.

Reads the committed prediction series rather than refitting anything. The
forecasts are already persisted for both evaluation windows, and the LSTM
columns cannot be reproduced without a CUDA device, so recomputing them here
would make the figures un-drawable on the machine that writes the report.

Every number rendered as a figure is also written to a table, so nothing the
write-up cites exists only inside a PNG.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Config, load_config
from src.diagnostics import (
    copying_report,
    cross_area_table,
    cross_correlation,
    error_by_daytype,
    error_by_hour,
    error_heatmap,
    residual_acf,
    worst_windows,
)
from src.eda import SelectedAreas
from src.evaluate import load_area_series
from src.plots import FIGURE_DPI
from src.splits import make_splits

__all__ = ["run_results", "load_predictions", "MODELS"]

# Baselines first so every figure legend puts the floor before the models.
MODELS = ("persistence", "seasonal_naive", "harmonic_arima", "lightgbm", "lstm", "lstm_ensemble")

# Models whose forecasts are plotted in the actual-vs-predicted grid. The plan
# asks for three models across three areas; the ensemble is shown because it is
# the series the LSTM column of the predictions file actually contains.
PLOTTED = ("harmonic_arima", "lightgbm", "lstm")


def _relative(path: Path) -> str:
    """A project-relative POSIX path, for artefacts that get committed."""
    from src.config import PROJECT_ROOT

    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # Outside the project (a Kaggle session, or a redirected test run).
        return Path(path).as_posix()


def _save(fig: Any, path: Path) -> Path:
    """Write a figure and close it."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write a list of flat dicts as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_predictions(config: Config, square_id: int, split_name: str) -> dict[str, np.ndarray]:
    """Observed and predicted series for one area and split.

    Raises:
        FileNotFoundError: If the prediction file has not been produced.
    """
    import polars as pl

    path = config.paths.predictions / f"{split_name}_area_{square_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the final fits (notebooks/03_kaggle_final.ipynb) "
            "and merge their output before producing results."
        )
    frame = pl.read_parquet(path)
    return {column: frame[column].to_numpy() for column in frame.columns}


def _area_diagnostics(config: Config, square_id: int, split_name: str) -> dict[str, Any]:
    """Every diagnostic for one area, computed from the saved forecasts."""
    columns = load_predictions(config, square_id, split_name)
    observed = columns["observed"]
    times = columns["local"]

    values, all_times = load_area_series(config, square_id)
    splits = make_splits(values, all_times, config)
    split = splits[split_name]

    persistence = columns.get("persistence")
    if persistence is None:
        persistence = splits.series[split.start - 1 : split.stop - 1]

    present = [m for m in MODELS if m in columns]
    return {
        "square_id": square_id,
        "split": split_name,
        "times": times,
        "observed": observed,
        "persistence": persistence,
        "models": present,
        "columns": columns,
        "by_hour": {m: error_by_hour(observed, columns[m], times) for m in present},
        "daytype": {m: error_by_daytype(observed, columns[m], times) for m in present},
        "heatmaps": {m: error_heatmap(observed, columns[m], times) for m in present},
        "acfs": {m: residual_acf(observed, columns[m]) for m in present},
        "correlations": {m: cross_correlation(observed, columns[m]) for m in present},
        "copying": [
            copying_report(m, square_id, observed, columns[m], persistence) for m in present
        ],
        "failures": [
            window
            for m in present
            for window in worst_windows(m, square_id, observed, columns[m], persistence, times)
        ],
    }


def run_results(config: Config, *, split_name: str = "test") -> dict[str, Any]:
    """Produce the Phase 6 figures and tables for one split."""
    from src import plots

    areas = SelectedAreas.load(config.paths.tables / "selected_areas.json")
    figures: list[Path] = []
    copying_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    daytype_rows: list[dict[str, Any]] = []
    hourly_rows: list[dict[str, Any]] = []

    for square_id in areas.forecast:
        data = _area_diagnostics(config, square_id, split_name)
        print(f"\n=== square {square_id}, {split_name} ===")

        # 1. Actual vs predicted, one figure per model, with a zoomed day.
        for model in PLOTTED:
            if model not in data["columns"]:
                continue
            figures.append(
                _save(
                    plots.plot_forecast(
                        data["times"],
                        data["observed"],
                        data["columns"][model],
                        model=model,
                        square_id=square_id,
                        persistence=data["persistence"],
                        daily_period=config.dataset.daily_period,
                    ),
                    config.paths.figures / f"forecast_{split_name}_{square_id}_{model}.png",
                )
            )

        # 4. Diagnostics.
        figures.append(
            _save(
                plots.plot_error_by_hour(data["by_hour"], square_id=square_id),
                config.paths.figures / f"error_by_hour_{split_name}_{square_id}.png",
            )
        )
        for model in PLOTTED:
            if model not in data["heatmaps"]:
                continue
            figures.append(
                _save(
                    plots.plot_error_heatmap(
                        data["heatmaps"][model], model=model, square_id=square_id
                    ),
                    config.paths.figures / f"error_heatmap_{split_name}_{square_id}_{model}.png",
                )
            )
        figures.append(
            _save(
                plots.plot_residual_acf(
                    {m: data["acfs"][m] for m in PLOTTED if m in data["acfs"]},
                    square_id=square_id,
                    daily_period=config.dataset.daily_period,
                ),
                config.paths.figures / f"residual_acf_{split_name}_{square_id}.png",
            )
        )
        figures.append(
            _save(
                plots.plot_cross_correlation(data["correlations"], square_id=square_id),
                config.paths.figures / f"cross_correlation_{split_name}_{square_id}.png",
            )
        )

        copying_rows.extend(report.to_row() for report in data["copying"])
        failure_rows.extend(window.to_row() for window in data["failures"])
        for model, summary in data["daytype"].items():
            daytype_rows.append({"square_id": square_id, "model": model, **summary})
        for model, by_hour in data["by_hour"].items():
            for hour, mae, n in zip(by_hour["hours"], by_hour["mae"], by_hour["n"], strict=True):
                hourly_rows.append(
                    {
                        "square_id": square_id,
                        "model": model,
                        "hour": int(hour),
                        "mae": round(float(mae), 4),
                        "n": int(n),
                    }
                )

        for report in data["copying"]:
            print(
                f"  {report.model:<16} peak lag {report.peak_lag:+d}  "
                f"copy ratio {report.copy_ratio:5.2f}  {report.verdict}"
            )

    # 3. Cross-area MASE, read from the metric tables the final runs wrote.
    metrics_path = config.paths.tables / f"final_metrics_all_{split_name}.csv"
    cross_area: list[dict[str, Any]] = []
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8", newline="") as fh:
            cross_area = cross_area_table(list(csv.DictReader(fh)))
        figures.append(
            _save(
                plots.plot_cross_area_mase(cross_area, areas=list(areas.forecast)),
                config.paths.figures / f"cross_area_mase_{split_name}.png",
            )
        )

    tables = [
        _write_csv(copying_rows, config.paths.tables / f"copying_{split_name}.csv"),
        _write_csv(failure_rows, config.paths.tables / f"failure_windows_{split_name}.csv"),
        _write_csv(daytype_rows, config.paths.tables / f"error_by_daytype_{split_name}.csv"),
        _write_csv(hourly_rows, config.paths.tables / f"error_by_hour_{split_name}.csv"),
        _write_csv(cross_area, config.paths.tables / f"cross_area_mase_{split_name}.csv"),
    ]

    summary = {
        "split": split_name,
        "areas": list(areas.forecast),
        # Relative to the project root, and with forward slashes: an absolute
        # path here bakes one machine's directory layout into a committed
        # artefact, so the file could never match on any other machine.
        "figures": [_relative(p) for p in figures],
        "tables": [_relative(p) for p in tables],
        "copying_verdicts": {r["model"]: r["verdict"] for r in copying_rows},
    }
    path = config.paths.tables / f"results_summary_{split_name}.json"
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"\nwrote {len(figures)} figures and {len(tables)} tables for the {split_name} split")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="Path to a config YAML.")
    parser.add_argument(
        "--split",
        default="test",
        choices=("test", "stress"),
        help="Split to analyse. Reported results use the test week.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Produce the evaluation figures and diagnostic tables."""
    import matplotlib

    matplotlib.use("Agg")

    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    config.paths.mkdirs()
    run_results(config, split_name=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
