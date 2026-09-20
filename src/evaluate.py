"""Evaluate forecasters on the test week and write the per-area results.

Reports every model on every study area, with the persistence and seasonal-naive
baselines in the same table. A model's error means little on its own: on this
data the lag-1 autocorrelation is 0.987, so repeating the last observation is
already a strong forecast, and a figure that looks respectable in isolation may
be worse than doing nothing.

All predictions come from ``walk_forward``: at each step the model sees the true
observed history and predicts one step ahead. Metrics are computed in original
units after inverse-transforming.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Config, load_config
from src.eda import SelectedAreas
from src.metrics import Metrics, evaluate
from src.models.base import Forecaster
from src.models.baselines import Persistence, SeasonalNaive
from src.splits import SplitSet, make_splits

__all__ = ["AreaResult", "evaluate_area", "evaluate_all", "load_area_series"]


@dataclass
class AreaResult:
    """Every model's score on one area and one split."""

    square_id: int
    split: str
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, model: str, metrics: Metrics, wall_s: float, n_params: int) -> None:
        self.rows.append(
            {
                "square_id": self.square_id,
                "split": self.split,
                "model": model,
                **metrics.to_row(),
                "inference_wall_s": round(wall_s, 4),
                "inference_ms_per_step": round(1000 * wall_s / max(metrics.n, 1), 4),
                "n_params": n_params,
            }
        )

    def table(self) -> str:
        """Fixed-width rendering for a terminal or a notebook."""
        header = (
            f"{'model':<16}{'MAE':>10}{'RMSE':>10}{'MAPE %':>9}"
            f"{'sMAPE %':>9}{'MASE':>8}{'R2':>8}"
        )
        lines = [header, "-" * len(header)]
        for row in sorted(self.rows, key=lambda r: r["mase"]):
            lines.append(
                f"{row['model']:<16}{row['mae']:>10.2f}{row['rmse']:>10.2f}"
                f"{row['mape']:>9.2f}{row['smape']:>9.2f}"
                f"{row['mase']:>8.3f}{row['r2']:>8.4f}"
            )
        return "\n".join(lines)


def load_area_series(config: Config, square_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Load one area's series and timestamps from the committed extract.

    Reads ``selected_series.parquet`` rather than the full matrix, so the
    modelling stages run from a clean checkout without the raw download.
    """
    import polars as pl

    path = config.paths.processed / "selected_series.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python run.py eda` to extract the study series."
        )
    frame = pl.read_parquet(path)
    column = f"square_{square_id}"
    if column not in frame.columns:
        raise KeyError(
            f"{column} is not in {path.name}; available: "
            f"{[c for c in frame.columns if c.startswith('square_')]}"
        )
    return frame[column].to_numpy().astype(np.float64), frame["local"].to_numpy()


def _baselines(config: Config) -> list[Forecaster]:
    """The reference floor, present in every table."""
    return [Persistence(), SeasonalNaive(seasonal_period=config.dataset.daily_period)]


def evaluate_area(
    square_id: int,
    models: list[Forecaster],
    splits: SplitSet,
    config: Config,
    *,
    split_name: str = "test",
) -> AreaResult:
    """Score every model on one area over one split.

    Args:
        square_id: The area being forecast.
        models: Fitted forecasters. Baselines are added automatically.
        splits: The area's split set.
        config: Study configuration.
        split_name: Which split to evaluate on.

    Returns:
        An :class:`AreaResult` holding one row per model.
    """
    split = splits[split_name]
    result = AreaResult(square_id=square_id, split=split_name)

    # MASE is scaled by the in-sample seasonal-naive error on the training data
    # only, never on the window being scored.
    train_values = splits.train.values

    for model in [*_baselines(config), *models]:
        started = time.perf_counter()
        predictions = model.walk_forward(splits.series, split.start, split.stop, times=splits.times)
        wall = time.perf_counter() - started

        metrics = evaluate(
            split.values,
            predictions,
            y_train=train_values,
            seasonal_period=config.evaluation.seasonal_period,
            mape_threshold=config.evaluation.mape_zero_threshold,
        )
        result.add(model.name, metrics, wall, model.n_params())

    return result


def evaluate_all(
    config: Config,
    *,
    models_by_area: dict[int, list[Forecaster]] | None = None,
    split_name: str = "test",
) -> list[AreaResult]:
    """Evaluate every configured forecast area and persist the tables.

    Args:
        config: Study configuration.
        models_by_area: Fitted models per square. Omit to report baselines only,
            which is what establishes the bar before any model is built.
        split_name: Split to evaluate on.

    Returns:
        One :class:`AreaResult` per area.
    """
    areas_path = config.paths.tables / "selected_areas.json"
    if not areas_path.exists():
        raise FileNotFoundError(
            f"{areas_path} not found. Run `python run.py eda` to resolve the study areas."
        )
    areas = SelectedAreas.load(areas_path)

    results: list[AreaResult] = []
    for square_id in areas.forecast:
        values, times = load_area_series(config, square_id)
        splits = make_splits(values, times, config)
        models = (models_by_area or {}).get(square_id, [])
        result = evaluate_area(square_id, models, splits, config, split_name=split_name)
        results.append(result)

        rank = areas.ranks[str(square_id)]
        print(f"\n=== square {square_id} (rank {rank}/10000), {split_name} split ===")
        print(result.table())

    _write_results(results, config, split_name)
    return results


def _write_results(results: list[AreaResult], config: Config, split_name: str) -> list[Path]:
    """Write one CSV per area plus a combined table."""
    written: list[Path] = []
    combined: list[dict[str, Any]] = []

    for result in results:
        combined.extend(result.rows)
        path = config.paths.tables / f"metrics_area_{result.square_id}_{split_name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(result.rows[0]))
            writer.writeheader()
            writer.writerows(result.rows)
        written.append(path)

    if combined:
        path = config.paths.tables / f"metrics_all_areas_{split_name}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(combined[0]))
            writer.writeheader()
            writer.writerows(combined)
        written.append(path)

    return written


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="Path to a config YAML.")
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "validation", "test", "stress"),
        help="Split to evaluate on. Reported results use the test week.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate the baselines, and any models a later phase supplies."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    config.paths.mkdirs()

    results = evaluate_all(config, split_name=args.split)

    summary = {
        "split": args.split,
        "areas": [r.square_id for r in results],
        "models": sorted({row["model"] for r in results for row in r.rows}),
    }
    path = config.paths.tables / f"metrics_summary_{args.split}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote per-area tables and {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
