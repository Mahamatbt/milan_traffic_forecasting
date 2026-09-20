"""The single source of named, artefact-derived figures.

Numbers quoted in prose went wrong repeatedly, and always the same way: a figure
measured early was quoted as though general, then never revisited when better
evidence arrived. Guarding against the specific values that went wrong is
reactive; this module addresses the mechanism.

Every figure the report might cite is computed here, once, from the artefacts,
and given a name. Prose then quotes a named fact rather than a number remembered
from a terminal, and a change in the underlying measurement propagates by
regeneration instead of by recollection.

Naming is the part that does the real work. One of the errors this replaces was
"up to 246 rows per cell", where 246 is the count of distinct country codes in a
file and the maximum rows for one cell is 36. Both are real measurements, so no
value check could catch the confusion -- but writing it down forces a choice
between ``distinct_country_codes_per_file`` and ``max_rows_per_cell``, and the
distinction becomes impossible to miss.

Run ``python -m src.facts`` to regenerate ``report/FACTS.json`` and
``report/FACTS.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import Config, load_config

__all__ = ["Fact", "collect_facts", "write_facts", "load_facts", "DATASET_TOTAL_BYTES"]

# Total size of the published dataset, summed from the Dataverse file listing for
# doi:10.7910/DVN/EGZHFV. It is a property of an immutable, DOI-versioned
# release, not of what happens to be downloaded here -- data/raw/manifest.json
# records only the local subset, and using it would understate the dataset by
# however many files were skipped.
DATASET_TOTAL_BYTES = 20_804_803_507


@dataclass(frozen=True)
class Fact:
    """One named figure, with enough context to quote it correctly."""

    key: str
    value: float | int | str
    unit: str
    source: str
    description: str
    display: str = ""

    def rendered(self) -> str:
        """How the figure should appear in prose."""
        if self.display:
            return self.display
        if isinstance(self.value, float):
            return f"{self.value:,.2f}"
        if isinstance(self.value, int):
            return f"{self.value:,}"
        return str(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "display": self.rendered(),
            "unit": self.unit,
            "source": self.source,
            "description": self.description,
        }


@dataclass
class FactSet:
    """Every fact, grouped by the report section it belongs to."""

    groups: dict[str, list[Fact]] = field(default_factory=dict)

    def add(self, group: str, fact: Fact) -> None:
        self.groups.setdefault(group, []).append(fact)

    def flat(self) -> dict[str, Fact]:
        return {f.key: f for facts in self.groups.values() for f in facts}

    def to_dict(self) -> dict[str, Any]:
        return {key: fact.to_dict() for key, fact in sorted(self.flat().items())}


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------


def _json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------


def _dataset_total_bytes(config: Config) -> int:
    """Total size of the dataset, not of whatever is on this machine.

    A local manifest is used only when it records every expected day; a partial
    download would otherwise be reported as the dataset size. This is the bug
    the registry caught on its first run: the manifest held four days, and the
    derived download rate came out at 21 Mbit/s instead of 469.
    """
    manifest = _json(config.paths.raw / "manifest.json")
    if manifest and manifest.get("n_files", 0) >= config.dataset.n_days:
        return int(manifest["total_bytes"])
    return DATASET_TOTAL_BYTES


def _dataset_facts(facts: FactSet, config: Config, tables: Path) -> None:
    """Size and shape of the raw dataset and the assembled matrix."""
    rows, cols = config.dataset.matrix_shape
    cells = rows * cols
    matrix_bytes = cells * 4

    facts.add(
        "Dataset",
        Fact("n_days", config.dataset.n_days, "days", "config", "Observation period length"),
    )
    facts.add("Dataset", Fact("n_squares", cols, "cells", "config", "Grid cells in Milan"))
    facts.add(
        "Dataset",
        Fact("n_timestamps", rows, "intervals", "config", "10-minute intervals in the period"),
    )
    facts.add(
        "Dataset", Fact("grid_cells_total", cells, "cells", "derived", "Timestamps x squares")
    )
    facts.add(
        "Dataset",
        Fact(
            "matrix_mib",
            matrix_bytes / 1024**2,
            "MiB",
            "derived",
            "Assembled matrix as float32",
            f"{matrix_bytes / 1024**2:.2f}",
        ),
    )

    total = _dataset_total_bytes(config)
    facts.add(
        "Dataset",
        Fact("raw_bytes", total, "bytes", "Dataverse file listing", "Total published dataset size"),
    )
    facts.add(
        "Dataset",
        Fact(
            "raw_gib",
            total / 1024**3,
            "GiB",
            "Dataverse file listing",
            "Total published dataset size",
            f"{total / 1024**3:.2f}",
        ),
    )

    summary = _json(tables / "ingest_summary.json")
    if not summary:
        return

    raw_rows = summary["total_raw_rows"]
    facts.add(
        "Dataset",
        Fact(
            "total_raw_rows",
            raw_rows,
            "rows",
            "ingest_summary.json",
            "Records parsed across all days",
        ),
    )
    facts.add(
        "Dataset",
        Fact(
            "mean_rows_per_day",
            raw_rows / config.dataset.n_days,
            "rows",
            "derived",
            "Mean records per day over the period",
            f"{raw_rows / config.dataset.n_days / 1e6:.2f} M",
        ),
    )

    matrix = _json(Path("data/processed/matrix_report.json"))
    if matrix:
        absent = matrix["total_absent_cells"]
        occupied = cells - absent
        facts.add(
            "Dataset",
            Fact(
                "absent_cells", absent, "cells", "matrix_report.json", "Cells with no record at all"
            ),
        )
        facts.add(
            "Dataset",
            Fact(
                "absent_cells_pct",
                100 * absent / cells,
                "%",
                "derived",
                "Absent cells as a share of the grid",
                f"{100 * absent / cells:.4f}",
            ),
        )
        facts.add(
            "Dataset",
            Fact(
                "mean_rows_per_cell",
                raw_rows / occupied,
                "rows",
                "derived",
                "Mean country-split records per occupied cell",
                f"{raw_rows / occupied:.1f}",
            ),
        )
        facts.add(
            "Dataset",
            Fact(
                "missing_intervals",
                len(matrix["missing_intervals"]),
                "intervals",
                "matrix_report.json",
                "Timestamps absent from the grid",
            ),
        )
        facts.add(
            "Dataset",
            Fact(
                "nan_after_policy",
                matrix["n_nan_after_policy"],
                "cells",
                "matrix_report.json",
                "NaN remaining after the missing-data policy",
            ),
        )
        facts.add(
            "Dataset",
            Fact(
                "total_internet",
                matrix["total_internet"],
                "activity",
                "matrix_report.json",
                "Total internet activity over the period",
                f"{matrix['total_internet']:,.2f}",
            ),
        )

    # Measured directly from a raw file; distinct from the count of country codes.
    facts.add(
        "Dataset",
        Fact(
            "max_rows_per_cell",
            36,
            "rows",
            "measured on 2013-11-01",
            "Most country-split records for a single (square, time) pair",
        ),
    )
    facts.add(
        "Dataset",
        Fact(
            "distinct_country_codes_per_file",
            246,
            "codes",
            "measured on 2013-11-01",
            "Distinct country codes in one daily file -- NOT rows per cell",
        ),
    )


def _timing_facts(facts: FactSet, tables: Path) -> None:
    """Download and ingest throughput."""
    summary = _json(tables / "ingest_summary.json")
    if not summary:
        return

    for key, unit, desc in (
        ("download_total_min", "min", "Total download time"),
        ("download_mean_s", "s", "Mean download time per day"),
        ("download_std_s", "s", "Standard deviation of download time per day"),
        ("ingest_total_min", "min", "Total ingest time"),
        ("ingest_mean_s", "s", "Mean ingest time per day"),
        ("ingest_std_s", "s", "Standard deviation of ingest time per day"),
        ("peak_rss_max_mib", "MiB", "Peak RSS during the streaming run"),
    ):
        facts.add("Timing", Fact(key, summary[key], unit, "ingest_summary.json", desc))

    total_bytes = DATASET_TOTAL_BYTES
    seconds = summary["download_total_min"] * 60
    facts.add(
        "Timing",
        Fact(
            "download_mib_per_s",
            total_bytes / 1024**2 / seconds,
            "MiB/s",
            "derived",
            "Sustained download throughput",
            f"{total_bytes / 1024**2 / seconds:.1f}",
        ),
    )
    facts.add(
        "Timing",
        Fact(
            "download_mbit_per_s",
            total_bytes * 8 / seconds / 1e6,
            "Mbit/s",
            "derived",
            "Sustained throughput in decimal megabits, not mebibits",
            f"{total_bytes * 8 / seconds / 1e6:.0f}",
        ),
    )


def _memory_facts(facts: FactSet, config: Config, tables: Path) -> None:
    """Peak memory by ingest strategy, from the isolated benchmark."""
    rows = _csv(tables / "memory_report.csv")
    if not rows:
        return

    by_strategy: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_strategy.setdefault(row["strategy"], []).append(row)

    for strategy, group in by_strategy.items():
        peaks = [int(r["per_day_peak_rss_bytes"]) / 1024**2 for r in group]
        walls = [float(r["per_day_wall_s"]) for r in group]
        mean_peak = statistics.fmean(peaks)
        facts.add(
            "Memory",
            Fact(
                f"{strategy}_peak_mib",
                mean_peak,
                "MiB",
                "memory_report.csv",
                f"Mean peak RSS for {strategy}, one process per measurement",
                f"{mean_peak:.2f}",
            ),
        )
        facts.add(
            "Memory",
            Fact(
                f"{strategy}_wall_s",
                statistics.fmean(walls),
                "s",
                "memory_report.csv",
                f"Mean wall time per day for {strategy}",
                f"{statistics.fmean(walls):.2f}",
            ),
        )
        resident = int(group[0]["per_day_inmem_bytes"]) / 1024**2
        facts.add(
            "Memory",
            Fact(
                f"{strategy}_resident_mib",
                resident,
                "MiB",
                "memory_report.csv",
                f"Data resident after {strategy} completes",
                f"{resident:.2f}",
            ),
        )

    naive = next((r for r in rows if r["strategy"] == "naive_pandas"), None)
    if naive and naive.get("projected_full_ram_bytes"):
        projected = int(naive["projected_full_ram_bytes"]) / 1024**3
        facts.add(
            "Memory",
            Fact(
                "naive_projected_gib",
                projected,
                "GiB",
                "derived, extrapolated",
                f"Holding all {config.dataset.n_days} days the naive way; not executed",
                f"{projected:.2f}",
            ),
        )


def _eda_facts(facts: FactSet, tables: Path) -> None:
    """Distribution, areas and time-series characterisation."""
    dist = _json(tables / "distribution_stats.json")
    if dist:
        for key, unit, desc, fmt in (
            ("gini", "", "Gini coefficient of cell totals", "{:.3f}"),
            ("skewness", "", "Skewness of cell totals", "{:.2f}"),
            ("kurtosis", "", "Excess kurtosis of cell totals", "{:.2f}"),
            ("max_over_median", "x", "Busiest cell relative to the median", "{:.1f}"),
            ("lognormal_mu", "", "Fitted lognormal mu", "{:.3f}"),
            ("lognormal_sigma", "", "Fitted lognormal sigma", "{:.3f}"),
            ("lognormal_ks_stat", "", "KS statistic against the lognormal fit", "{:.4f}"),
        ):
            facts.add(
                "Distribution",
                Fact(key, dist[key], unit, "distribution_stats.json", desc, fmt.format(dist[key])),
            )
        for key in ("top_1pct_share", "top_5pct_share", "top_10pct_share", "bottom_50pct_share"):
            facts.add(
                "Distribution",
                Fact(
                    key,
                    dist[key],
                    "%",
                    "distribution_stats.json",
                    key.replace("_", " "),
                    f"{dist[key]:.1%}",
                ),
            )
        facts.add(
            "Distribution",
            Fact(
                "n_zero_cells",
                dist["n_zero_cells"],
                "cells",
                "distribution_stats.json",
                "Cells with no activity at all",
            ),
        )

    areas = _json(tables / "selected_areas.json")
    if areas:
        facts.add(
            "Areas",
            Fact(
                "top_traffic_square",
                areas["top_traffic"],
                "",
                "selected_areas.json",
                "Highest-traffic cell",
            ),
        )
        facts.add(
            "Areas",
            Fact(
                "top_three_squares",
                ", ".join(map(str, areas["top_three"])),
                "",
                "selected_areas.json",
                "Three highest-traffic cells",
            ),
        )
        facts.add(
            "Areas",
            Fact(
                "forecast_squares",
                ", ".join(map(str, areas["forecast"])),
                "",
                "selected_areas.json",
                "Cells modelled in Section 4",
            ),
        )
        for square, rank in areas["ranks"].items():
            facts.add(
                "Areas",
                Fact(
                    f"rank_{square}",
                    rank,
                    "",
                    "selected_areas.json",
                    f"Rank of square {square} by total activity",
                ),
            )

    decomposition = _json(tables / "decomposition.json")
    if decomposition:
        for name, share in decomposition["variance_share"].items():
            facts.add(
                "Decomposition",
                Fact(
                    f"variance_{name}",
                    share,
                    "%",
                    "decomposition.json",
                    f"Variance share of the {name} component",
                    f"{share:.1%}",
                ),
            )
        for name, strength in decomposition["strength"].items():
            facts.add(
                "Decomposition",
                Fact(
                    f"strength_{name}",
                    strength,
                    "",
                    "decomposition.json",
                    f"Strength of the {name} component",
                    f"{strength:.3f}",
                ),
            )
        ratio = decomposition["residual_std"] / decomposition["observed_std"]
        facts.add(
            "Decomposition",
            Fact(
                "residual_over_observed_std",
                ratio,
                "",
                "derived",
                "Residual standard deviation as a share of observed",
                f"{ratio:.1%}",
            ),
        )

    acf = _json(tables / "autocorrelation.json")
    if acf:
        for key, desc in (
            ("lag_1", "Lag-1 autocorrelation"),
            ("acf_at_daily", "Autocorrelation at the daily lag"),
            ("acf_at_weekly", "Autocorrelation at the weekly lag"),
        ):
            facts.add(
                "Autocorrelation",
                Fact(key, acf[key], "", "autocorrelation.json", desc, f"{acf[key]:.3f}"),
            )
        facts.add(
            "Autocorrelation",
            Fact(
                "notable_lags",
                ", ".join(map(str, acf["notable_lags"][:6])),
                "",
                "autocorrelation.json",
                "Local ACF maxima",
            ),
        )

    peaks = _csv(tables / "spectral_peaks.csv")
    if peaks:
        for index, peak in enumerate(peaks[:3], start=1):
            hours = float(peak["period_hours"])
            facts.add(
                "Spectrum",
                Fact(
                    f"spectral_peak_{index}_hours",
                    hours,
                    "h",
                    "spectral_peaks.csv",
                    f"Dominant cycle {index}",
                    f"{hours:.2f}",
                ),
            )

    anomalies = _csv(tables / "anomalies.csv")
    if anomalies:
        evaluable = 8928 - 144
        on_holiday = sum(1 for r in anomalies if r["is_holiday"] == "True")
        facts.add(
            "Anomalies",
            Fact(
                "anomalies_flagged",
                len(anomalies),
                "intervals",
                "anomalies.csv",
                "Intervals flagged against a seasonal-naive predictor",
            ),
        )
        facts.add(
            "Anomalies",
            Fact(
                "anomalies_evaluable",
                evaluable,
                "intervals",
                "derived",
                "Intervals the detector can assess; the first day cannot be",
            ),
        )
        facts.add(
            "Anomalies",
            Fact(
                "anomalies_rate",
                len(anomalies) / evaluable,
                "%",
                "derived",
                "Flagged share of evaluable intervals",
                f"{len(anomalies) / evaluable:.2%}",
            ),
        )
        facts.add(
            "Anomalies",
            Fact(
                "anomalies_on_holidays",
                on_holiday,
                "intervals",
                "anomalies.csv",
                "Flagged intervals falling on a holiday",
            ),
        )
        facts.add(
            "Anomalies",
            Fact(
                "anomalies_holiday_share",
                on_holiday / len(anomalies),
                "%",
                "derived",
                "Share of flags on holidays",
                f"{on_holiday / len(anomalies):.1%}",
            ),
        )

    summary_rows = _csv(tables / "area_summary.csv")
    for row in summary_rows:
        square = row["square_id"]
        if row.get("nearest"):
            facts.add(
                "Areas",
                Fact(
                    f"nearest_{square}",
                    row["nearest"],
                    "",
                    "area_summary.csv",
                    f"Nearest reference point to square {square}",
                ),
            )
            facts.add(
                "Areas",
                Fact(
                    f"nearest_m_{square}",
                    int(row["nearest_m"]),
                    "m",
                    "area_summary.csv",
                    f"Distance from square {square}",
                ),
            )
        for column in ("cv", "peak_to_trough", "night_floor_over_mean", "weekend_over_weekday"):
            if column in row:
                value = float(row[column])
                facts.add(
                    "Areas",
                    Fact(
                        f"{column}_{square}",
                        value,
                        "",
                        "area_summary.csv",
                        f"{column.replace('_', ' ')} for square {square}",
                        f"{value:.3f}",
                    ),
                )


def _model_facts(facts: FactSet, tables: Path) -> None:
    """Named figures from tuning, the final fits and the diagnostics.

    Naming each one is what forces the distinctions that are easy to blur in
    prose: the LSTM's error and its ensemble's error are different quantities,
    a training time on a GPU is not comparable to one on a CPU, and a model that
    wins on validation need not win on test.
    """
    selected = _json(tables / "selected_hyperparameters.json")
    if selected:
        harmonic = selected.get("harmonic_arima", {})
        gbm = selected.get("lightgbm", {})
        lstm = selected.get("lstm", {})
        facts.add(
            "Models",
            Fact(
                "tuning_area",
                selected.get("tuning_area", 0),
                "square id",
                "selected_hyperparameters.json",
                "Area hyperparameters were selected on; the other two reuse them",
            ),
        )
        if harmonic:
            facts.add(
                "Models",
                Fact(
                    "harmonic_order",
                    str(tuple(harmonic.get("order", []))),
                    "",
                    "selected_hyperparameters.json",
                    "ARIMA(p,d,q) chosen on validation MAE with harmonics fixed",
                ),
            )
            facts.add(
                "Models",
                Fact(
                    "harmonic_fourier_orders",
                    f"K1={harmonic.get('k_daily')}, K2={harmonic.get('k_weekly')}",
                    "",
                    "selected_hyperparameters.json",
                    "Daily and weekly Fourier orders selected by AICc",
                ),
            )
        if gbm:
            facts.add(
                "Models",
                Fact(
                    "lightgbm_trees",
                    int(gbm.get("best_iteration", 0)),
                    "trees",
                    "selected_hyperparameters.json",
                    "Trees early stopping chose, against the n_estimators ceiling",
                ),
            )
        if lstm:
            facts.add(
                "Models",
                Fact(
                    "lstm_sequence_length",
                    int(lstm.get("sequence_length", 0)),
                    "steps",
                    "selected_hyperparameters.json",
                    "Window length the staged sweep selected",
                ),
            )
            facts.add(
                "Models",
                Fact(
                    "lstm_best_epoch",
                    int(lstm.get("best_epoch", -1)),
                    "epochs",
                    "selected_hyperparameters.json",
                    "Epoch that was best on validation; the final fit trains this many",
                ),
            )

    # The experiment log lives beside the tables directory, not inside it.
    experiments = tables.parent / "experiments.csv"
    if experiments.exists():
        rows = _csv(experiments)
        facts.add(
            "Models",
            Fact(
                "experiments_logged",
                len(rows),
                "runs",
                "experiments.csv",
                "Candidates logged across all three models, each with a rationale",
            ),
        )

    # --- Test-week accuracy -------------------------------------------------
    combined = tables / "final_metrics_all_test.csv"
    if combined.exists():
        rows = _csv(combined)
        by_model: dict[str, list[float]] = {}
        for row in rows:
            by_model.setdefault(row["model"], []).append(float(row["mase"]))

        for model, values in by_model.items():
            facts.add(
                "Results",
                Fact(
                    f"mase_mean_{model}",
                    sum(values) / len(values),
                    "MASE",
                    "final_metrics_all_test.csv",
                    f"Mean MASE for {model} across the three areas, test week",
                    display=f"{sum(values) / len(values):.3f}",
                ),
            )

        persistence = sum(by_model.get("persistence", [1.0])) / max(
            len(by_model.get("persistence", [1.0])), 1
        )
        ranked = sorted((sum(v) / len(v), m) for m, v in by_model.items())
        best_mase, best_model = ranked[0]
        facts.add(
            "Results",
            Fact(
                "best_model_test",
                best_model,
                "",
                "final_metrics_all_test.csv",
                "Model with the lowest mean MASE across areas on the test week",
            ),
        )
        facts.add(
            "Results",
            Fact(
                "best_model_gain_over_persistence",
                (persistence - best_mase) / persistence * 100,
                "%",
                "final_metrics_all_test.csv",
                "How much the best model improves on persistence, mean MASE",
                display=f"{(persistence - best_mase) / persistence * 100:.1f}%",
            ),
        )
        beat = [m for m, v in by_model.items() if sum(v) / len(v) < persistence]
        facts.add(
            "Results",
            Fact(
                "models_beating_persistence",
                len([m for m in beat if m != "persistence"]),
                "models",
                "final_metrics_all_test.csv",
                "Models whose mean MASE beats persistence on the test week",
            ),
        )

    # --- Holiday degradation ------------------------------------------------
    stress = tables / "final_metrics_all_stress.csv"
    if stress.exists() and combined.exists():
        stress_rows = _csv(stress)
        by_area_model = {(r["square_id"], r["model"]): float(r["mase"]) for r in stress_rows}
        persistence_by_area = {
            r["square_id"]: float(r["mase"]) for r in stress_rows if r["model"] == "persistence"
        }
        for model in ("harmonic_arima", "lightgbm", "lstm"):
            ratios = [
                by_area_model[(area, model)] / base
                for area, base in persistence_by_area.items()
                if (area, model) in by_area_model
            ]
            if not ratios:
                continue
            worst = max(ratios)
            facts.add(
                "Failure analysis",
                Fact(
                    f"stress_worst_ratio_{model}",
                    worst,
                    "x persistence",
                    "final_metrics_all_stress.csv",
                    f"Worst per-area MASE for {model} on the holiday split, "
                    "relative to persistence on the same area",
                    display=f"{worst:.2f}x",
                ),
            )
        wins = sum(
            1
            for area, base in persistence_by_area.items()
            if all(
                by_area_model.get((area, m), float("inf")) >= base
                for m in ("harmonic_arima", "lightgbm", "lstm")
            )
        )
        facts.add(
            "Failure analysis",
            Fact(
                "stress_areas_persistence_wins",
                wins,
                "areas",
                "final_metrics_all_stress.csv",
                "Areas on the holiday split where no model beats persistence",
            ),
        )

    # --- Copying ------------------------------------------------------------
    copying = tables / "copying_test.csv"
    if copying.exists():
        rows = [r for r in _csv(copying) if r["model"] not in {"persistence", "seasonal_naive"}]
        if rows:
            ratios = [float(r["copy_ratio"]) for r in rows]
            facts.add(
                "Diagnostics",
                Fact(
                    "copy_ratio_min",
                    min(ratios),
                    "",
                    "copying_test.csv",
                    "Closest any model comes to persistence; 0.00 would be a collapse",
                    display=f"{min(ratios):.2f}",
                ),
            )
            facts.add(
                "Diagnostics",
                Fact(
                    "models_collapsed_to_persistence",
                    sum(1 for r in rows if r["verdict"] == "collapsed"),
                    "models",
                    "copying_test.csv",
                    "Models judged to have collapsed to repeating their last input",
                ),
            )

    # --- Cost ---------------------------------------------------------------
    timing = tables / "timing_test.csv"
    if timing.exists():
        rows = [r for r in _csv(timing) if r["square_id"] == "5161"]
        per_step = {r["model"]: float(r["inference_ms_per_step"]) for r in rows}
        if per_step.get("harmonic_arima") and per_step.get("lstm"):
            ratio = per_step["harmonic_arima"] / per_step["lstm"]
            facts.add(
                "Cost",
                Fact(
                    "inference_ratio_harmonic_over_lstm",
                    ratio,
                    "x",
                    "timing_test.csv",
                    "Harmonic ARIMA inference cost per step relative to the LSTM; "
                    "the cheapest model to fit is the most expensive to serve",
                    display=f"{ratio:,.0f}x",
                ),
            )
        for model, value in per_step.items():
            if model in {"persistence", "seasonal_naive"}:
                continue
            facts.add(
                "Cost",
                Fact(
                    f"inference_ms_per_step_{model}",
                    value,
                    "ms",
                    "timing_test.csv",
                    f"Wall-clock milliseconds per one-step forecast for {model}, square 5161",
                    display=f"{value:.3f} ms",
                ),
            )


def collect_facts(config: Config) -> FactSet:
    """Compute every named figure from the artefacts on disk."""
    tables = config.paths.tables
    facts = FactSet()
    _dataset_facts(facts, config, tables)
    _timing_facts(facts, tables)
    _memory_facts(facts, config, tables)
    _eda_facts(facts, tables)
    _model_facts(facts, tables)
    return facts


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def write_facts(facts: FactSet, report_dir: Path) -> tuple[Path, Path]:
    """Write the machine-readable registry and its human-readable index."""
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "FACTS.json"
    json_path.write_text(json.dumps(facts.to_dict(), indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Fact registry",
        "",
        "Every figure the report may cite, computed from the artefacts and named.",
        "Quote a fact by its key so that a change in measurement propagates by",
        "regeneration rather than by recollection.",
        "",
        "Regenerate with `python -m src.facts`.",
        "",
    ]
    for group, entries in facts.groups.items():
        lines += [
            f"## {group}",
            "",
            "| key | value | unit | source | description |",
            "|---|---|---|---|---|",
        ]
        for fact in sorted(entries, key=lambda f: f.key):
            lines.append(
                f"| `{fact.key}` | {fact.rendered()} | {fact.unit} | `{fact.source}` | {fact.description} |"
            )
        lines.append("")

    md_path = report_dir / "FACTS.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def load_facts(report_dir: Path) -> dict[str, dict[str, Any]]:
    """Read the registry back, for tests and for rendering prose."""
    path = report_dir / "FACTS.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run python -m src.facts")
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="Path to a config YAML.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Regenerate the fact registry from the artefacts."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    facts = collect_facts(config)
    report_dir = config.paths.results.parent / "report"
    json_path, md_path = write_facts(facts, report_dir)

    total = len(facts.flat())
    print(f"{total} facts across {len(facts.groups)} groups")
    for group, entries in facts.groups.items():
        print(f"  {group:<16} {len(entries):>3}")
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
