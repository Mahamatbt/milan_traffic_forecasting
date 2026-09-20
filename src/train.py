"""Staged hyperparameter search, with the reasoning recorded as it happens.

Tuning proceeds one axis at a time on the highest-traffic area only, selecting on
validation MAE. Searching every axis jointly would be more thorough and far less
interpretable: with a single joint search there is nothing to say about *why* a
setting was chosen beyond that the optimiser preferred it. Staging makes each
decision attributable, which is what the experimentation criterion asks for.

Every run appends a row to ``results/experiments.csv`` with a rationale
generated from the comparison it just completed -- what won, by how much, and
what that implies for the next stage. The rationale is derived from the numbers
rather than written in advance, so a stage that produces no improvement says so.

Model-specific selection differs where the model demands it. Harmonic order is
chosen by AICc rather than by validation error, because refitting and
walk-forwarding 28 candidates would cost far more than an information criterion
computed in-sample, and AICc is the standard criterion for exactly this choice.
LightGBM uses Optuna over its continuous parameters, since tree hyperparameters
interact too strongly for a staged sweep to be meaningful.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from typing import Any

import numpy as np

from src.config import Config, load_config
from src.eda import SelectedAreas
from src.evaluate import load_area_series
from src.experiments import ExperimentLog, ExperimentRecord
from src.metrics import evaluate
from src.models.base import Forecaster
from src.models.gbm import GBMConfig, GBMForecaster
from src.models.harmonic_arima import HarmonicARIMA
from src.models.lstm import LSTMConfig, LSTMForecaster
from src.splits import SplitSet, make_splits

__all__ = ["tune_harmonic", "tune_lstm", "tune_gbm", "train_all"]

# Sweeps use a shorter budget than the final fits: the ranking between
# configurations settles well before convergence, and a full budget per
# candidate would multiply the search cost for no change in the decision.
SWEEP_EPOCHS = 30
SWEEP_PATIENCE = 5


def _score(model: Forecaster, splits: SplitSet, config: Config) -> Any:
    """Validation metrics for a fitted model, in original units."""
    predictions = model.walk_forward(
        splits.series, splits.validation.start, splits.validation.stop, times=splits.times
    )
    return evaluate(
        splits.validation.values,
        predictions,
        y_train=splits.train.values,
        seasonal_period=config.evaluation.seasonal_period,
        mape_threshold=config.evaluation.mape_zero_threshold,
    )


def _log(
    log: ExperimentLog,
    *,
    model: str,
    area: int,
    stage: str,
    hyperparams: dict[str, Any],
    feature_set: str,
    metrics: Any,
    wall: float,
    n_params: int,
    rationale: str,
    seed: int = 0,
) -> None:
    log.append(
        ExperimentRecord(
            model=model,
            area=area,
            stage=stage,
            seed=seed,
            hyperparams=hyperparams,
            feature_set=feature_set,
            valid_mae=float(metrics.mae),
            valid_rmse=float(metrics.rmse),
            valid_mase=float(metrics.mase),
            train_wall_s=round(wall, 3),
            n_params=n_params,
            rationale_for_next_change=rationale,
        )
    )


def _improvement(best: float, previous: float | None) -> str:
    """How much a stage gained, phrased for the rationale field."""
    if previous is None or not np.isfinite(previous):
        return "first stage, no prior baseline"
    delta = previous - best
    if abs(delta) < 1e-9:
        return "no change against the previous stage"
    return f"{'improved' if delta > 0 else 'worsened'} validation MAE by {abs(delta):.2f} ({abs(delta) / previous:.1%})"


# --------------------------------------------------------------------------
# Model 1: harmonic order by AICc
# --------------------------------------------------------------------------


def tune_harmonic(area: int, splits: SplitSet, config: Config, log: ExperimentLog) -> HarmonicARIMA:
    """Select Fourier orders by AICc, then score the winner on validation.

    The periodogram found a 12-hour component alongside the 24-hour one, so
    ``k_daily`` starts at 2; a single daily harmonic cannot represent a cycle
    with structure inside the day.
    """
    print(f"\n--- harmonic_arima, area {area}: Fourier order by AICc ---")
    candidates: list[tuple[float, int, int]] = []

    for k_daily in (2, 3, 4, 6, 8):
        for k_weekly in (1, 2, 3):
            model = HarmonicARIMA(
                daily_period=config.dataset.daily_period,
                weekly_period=config.dataset.weekly_period,
                k_daily=k_daily,
                k_weekly=k_weekly,
            )
            started = time.perf_counter()
            model.fit(splits.train.values)
            wall = time.perf_counter() - started
            candidates.append((model.aicc, k_daily, k_weekly))
            print(f"  K1={k_daily} K2={k_weekly}  AICc {model.aicc:>10.1f}  ({wall:.1f}s)")

    candidates.sort()
    best_aicc, k_daily, k_weekly = candidates[0]
    worst_aicc = candidates[-1][0]

    model = HarmonicARIMA(
        daily_period=config.dataset.daily_period,
        weekly_period=config.dataset.weekly_period,
        k_daily=k_daily,
        k_weekly=k_weekly,
    )
    started = time.perf_counter()
    model.fit(splits.train.values)
    wall = time.perf_counter() - started
    metrics = _score(model, splits, config)

    _log(
        log,
        model=model.name,
        area=area,
        stage="fourier_order",
        hyperparams=model.describe(),
        feature_set=f"fourier(K1={k_daily},K2={k_weekly})",
        metrics=metrics,
        wall=wall,
        n_params=model.n_params(),
        rationale=(
            f"AICc selected K1={k_daily}, K2={k_weekly} from {len(candidates)} "
            f"candidates, spanning {best_aicc:.0f} to {worst_aicc:.0f}. "
            f"K1>=2 was required by the 12-hour spectral peak. Validation MAE "
            f"{metrics.mae:.2f}, MASE {metrics.mase:.3f}. Next: search ARIMA(p,d,q) "
            f"around the selected harmonics, holding d=0 since ADF and KPSS agree "
            f"there is no stochastic trend."
        ),
    )

    print(f"  selected K1={k_daily} K2={k_weekly}; validation MAE {metrics.mae:.2f}")
    return _tune_harmonic_order(area, splits, config, log, k_daily, k_weekly, metrics.mae)


def _tune_harmonic_order(
    area: int,
    splits: SplitSet,
    config: Config,
    log: ExperimentLog,
    k_daily: int,
    k_weekly: int,
    previous_mae: float,
) -> HarmonicARIMA:
    """Search the ARIMA error structure with the harmonics held fixed."""
    print(f"--- harmonic_arima, area {area}: ARIMA order on validation MAE ---")
    best: tuple[float, HarmonicARIMA] | None = None

    for order in ((1, 0, 0), (2, 0, 0), (2, 0, 1), (3, 0, 1), (1, 0, 1)):
        model = HarmonicARIMA(
            daily_period=config.dataset.daily_period,
            weekly_period=config.dataset.weekly_period,
            k_daily=k_daily,
            k_weekly=k_weekly,
            order=order,
        )
        started = time.perf_counter()
        model.fit(splits.train.values)
        wall = time.perf_counter() - started
        metrics = _score(model, splits, config)
        print(f"  ARIMA{order}  MAE {metrics.mae:>8.2f}  MASE {metrics.mase:.3f}  ({wall:.1f}s)")

        _log(
            log,
            model=model.name,
            area=area,
            stage="arima_order",
            hyperparams=model.describe(),
            feature_set=f"fourier(K1={k_daily},K2={k_weekly})",
            metrics=metrics,
            wall=wall,
            n_params=model.n_params(),
            rationale=(
                f"ARIMA{order} with the AICc-selected harmonics gave validation MAE "
                f"{metrics.mae:.2f}, {_improvement(metrics.mae, previous_mae)}. "
                "d is held at 0 throughout because both stationarity tests agree the "
                "raw series has no unit root; differencing would remove signal."
            ),
        )
        if best is None or metrics.mae < best[0]:
            best = (metrics.mae, model)

    assert best is not None
    print(f"  selected ARIMA{best[1].order}; validation MAE {best[0]:.2f}")
    return best[1]


# --------------------------------------------------------------------------
# Model 2: staged LSTM sweeps
# --------------------------------------------------------------------------


def _fit_lstm(
    splits: SplitSet, config: Config, lstm_config: LSTMConfig, seed: int
) -> tuple[LSTMForecaster, Any, float]:
    model = LSTMForecaster(lstm_config, seed=seed)
    started = time.perf_counter()
    model.fit(
        splits.train.values,
        train_times=splits.train.times,
        valid=splits.validation.values,
        valid_times=splits.validation.times,
    )
    wall = time.perf_counter() - started
    return model, _score(model, splits, config), wall


def tune_lstm(area: int, splits: SplitSet, config: Config, log: ExperimentLog) -> dict[str, Any]:
    """Search one axis at a time, carrying the winner forward.

    Returns a selection block rather than the fitted model, because the final
    run refits on train plus validation. The block carries ``best_epoch``
    alongside the configuration: the final fit has no held-out set to stop on,
    so it trains for the epoch count validation chose here. Returning only the
    configuration would drop that number and the final fit would silently run
    the full sweep budget instead.
    """
    current = LSTMConfig(max_epochs=SWEEP_EPOCHS, patience=SWEEP_PATIENCE)
    previous_mae: float | None = None

    stages: list[tuple[str, str, list[dict[str, Any]]]] = [
        (
            "sequence_length",
            "How much history the window carries",
            [{"sequence_length": value} for value in (36, 72, 144, 288)],
        ),
        (
            "capacity",
            "Hidden size and depth",
            [{"hidden_size": h, "num_layers": layers} for h in (32, 64, 128) for layers in (1, 2)],
        ),
        (
            "optimisation",
            "Learning rate and batch size",
            [
                {"learning_rate": lr, "batch_size": bs}
                for lr in (1e-3, 3e-4)
                for bs in (32, 64, 128)
            ],
        ),
        (
            "regularisation",
            "Dropout and weight decay",
            [{"dropout": d, "weight_decay": wd} for d in (0.0, 0.2) for wd in (0.0, 1e-4)],
        ),
        (
            "calendar_ablation",
            "With and without calendar features",
            [{"use_calendar": True}, {"use_calendar": False}],
        ),
    ]

    for stage, description, grid in stages:
        print(f"\n--- lstm, area {area}: {stage} ({description}) ---")
        results: list[tuple[float, dict[str, Any], LSTMForecaster, Any, float]] = []

        for overrides in grid:
            candidate = replace(current, **overrides)
            model, metrics, wall = _fit_lstm(splits, config, candidate, seed=config.seeds[0])
            results.append((metrics.mae, overrides, model, metrics, wall))
            shown = ", ".join(f"{k}={v}" for k, v in overrides.items())
            print(
                f"  {shown:<40} MAE {metrics.mae:>8.2f}  MASE {metrics.mase:.3f}  "
                f"({wall:.0f}s, best epoch {model.history.best_epoch})",
                flush=True,
            )

            # Every candidate is logged, not only the stage winner. A rejected
            # candidate is evidence for the choice that was made, and the two
            # other models already log at this granularity.
            _log(
                log,
                model="lstm",
                area=area,
                stage=f"{stage}_candidate",
                hyperparams=model.describe(),
                feature_set="window+calendar" if candidate.use_calendar else "window",
                metrics=metrics,
                wall=wall,
                n_params=model.n_params(),
                rationale=(
                    f"Candidate in the {stage} sweep with {shown}: validation MAE "
                    f"{metrics.mae:.2f}, MASE {metrics.mase:.3f}, best epoch "
                    f"{model.history.best_epoch} of {len(model.history.train_loss)} run. "
                    "Logged so the axis can be reconstructed from the record rather "
                    "than from console output."
                ),
                seed=config.seeds[0],
            )

        results.sort(key=lambda row: row[0])
        best_mae, best_overrides, best_model, best_metrics, best_wall = results[0]
        worst_mae = results[-1][0]
        current = replace(current, **best_overrides)

        spread = (worst_mae - best_mae) / best_mae if best_mae else 0.0
        if stage == "calendar_ablation":
            with_cal = next(r[0] for r in results if r[1]["use_calendar"])
            without = next(r[0] for r in results if not r[1]["use_calendar"])
            delta = without - with_cal
            rationale = (
                f"Calendar features changed validation MAE by {delta:+.2f} "
                f"({delta / with_cal:+.1%}); "
                f"{'kept' if delta > 0 else 'dropped'} on that evidence. "
                "Staged search complete; the selected configuration is refit on "
                "train plus validation with the full epoch budget for the final run."
            )
        else:
            rationale = (
                f"{stage}: {', '.join(f'{k}={v}' for k, v in best_overrides.items())} "
                f"won {len(grid)} candidates with validation MAE {best_mae:.2f}; "
                f"spread across the axis was {spread:.1%}. "
                f"{_improvement(best_mae, previous_mae)}. "
                + (
                    "The axis matters, so it is worth holding this value fixed."
                    if spread > 0.02
                    else "The axis barely matters here, so later stages are unlikely "
                    "to be sensitive to it."
                )
            )

        _log(
            log,
            model="lstm",
            area=area,
            stage=stage,
            hyperparams=best_model.describe(),
            feature_set="window+calendar" if current.use_calendar else "window",
            metrics=best_metrics,
            wall=best_wall,
            n_params=best_model.n_params(),
            rationale=rationale,
            seed=config.seeds[0],
        )
        previous_mae = best_mae
        selected_epoch = best_model.history.best_epoch
        print(f"  -> {best_overrides}, validation MAE {best_mae:.2f}")

    print(f"  best epoch on validation: {selected_epoch}")
    return {**current.to_dict(), "best_epoch": selected_epoch}


# --------------------------------------------------------------------------
# Model 3: LightGBM via Optuna
# --------------------------------------------------------------------------


def tune_gbm(
    area: int, splits: SplitSet, config: Config, log: ExperimentLog, *, n_trials: int = 30
) -> dict[str, Any]:
    """Search tree hyperparameters jointly.

    Staged search is the wrong tool here: ``num_leaves``, ``min_child_samples``
    and the sampling fractions trade off against one another, so a value chosen
    for one while the others are fixed is not the value that would be chosen
    jointly. Trials are logged individually so the search is still inspectable.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"\n--- lightgbm, area {area}: Optuna over tree parameters ({n_trials} trials) ---")

    def objective(trial: optuna.Trial) -> float:
        candidate = GBMConfig(
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        )
        model = GBMForecaster(candidate, seed=config.seeds[0])
        started = time.perf_counter()
        model.fit(
            splits.train.values,
            train_times=splits.train.times,
            valid=splits.validation.values,
            valid_times=splits.validation.times,
        )
        wall = time.perf_counter() - started
        metrics = _score(model, splits, config)

        _log(
            log,
            model="lightgbm",
            area=area,
            stage="optuna",
            hyperparams=model.describe(),
            feature_set="lags+rolling+calendar",
            metrics=metrics,
            wall=wall,
            n_params=model.n_params(),
            rationale=(
                f"Optuna trial {trial.number}: validation MAE {metrics.mae:.2f}, "
                f"MASE {metrics.mase:.3f}, {model.best_iteration} trees after early "
                "stopping. Tree parameters are searched jointly because num_leaves, "
                "min_child_samples and the sampling fractions trade off against one "
                "another; a staged sweep would fix each at a value chosen while the "
                "others were wrong."
            ),
            seed=config.seeds[0],
        )
        return float(metrics.mae)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.seeds[0])
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  best trial {study.best_trial.number}: MAE {study.best_value:.2f}")
    print(f"  params: {study.best_params}")

    selected = GBMConfig(**study.best_params)
    model = GBMForecaster(selected, seed=config.seeds[0])
    model.fit(
        splits.train.values,
        train_times=splits.train.times,
        valid=splits.validation.values,
        valid_times=splits.validation.times,
    )
    print("  top features by gain:")
    for name, gain in model.importances(top=6):
        print(f"    {name:<22} {gain:6.1%}")

    # best_iteration travels with the configuration. The final fit has no
    # held-out set to stop on, so it uses the tree count early stopping chose
    # here; without it the refit would run the full n_estimators ceiling and be
    # a substantially more overfit model than the one that was selected.
    print(f"  trees after early stopping: {model.best_iteration}")
    return {**selected.to_dict(), "best_iteration": model.best_iteration}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


ALL_MODELS = ("harmonic_arima", "lstm", "lightgbm")


def train_all(
    config: Config,
    *,
    gbm_trials: int = 30,
    models: tuple[str, ...] = ALL_MODELS,
) -> dict[str, Any]:
    """Tune the requested models on the highest-traffic area, persisting each.

    Models are selectable because they do not want the same hardware. The LSTM
    is a 288-step recurrence that CPU threads cannot parallelise, so it belongs
    on a GPU, while the other two finish in minutes locally. Selections are
    merged into the existing file rather than replacing it, so a run on one
    machine does not discard what another already chose.
    """
    unknown = set(models) - set(ALL_MODELS)
    if unknown:
        raise ValueError(f"unknown models {sorted(unknown)}; choose from {list(ALL_MODELS)}")

    areas = SelectedAreas.load(config.paths.tables / "selected_areas.json")
    tuning_area = areas.top_traffic
    values, times = load_area_series(config, tuning_area)
    splits = make_splits(values, times, config)

    log = ExperimentLog(config.paths.experiments_csv)
    print(f"tuning on area {tuning_area} (rank 1), selecting on validation MAE")
    print(f"models this run: {', '.join(models)}")
    print(splits.describe())

    path = config.paths.tables / "selected_hyperparameters.json"
    selected: dict[str, Any] = {}
    if path.exists():
        selected = json.loads(path.read_text(encoding="utf-8"))
        kept = [m for m in ALL_MODELS if m in selected and m not in models]
        if kept:
            print(f"keeping existing selections for: {', '.join(kept)}")

    if selected.get("tuning_area", tuning_area) != tuning_area:
        raise ValueError(
            f"{path} was written for area {selected['tuning_area']} but this run tunes "
            f"on {tuning_area}; selections from different areas must not be mixed."
        )
    selected["tuning_area"] = tuning_area

    if "harmonic_arima" in models:
        selected["harmonic_arima"] = tune_harmonic(tuning_area, splits, config, log).describe()
    if "lstm" in models:
        selected["lstm"] = tune_lstm(tuning_area, splits, config, log)
    if "lightgbm" in models:
        selected["lightgbm"] = tune_gbm(tuning_area, splits, config, log, n_trials=gbm_trials)

    path.write_text(json.dumps(selected, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"\n{log.summary()}")
    print(f"\nwrote {path}")
    return selected


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="Path to a config YAML.")
    parser.add_argument("--gbm-trials", type=int, default=30, help="Optuna trials for LightGBM.")
    parser.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help=(
            "Comma-separated models to tune this run. Selections for models not "
            "listed are kept from the existing file. The LSTM wants a GPU; the "
            "other two are minutes on a CPU."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the staged search and write the selected hyperparameters."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    config.paths.mkdirs()
    models = tuple(name.strip() for name in args.models.split(",") if name.strip())
    train_all(config, gbm_trials=args.gbm_trials, models=models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
