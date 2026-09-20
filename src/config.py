"""Configuration loading and validation.

The YAML files under ``config/`` are the single source of truth for paths,
dates, split boundaries and hyperparameter defaults. Nothing in ``src/`` may
hardcode any of those values. Loading is strict: a missing or unknown key
raises rather than silently defaulting, because a typo in a split boundary
would otherwise corrupt every downstream result without surfacing an error.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "ConfigError", "DateRange", "load_config", "detect_env", "PROJECT_ROOT"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
_KAGGLE_CONFIG = PROJECT_ROOT / "config" / "kaggle.yaml"


class ConfigError(ValueError):
    """Raised when the configuration is missing keys or internally inconsistent."""


# --------------------------------------------------------------------------
# Leaf types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DateRange:
    """An inclusive date range at day granularity.

    Both endpoints are inclusive, so ``["2013-12-16", "2013-12-22"]`` spans
    ``2013-12-16 00:00`` through ``2013-12-22 23:50`` at 10-minute resolution.
    """

    start: dt.date
    end: dt.date

    @classmethod
    def from_list(cls, raw: Any, name: str) -> DateRange:
        """Build from a two-element ``[start, end]`` YAML list."""
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ConfigError(f"splits.{name} must be a [start, end] pair, got {raw!r}")
        start, end = (_as_date(v, f"splits.{name}") for v in raw)
        if end < start:
            raise ConfigError(f"splits.{name}: end {end} precedes start {start}")
        return cls(start=start, end=end)

    @property
    def n_days(self) -> int:
        """Number of whole days covered, inclusive of both endpoints."""
        return (self.end - self.start).days + 1

    def n_points(self, per_day: int) -> int:
        """Number of observations at ``per_day`` samples per day."""
        return self.n_days * per_day

    def __str__(self) -> str:
        return f"{self.start.isoformat()}..{self.end.isoformat()}"


@dataclass(frozen=True)
class Paths:
    """Filesystem roots. Relative paths resolve against the project root."""

    raw: Path
    interim: Path
    processed: Path
    results: Path

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def tables(self) -> Path:
        return self.results / "tables"

    @property
    def predictions(self) -> Path:
        return self.results / "predictions"

    @property
    def experiments_csv(self) -> Path:
        return self.results / "experiments.csv"

    @property
    def environment_json(self) -> Path:
        return self.results / "environment.json"

    def mkdirs(self) -> None:
        """Create every directory this project writes to."""
        for p in (
            self.raw,
            self.interim,
            self.processed,
            self.results,
            self.figures,
            self.tables,
            self.predictions,
        ):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetCfg:
    """Physical description of the Telecom Italia CDR dataset."""

    doi_traffic: str
    doi_grid: str
    dataverse_base: str
    start_date: dt.date
    end_date: dt.date
    timezone: str
    interval_minutes: int
    n_squares: int
    daily_period: int
    weekly_period: int

    @property
    def n_days(self) -> int:
        """Total days in the observation period, inclusive."""
        return (self.end_date - self.start_date).days + 1

    @property
    def n_timestamps(self) -> int:
        """Expected number of 10-minute intervals across the whole period."""
        return self.n_days * self.daily_period

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """Expected shape of the full traffic matrix."""
        return (self.n_timestamps, self.n_squares)


@dataclass(frozen=True)
class IngestCfg:
    """Controls for the raw-to-parquet ingest pass."""

    delete_raw_after_ingest: bool
    null_internet_fill: float
    compression: str


@dataclass(frozen=True)
class AreasCfg:
    """Geographical areas under study.

    ``top_traffic`` is null until Phase 3 resolves it empirically; runtime code
    reads the resolved value from ``results/tables/selected_areas.json`` rather
    than from this file.
    """

    top_traffic: int | None
    fixed: tuple[int, ...]
    appendix_top2_top3: tuple[int, ...] | None


@dataclass(frozen=True)
class SplitsCfg:
    """The four chronological, non-overlapping evaluation splits."""

    train: DateRange
    validation: DateRange
    test: DateRange
    stress: DateRange

    @property
    def ordered(self) -> tuple[tuple[str, DateRange], ...]:
        """Splits in chronological order, paired with their names."""
        return (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
            ("stress", self.stress),
        )


@dataclass(frozen=True)
class MissingDataCfg:
    """Policy for gaps in the reindexed 10-minute grid."""

    max_interpolate_gap: int
    escalate_if_in_test: bool


@dataclass(frozen=True)
class PreprocessingCfg:
    """Target transform, scaling and input-window defaults."""

    transform: str
    scaler: str
    sequence_length: int
    calendar_features: bool


@dataclass(frozen=True)
class EvaluationCfg:
    """Forecast horizon and metric conventions."""

    horizon: int
    seasonal_period: int
    mape_zero_threshold: float


@dataclass(frozen=True)
class Config:
    """Fully validated study configuration."""

    env: str
    paths: Paths
    dataset: DatasetCfg
    ingest: IngestCfg
    areas: AreasCfg
    splits: SplitsCfg
    missing_data: MissingDataCfg
    preprocessing: PreprocessingCfg
    evaluation: EvaluationCfg
    seeds: tuple[int, ...]
    source_files: tuple[Path, ...] = field(default=(), repr=False)

    @property
    def is_kaggle(self) -> bool:
        """True when the Kaggle overrides were applied."""
        return self.env == "kaggle"


# --------------------------------------------------------------------------
# Coercion helpers
# --------------------------------------------------------------------------


def _as_date(value: Any, where: str) -> dt.date:
    """Coerce a YAML scalar to a ``date``, rejecting anything ambiguous."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value)
        except ValueError as exc:
            raise ConfigError(f"{where}: {value!r} is not an ISO date") from exc
    raise ConfigError(f"{where}: expected a date, got {type(value).__name__}")


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    """Fetch ``key`` or raise with a path-qualified message."""
    if key not in mapping:
        raise ConfigError(f"missing required key: {where}.{key}")
    return mapping[key]


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Fetch a mapping section, rejecting a non-mapping value."""
    value = _require(raw, name, "config")
    if not isinstance(value, Mapping):
        raise ConfigError(f"config.{name} must be a mapping, got {type(value).__name__}")
    return value


def _build(cls: type, raw: Mapping[str, Any], where: str) -> Any:
    """Instantiate a flat dataclass from a mapping, rejecting unknown keys.

    Rejecting unknown keys is deliberate: a misspelled YAML key that was
    silently ignored would leave the intended setting at its default, and the
    resulting bug would surface only as a wrong number in the final report.
    """
    names = {f.name for f in fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}")
    kwargs = {}
    for f in fields(cls):
        if f.name not in raw:
            raise ConfigError(f"missing required key: {where}.{f.name}")
        kwargs[f.name] = raw[f.name]
    return cls(**kwargs)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _resolve_path(value: Any, where: str) -> Path:
    """Resolve a configured path, treating relative paths as project-relative.

    A POSIX-rooted path such as ``/kaggle/temp/raw`` must stay absolute even
    when this config is loaded on Windows for testing. ``WindowsPath`` reports
    such a path as *relative* because it carries no drive letter, so resolving
    on ``is_absolute()`` alone would silently rewrite the Kaggle paths to sit
    under the project root -- and the error would only surface as an ingest
    writing 20 GB into the wrong filesystem.
    """
    if not isinstance(value, str):
        raise ConfigError(f"{where}: expected a path string, got {type(value).__name__}")
    if value.startswith(("/", "\\")):
        return Path(value)
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _as_int_tuple(value: Any, where: str) -> tuple[int, ...] | None:
    """Coerce an optional YAML scalar or list of ints to a tuple."""
    if value is None:
        return None
    if isinstance(value, int):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{where}: expected a list of integers, got {value!r}")
    return tuple(int(v) for v in value)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _validate_dataset(dataset: DatasetCfg) -> None:
    """Assert the sampling geometry is self-consistent."""
    if dataset.end_date < dataset.start_date:
        raise ConfigError("dataset.end_date precedes dataset.start_date")
    expected_daily = (24 * 60) // dataset.interval_minutes
    if dataset.daily_period != expected_daily:
        raise ConfigError(
            f"dataset.daily_period is {dataset.daily_period} but a "
            f"{dataset.interval_minutes}-minute interval implies {expected_daily}"
        )
    if dataset.weekly_period != 7 * dataset.daily_period:
        raise ConfigError(
            f"dataset.weekly_period is {dataset.weekly_period}, expected "
            f"{7 * dataset.daily_period}"
        )


def _validate_splits(splits: SplitsCfg, dataset: DatasetCfg) -> None:
    """Assert the four splits are contiguous, non-overlapping and in range.

    Contiguity matters because a silent one-day gap between train and
    validation would shift every reported metric with no error raised.
    """
    ordered = splits.ordered
    for (prev_name, prev), (name, curr) in zip(ordered, ordered[1:], strict=False):
        if curr.start <= prev.end:
            raise ConfigError(f"splits.{name} ({curr}) overlaps splits.{prev_name} ({prev})")
        gap_days = (curr.start - prev.end).days - 1
        if gap_days:
            raise ConfigError(
                f"splits.{name} starts {gap_days} day(s) after splits.{prev_name} "
                "ends; splits must be contiguous"
            )

    first, last = ordered[0][1], ordered[-1][1]
    if first.start < dataset.start_date:
        raise ConfigError(
            f"splits.{ordered[0][0]} starts {first.start}, before dataset start "
            f"{dataset.start_date}"
        )
    if last.end > dataset.end_date:
        raise ConfigError(
            f"splits.{ordered[-1][0]} ends {last.end}, after dataset end " f"{dataset.end_date}"
        )


def _validate_preprocessing(pre: PreprocessingCfg) -> None:
    """Assert transform and scaler names are ones the codebase implements."""
    if pre.transform not in {"log1p", "none"}:
        raise ConfigError(f"preprocessing.transform {pre.transform!r} is not supported")
    if pre.scaler not in {"standard", "minmax", "none"}:
        raise ConfigError(f"preprocessing.scaler {pre.scaler!r} is not supported")
    if pre.sequence_length < 1:
        raise ConfigError("preprocessing.sequence_length must be >= 1")


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def detect_env() -> str:
    """Return ``"kaggle"`` inside a Kaggle notebook session, else ``"local"``.

    Kaggle sets ``KAGGLE_KERNEL_RUN_TYPE`` in every session and mounts
    ``/kaggle/input``; either signal is sufficient.
    """
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if Path("/kaggle/input").exists():
        return "kaggle"
    return "local"


def load_config(
    path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    auto_env: bool = True,
) -> Config:
    """Load, merge and validate the study configuration.

    ``config/default.yaml`` is always the base layer and any explicitly passed
    file is merged over it. That is what makes ``config/kaggle.yaml`` usable:
    it is an overlay holding only the handful of keys Kaggle changes, so
    loading it on its own would otherwise drop every other section.

    Args:
        path: Config file layered over the defaults. Pass
            ``config/kaggle.yaml`` to force the Kaggle paths.
        overrides: Extra mapping deep-merged last, for tests and ad-hoc runs.
        auto_env: When True and no explicit ``path`` is given, layer
            ``config/kaggle.yaml`` over the defaults if a Kaggle session is
            detected.

    Returns:
        A fully validated, immutable :class:`Config`.

    Raises:
        ConfigError: On a missing key, an unknown key, or an inconsistency
            between the dataset geometry and the split boundaries.
    """
    sources: list[Path] = [_DEFAULT_CONFIG]
    if path is not None:
        explicit = Path(path)
        if explicit.resolve() != _DEFAULT_CONFIG.resolve():
            sources.append(explicit)
    elif auto_env and detect_env() == "kaggle":
        sources.append(_KAGGLE_CONFIG)

    raw: dict[str, Any] = {}
    for src in sources:
        if not src.exists():
            raise ConfigError(f"config file not found: {src}")
        with src.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, Mapping):
            raise ConfigError(f"{src}: top level must be a mapping")
        raw = _deep_merge(raw, loaded)
    if overrides:
        raw = _deep_merge(raw, overrides)

    paths_raw = _section(raw, "paths")
    paths = Paths(
        raw=_resolve_path(_require(paths_raw, "raw", "paths"), "paths.raw"),
        interim=_resolve_path(_require(paths_raw, "interim", "paths"), "paths.interim"),
        processed=_resolve_path(_require(paths_raw, "processed", "paths"), "paths.processed"),
        results=_resolve_path(_require(paths_raw, "results", "paths"), "paths.results"),
    )

    ds_raw = dict(_section(raw, "dataset"))
    ds_raw["start_date"] = _as_date(_require(ds_raw, "start_date", "dataset"), "dataset.start_date")
    ds_raw["end_date"] = _as_date(_require(ds_raw, "end_date", "dataset"), "dataset.end_date")
    dataset = _build(DatasetCfg, ds_raw, "dataset")

    ingest = _build(IngestCfg, _section(raw, "ingest"), "ingest")

    areas_raw = _section(raw, "areas")
    areas = AreasCfg(
        top_traffic=_require(areas_raw, "top_traffic", "areas"),
        fixed=tuple(_as_int_tuple(_require(areas_raw, "fixed", "areas"), "areas.fixed") or ()),
        appendix_top2_top3=_as_int_tuple(
            _require(areas_raw, "appendix_top2_top3", "areas"),
            "areas.appendix_top2_top3",
        ),
    )

    splits_raw = _section(raw, "splits")
    splits = SplitsCfg(
        **{
            name: DateRange.from_list(_require(splits_raw, name, "splits"), name)
            for name in ("train", "validation", "test", "stress")
        }
    )

    missing_data = _build(MissingDataCfg, _section(raw, "missing_data"), "missing_data")
    preprocessing = _build(PreprocessingCfg, _section(raw, "preprocessing"), "preprocessing")
    evaluation = _build(EvaluationCfg, _section(raw, "evaluation"), "evaluation")

    seeds = _as_int_tuple(_require(raw, "seeds", "config"), "seeds") or ()
    if not seeds:
        raise ConfigError("config.seeds must list at least one integer seed")

    _validate_dataset(dataset)
    _validate_splits(splits, dataset)
    _validate_preprocessing(preprocessing)

    return Config(
        env=str(_require(raw, "env", "config")),
        paths=paths,
        dataset=dataset,
        ingest=ingest,
        areas=areas,
        splits=splits,
        missing_data=missing_data,
        preprocessing=preprocessing,
        evaluation=evaluation,
        seeds=seeds,
        source_files=tuple(sources),
    )
