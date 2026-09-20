"""Tests for configuration loading and validation.

The split boundaries decide every reported number, so the validator is tested
against the specific ways it could silently go wrong: an overlap, a gap, or a
range that runs outside the observation period.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import FrozenInstanceError

import pytest
import yaml

from src.config import PROJECT_ROOT, Config, ConfigError, DateRange, load_config


@pytest.fixture(scope="module")
def config() -> Config:
    """The real default configuration, loaded once for the module."""
    return load_config(auto_env=False)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_default_config_loads(config: Config) -> None:
    assert isinstance(config, Config)
    assert config.env == "local"


def test_kaggle_config_overrides_paths_and_ingest() -> None:
    kaggle = load_config(PROJECT_ROOT / "config" / "kaggle.yaml")
    assert kaggle.env == "kaggle"
    assert kaggle.is_kaggle
    # Raw data must land in scratch, never in the 20 GB persisted working dir.
    assert str(kaggle.paths.raw).replace("\\", "/").startswith("/kaggle/temp")
    assert str(kaggle.paths.processed).replace("\\", "/").startswith("/kaggle/working")
    assert kaggle.ingest.delete_raw_after_ingest is True


def test_config_is_immutable(config: Config) -> None:
    with pytest.raises(FrozenInstanceError):
        config.env = "mutated"  # type: ignore[misc]


def test_paths_resolve_to_absolute(config: Config) -> None:
    for path in (
        config.paths.raw,
        config.paths.interim,
        config.paths.processed,
        config.paths.results,
    ):
        assert path.is_absolute()


# --------------------------------------------------------------------------
# Dataset geometry
# --------------------------------------------------------------------------


def test_observation_period_is_62_days(config: Config) -> None:
    """One file per day from 2013-11-01 to 2014-01-01 inclusive."""
    assert config.dataset.n_days == 62


def test_matrix_shape_matches_the_brief(config: Config) -> None:
    """62 days x 144 intervals x 10,000 squares."""
    assert config.dataset.matrix_shape == (8928, 10000)


def test_daily_period_consistent_with_interval(config: Config) -> None:
    assert config.dataset.daily_period == 24 * 60 // config.dataset.interval_minutes
    assert config.dataset.weekly_period == 7 * config.dataset.daily_period


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def test_split_lengths(config: Config) -> None:
    """Split sizes the rest of the pipeline asserts against."""
    per_day = config.dataset.daily_period
    assert config.splits.train.n_points(per_day) == 5472
    assert config.splits.validation.n_points(per_day) == 1008
    assert config.splits.test.n_points(per_day) == 1008
    assert config.splits.stress.n_points(per_day) == 1440


def test_test_split_is_the_assigned_week(config: Config) -> None:
    """The brief fixes the evaluation window at 16-22 December 2013."""
    assert config.splits.test.start == dt.date(2013, 12, 16)
    assert config.splits.test.end == dt.date(2013, 12, 22)
    assert config.splits.test.start.weekday() == 0  # Monday
    assert config.splits.test.end.weekday() == 6  # Sunday


def test_splits_are_contiguous_and_ordered(config: Config) -> None:
    ordered = config.splits.ordered
    for (_, prev), (_, curr) in zip(ordered, ordered[1:], strict=False):
        assert curr.start == prev.end + dt.timedelta(days=1)


def test_splits_lie_within_the_observation_period(config: Config) -> None:
    assert config.splits.train.start >= config.dataset.start_date
    assert config.splits.stress.end <= config.dataset.end_date


# --------------------------------------------------------------------------
# Validation failures
# --------------------------------------------------------------------------


def test_overlapping_splits_are_rejected() -> None:
    with pytest.raises(ConfigError, match="overlaps"):
        load_config(
            auto_env=False,
            overrides={"splits": {"validation": ["2013-12-08", "2013-12-15"]}},
        )


def test_gap_between_splits_is_rejected() -> None:
    with pytest.raises(ConfigError, match="contiguous"):
        load_config(
            auto_env=False,
            overrides={"splits": {"validation": ["2013-12-10", "2013-12-15"]}},
        )


def test_split_outside_observation_period_is_rejected() -> None:
    with pytest.raises(ConfigError, match="before dataset start"):
        load_config(
            auto_env=False,
            overrides={"splits": {"train": ["2013-10-01", "2013-12-08"]}},
        )


def test_unknown_key_is_rejected() -> None:
    """A typo must fail loudly rather than leave a default silently in place."""
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(
            auto_env=False,
            overrides={"preprocessing": {"sequence_lenght": 144}},
        )


def test_unsupported_transform_is_rejected() -> None:
    with pytest.raises(ConfigError, match="transform"):
        load_config(auto_env=False, overrides={"preprocessing": {"transform": "boxcox"}})


def test_inconsistent_daily_period_is_rejected() -> None:
    with pytest.raises(ConfigError, match="daily_period"):
        load_config(auto_env=False, overrides={"dataset": {"daily_period": 96}})


def test_missing_config_file_is_rejected(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_overlay_file_layers_over_defaults(tmp_path) -> None:
    """A partial file supplies only what it changes; the rest comes from defaults."""
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        yaml.safe_dump({"preprocessing": {"sequence_length": 288}}), encoding="utf-8"
    )
    cfg = load_config(overlay)
    assert cfg.preprocessing.sequence_length == 288
    assert cfg.dataset.matrix_shape == (8928, 10000)  # inherited from defaults


def test_section_replaced_by_a_scalar_is_rejected(tmp_path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump({"dataset": "not-a-mapping"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(broken)


def test_missing_required_key_is_rejected() -> None:
    """A section missing a field must fail loudly rather than default silently."""
    from src.config import EvaluationCfg, _build

    with pytest.raises(ConfigError, match="missing required key: evaluation.horizon"):
        _build(EvaluationCfg, {"seasonal_period": 144, "mape_zero_threshold": 1.0}, "evaluation")


# --------------------------------------------------------------------------
# DateRange
# --------------------------------------------------------------------------


def test_date_range_inclusive_of_both_endpoints() -> None:
    rng = DateRange.from_list(["2013-12-16", "2013-12-22"], "test")
    assert rng.n_days == 7
    assert rng.n_points(144) == 1008


def test_date_range_rejects_reversed_bounds() -> None:
    with pytest.raises(ConfigError, match="precedes"):
        DateRange.from_list(["2013-12-22", "2013-12-16"], "test")


def test_date_range_rejects_malformed_input() -> None:
    with pytest.raises(ConfigError, match="pair"):
        DateRange.from_list(["2013-12-16"], "test")
    with pytest.raises(ConfigError, match="not an ISO date"):
        DateRange.from_list(["16/12/2013", "2013-12-22"], "test")


# --------------------------------------------------------------------------
# Areas
# --------------------------------------------------------------------------


def test_fixed_areas_are_the_two_from_the_brief(config: Config) -> None:
    assert config.areas.fixed == (4159, 4556)


def test_top_traffic_is_unresolved_until_eda(config: Config) -> None:
    """Phase 3 resolves this empirically; it must not be guessed in advance."""
    assert config.areas.top_traffic is None
