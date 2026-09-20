"""Tests for the fact registry.

The registry exists to stop figures going stale, so what matters is that each
fact is genuinely derived from an artefact, that distinct quantities keep
distinct names, and that regenerating is deterministic.
"""

from __future__ import annotations

import json

import pytest

from src.config import load_config
from src.facts import Fact, collect_facts, load_facts, write_facts


@pytest.fixture(scope="module")
def config():
    return load_config(auto_env=False)


@pytest.fixture(scope="module")
def facts(config):
    collected = collect_facts(config)
    if not collected.flat():
        pytest.skip("no artefacts present; run python run.py eda")
    return collected


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def test_every_fact_names_its_source(facts) -> None:
    """A figure without provenance cannot be checked, so it is not a fact."""
    unsourced = [f.key for f in facts.flat().values() if not f.source.strip()]
    assert unsourced == []


def test_every_fact_has_a_description(facts) -> None:
    undescribed = [f.key for f in facts.flat().values() if not f.description.strip()]
    assert undescribed == []


def test_keys_are_unique_across_groups(facts) -> None:
    all_keys = [f.key for group in facts.groups.values() for f in group]
    duplicates = {k for k in all_keys if all_keys.count(k) > 1}
    assert duplicates == set()


def test_rendering_is_never_empty(facts) -> None:
    assert all(f.rendered().strip() for f in facts.flat().values())


# --------------------------------------------------------------------------
# The distinction the registry exists to enforce
# --------------------------------------------------------------------------


def test_country_codes_and_rows_per_cell_are_separate_facts(facts) -> None:
    """The error that motivated this module: 246 and 36 are different things.

    Both are real measurements, so no value check could tell them apart. Naming
    them separately is what makes the confusion impossible to restate.
    """
    flat = facts.flat()
    assert "max_rows_per_cell" in flat
    assert "distinct_country_codes_per_file" in flat
    assert flat["max_rows_per_cell"].value == 36
    assert flat["distinct_country_codes_per_file"].value == 246
    assert flat["max_rows_per_cell"].value != flat["distinct_country_codes_per_file"].value


def test_mean_rows_per_day_is_the_period_mean_not_one_day(facts) -> None:
    """4.8 M is 2013-11-01; the period mean is 5.16 M."""
    flat = facts.flat()
    if "mean_rows_per_day" not in flat:
        pytest.skip("ingest summary not present")
    assert 5.1e6 < flat["mean_rows_per_day"].value < 5.2e6


def test_download_rate_is_in_decimal_megabits(facts) -> None:
    """447 was mebibits per second; the decimal rate is ~469 Mbit/s."""
    flat = facts.flat()
    if "download_mbit_per_s" not in flat:
        pytest.skip("timing facts not present")
    assert 460 < flat["download_mbit_per_s"].value < 480


def test_anomaly_rate_excludes_the_unevaluable_first_day(facts) -> None:
    flat = facts.flat()
    if "anomalies_evaluable" not in flat:
        pytest.skip("anomaly facts not present")
    assert flat["anomalies_evaluable"].value == 8928 - 144


# --------------------------------------------------------------------------
# Consistency with the artefacts
# --------------------------------------------------------------------------


def test_matrix_size_derives_from_the_configured_shape(facts, config) -> None:
    rows, cols = config.dataset.matrix_shape
    expected = rows * cols * 4 / 1024**2
    assert facts.flat()["matrix_mib"].value == pytest.approx(expected)


def test_absent_cells_agree_with_the_matrix_report(facts) -> None:
    flat = facts.flat()
    if "absent_cells" not in flat:
        pytest.skip("matrix report not present")
    report = json.loads(
        (load_config(auto_env=False).paths.processed / "matrix_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert flat["absent_cells"].value == report["total_absent_cells"]


def test_memory_facts_cover_every_benchmarked_strategy(facts) -> None:
    flat = facts.flat()
    if "polars_lazy_peak_mib" not in flat:
        pytest.skip("benchmark not present")
    for strategy in ("naive_pandas", "pandas_chunked", "polars_lazy"):
        assert f"{strategy}_peak_mib" in flat
        assert f"{strategy}_wall_s" in flat


def test_area_facts_cover_every_study_area(facts) -> None:
    flat = facts.flat()
    if "rank_5161" not in flat:
        pytest.skip("selected areas not present")
    for square in (5161, 5059, 5259, 4159, 4556):
        assert f"rank_{square}" in flat


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_write_and_reload_round_trip(facts, tmp_path) -> None:
    json_path, md_path = write_facts(facts, tmp_path)
    assert json_path.exists() and md_path.exists()

    reloaded = load_facts(tmp_path)
    assert set(reloaded) == set(facts.flat())
    for entry in reloaded.values():
        assert set(entry) == {"value", "display", "unit", "source", "description"}


def test_regeneration_is_deterministic(config, tmp_path) -> None:
    """Regenerating must not churn the file, or diffs become unreadable."""
    first = write_facts(collect_facts(config), tmp_path / "a")[0].read_text(encoding="utf-8")
    second = write_facts(collect_facts(config), tmp_path / "b")[0].read_text(encoding="utf-8")
    assert first == second


def test_markdown_index_lists_every_fact(facts, tmp_path) -> None:
    _, md_path = write_facts(facts, tmp_path)
    text = md_path.read_text(encoding="utf-8")
    for key in facts.flat():
        assert f"`{key}`" in text, key


def test_load_facts_reports_a_missing_registry(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="run python -m src.facts"):
        load_facts(tmp_path)


def test_fact_rendering_formats_by_type() -> None:
    assert Fact("a", 1234567, "", "x", "d").rendered() == "1,234,567"
    assert Fact("b", 3.14159, "", "x", "d").rendered() == "3.14"
    assert Fact("c", 0.5, "", "x", "d", "50%").rendered() == "50%"
    assert Fact("d", "5161, 5059", "", "x", "d").rendered() == "5161, 5059"
