"""Tests for the results stage.

The stage reads committed forecasts and writes committed artefacts, so what is
checked here is that the artefacts are portable: a summary that records where a
figure lives must not record where it lives *on one machine*.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT
from src.run_results import MODELS, PLOTTED, _relative


def test_paths_are_recorded_relative_to_the_project() -> None:
    absolute = PROJECT_ROOT / "results" / "figures" / "forecast_test_5161_lstm.png"
    assert _relative(absolute) == "results/figures/forecast_test_5161_lstm.png"


def test_relative_paths_use_forward_slashes() -> None:
    """The committed file must read the same on Windows and Linux."""
    rendered = _relative(PROJECT_ROOT / "results" / "tables" / "copying_test.csv")
    assert "\\" not in rendered


def test_paths_outside_the_project_are_left_alone() -> None:
    """A Kaggle session writes to /kaggle/working, which has no project root."""
    outside = Path("/kaggle/working/results/figures/x.png")
    assert _relative(outside).endswith("results/figures/x.png")


@pytest.mark.parametrize("split", ["test", "stress"])
def test_committed_summary_holds_no_absolute_paths(split: str) -> None:
    """Regression: these summaries embedded one machine's directory layout.

    An absolute path in a committed artefact can never match on another
    machine, and leaks the author's filesystem into the repository.
    """
    path = PROJECT_ROOT / "results" / "tables" / f"results_summary_{split}.json"
    if not path.exists():
        pytest.skip("results summary not produced yet")
    summary = json.loads(path.read_text(encoding="utf-8"))

    for key in ("figures", "tables"):
        for entry in summary[key]:
            assert not Path(entry).is_absolute(), f"{key} holds an absolute path: {entry}"
            assert ":" not in entry, f"{key} holds a drive letter: {entry}"
            assert entry.startswith("results/"), entry


def test_plotted_models_are_a_subset_of_known_models() -> None:
    assert set(PLOTTED) <= set(MODELS)
