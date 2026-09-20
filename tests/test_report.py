"""Integrity checks for the consolidated report.

The report restates numbers that live in the artefacts, so the failure mode is
drift: a table is regenerated, the prose keeps the old figure, and nothing
complains. These tests tie the document to the files it was written from, and
check the things a reader notices first -- a missing figure, a citation with no
entry, a table numbered twice.
"""

from __future__ import annotations

import csv
import json
import re

import pytest

from src.config import PROJECT_ROOT

REPORT = PROJECT_ROOT / "report" / "REPORT.md"


@pytest.fixture(scope="module")
def report() -> str:
    if not REPORT.exists():
        pytest.skip("REPORT.md not written")
    return REPORT.read_text(encoding="utf-8")


def test_every_referenced_figure_exists(report: str) -> None:
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report)
    assert refs, "the report references no figures"
    missing = [r for r in refs if not (REPORT.parent / r).exists()]
    assert missing == [], f"referenced figures that do not exist: {missing}"


def test_figures_are_numbered_contiguously_from_one(report: str) -> None:
    numbers = [int(n) for n in re.findall(r"\*\*Figure (\d+) —", report)]
    assert numbers == list(range(1, len(numbers) + 1)), f"figure numbering is {numbers}"


def test_tables_are_numbered_contiguously_from_one(report: str) -> None:
    numbers = [int(n) for n in re.findall(r"\*\*Table (\d+) —", report)]
    assert numbers == list(range(1, len(numbers) + 1)), f"table numbering is {numbers}"


def test_every_citation_has_an_entry_and_every_entry_is_cited(report: str) -> None:
    body, _, references = report.partition("## References")
    assert references, "no references section"
    used = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    defined = {int(n) for n in re.findall(r"^\[(\d+)\]", references, re.M)}
    assert used - defined == set(), f"cited but not listed: {sorted(used - defined)}"
    assert defined - used == set(), f"listed but never cited: {sorted(defined - used)}"


def test_reported_mase_values_match_the_metrics_table(report: str) -> None:
    """Every MASE in the results table must appear in the report as written."""
    path = PROJECT_ROOT / "results" / "tables" / "final_metrics_all_test.csv"
    if not path.exists():
        pytest.skip("test metrics not present")
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    missing = [
        f"{r['model']}@{r['square_id']}={float(r['mase']):.3f}"
        for r in rows
        if f"{float(r['mase']):.3f}" not in report
    ]
    assert missing == [], f"MASE values absent from the report: {missing}"


def test_headline_facts_appear_in_the_report(report: str) -> None:
    """The named facts the conclusions rest on must be the ones quoted."""
    path = PROJECT_ROOT / "report" / "FACTS.json"
    if not path.exists():
        pytest.skip("fact registry not built")
    facts = json.loads(path.read_text(encoding="utf-8"))

    for key in ("best_model_gain_over_persistence", "inference_ratio_harmonic_over_lstm"):
        rendered = facts[key]["display"].rstrip("x%")
        assert rendered in report, f"{key} ({facts[key]['display']}) is not quoted"


def test_report_states_the_baseline_result(report: str) -> None:
    """The finding most easily omitted: not every model beat persistence."""
    assert "only two of the three models beat persistence" in report.lower()


def test_report_keeps_its_limitations_section(report: str) -> None:
    assert "## 7.4 Limitations" in report or "Limitations" in report
    # The device-dependence of the LSTM selection is uncomfortable and must stay.
    assert "device-invariant" in report


def test_ai_declaration_is_present_and_written(report: str) -> None:
    """The brief requires a declaration, and it must be the author's own words.

    This asserts the section exists and is filled in -- not that it says anything
    in particular, because its content is the author's to decide.
    """
    heading = "## Appendix C — Declaration on the use of AI tools"
    assert heading in report
    body = report.split(heading, 1)[1].split("---", 1)[0].strip()
    assert body, "the declaration section is empty"
    assert "to be completed" not in body.lower(), "the declaration is still a placeholder"
    assert len(body.split()) >= 20, "the declaration looks too short to be a real statement"


def test_repository_link_is_filled_in(report: str) -> None:
    """A submitted report with a placeholder repo link has no repo link."""
    tail = report.rsplit("---", 1)[-1]
    assert "**Source code:**" in tail
    source_line = next(ln for ln in tail.splitlines() if "**Source code:**" in ln)
    assert "<" not in source_line, f"repository URL is still a placeholder: {source_line}"


def test_video_link_is_filled_in(report: str) -> None:
    """Skipped rather than failed while the video is outstanding.

    The demonstration video is a separate deliverable that cannot be produced from
    this repository, so a permanent red suite here would train the reader to ignore
    failures. The skip message is the reminder; run pytest with -rs to see it.
    """
    tail = report.rsplit("---", 1)[-1]
    video_line = next((ln for ln in tail.splitlines() if "**Demonstration video:**" in ln), "")
    assert video_line, "the report has no demonstration-video line"
    if "<" in video_line:
        pytest.skip("video URL not yet added to REPORT.md - required before submission")
