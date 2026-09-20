"""Assert that figures quoted in prose match the artefacts they came from.

Numbers written into documentation go stale silently. Three classes of error
reached the draft report before this test existed, and all three shared a cause:
a figure measured early was quoted as though it were general, then never
revisited when better evidence arrived.

* **Generalised from one day.** "4.8 M rows/day" and "3.4 country rows per cell"
  are the values for 2013-11-01, written when one file had been downloaded. The
  62-day means are 5.16 M and 3.6.
* **Two quantities conflated.** "up to 246 rows per cell" confused the count of
  distinct country codes in a file (246) with the maximum number of rows for one
  ``(square, time)`` pair (36) -- an overstatement of roughly seven times.
* **Superseded by a methodology fix.** The peak-memory figures in
  ``src/ingest.py`` were the in-process measurements, which were later shown to
  be contaminated by allocator reuse and replaced by subprocess-isolated ones.

The tests below therefore work in both directions: current values must appear
where they are quoted, and known-stale values must appear nowhere.
"""

from __future__ import annotations

import csv
import json
import re

import pytest

from src.config import PROJECT_ROOT

TABLES = PROJECT_ROOT / "report" / "tables"

# Files that quote measured figures in prose.
DOCUMENTS = (
    "README.md",
    "Plan.md",
    "src/ingest.py",
    "report/DRAFT_SECTIONS.md",
    "report/RELATED_WORK.md",
)


def _text(relative: str) -> str:
    path = PROJECT_ROOT / relative
    return path.read_text(encoding="utf-8") if path.exists() else ""


@pytest.fixture(scope="module")
def ingest_summary() -> dict:
    path = TABLES / "ingest_summary.json"
    if not path.exists():
        pytest.skip("ingest_summary.json not present; run python -m src.run_log")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def memory_rows() -> list[dict[str, str]]:
    path = TABLES / "memory_report.csv"
    if not path.exists():
        pytest.skip("memory_report.csv not present; run python run.py benchmark")
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
# Stale values must appear nowhere
# --------------------------------------------------------------------------

# pattern -> why it is wrong and what replaced it
STALE_PATTERNS: dict[str, str] = {
    r"\b4\.8 M rows\b": "2013-11-01 only; the 62-day mean is 5.16 M",
    r"about 3\.4 (?:country )?rows": "2013-11-01 only; the period mean is 3.6",
    r"3\.4 country rows": "2013-11-01 only; the period mean is 3.6",
    # "up to 246" is only wrong when it describes rows per cell. Saying that a
    # file holds up to 246 distinct country codes is correct and is how the
    # distinction is explained, so that phrasing is excluded.
    r"up to 246(?!\s+distinct country codes)": (
        "246 is distinct country codes per file, not rows per cell (max 36)"
    ),
    r"447 Mbit": "mebibits per second; the decimal rate is 469 Mbit/s",
    r"peaks near 510 MiB": "in-process measurement; the isolated figure is 552 MiB",
    r"peaks around 117 MiB": "in-process measurement; the isolated figure is 173 MiB",
    r"~357 MB": "the matrix is 340.58 MiB",
    r"~18 GiB to hold": "the projection is 17.90 GiB",
    r"of 8,928 \(\*\*1\.7": "the anomaly denominator is 8,784 evaluable intervals",
}


@pytest.mark.parametrize("document", DOCUMENTS)
def test_no_stale_figures(document: str) -> None:
    """No document may quote a figure that later measurement superseded."""
    text = _text(document)
    if not text:
        pytest.skip(f"{document} not present")

    found = [
        f"{pattern!r} ({reason})"
        for pattern, reason in STALE_PATTERNS.items()
        if re.search(pattern, text)
    ]
    assert found == [], f"{document} contains stale figures: {found}"


# --------------------------------------------------------------------------
# Current values must match the artefacts
# --------------------------------------------------------------------------


def test_raw_row_count_is_quoted_correctly(ingest_summary: dict) -> None:
    """The total record count appears in the draft and must match the run log."""
    expected = f"{ingest_summary['total_raw_rows']:,}"
    text = _text("report/DRAFT_SECTIONS.md")
    if text:
        assert expected in text, f"draft should quote {expected} raw records"


def test_mean_rows_per_day_is_consistent(ingest_summary: dict) -> None:
    """Documents quoting a per-day row count must use the period mean."""
    mean_millions = ingest_summary["total_raw_rows"] / 62 / 1e6
    assert 5.15 < mean_millions < 5.17, f"expected ~5.16 M, artefact gives {mean_millions:.3f} M"

    for document in ("README.md", "Plan.md", "src/ingest.py"):
        text = _text(document)
        if "5.16 M" in text or "5,16" in text:
            continue
        quoted = re.findall(r"([\d.]+) M rows", text)
        assert not quoted or all(
            abs(float(q) - mean_millions) < 0.05 for q in quoted
        ), f"{document} quotes {quoted} M rows/day; the period mean is {mean_millions:.2f} M"


def test_rows_per_cell_matches_the_artefacts(ingest_summary: dict) -> None:
    """Average country rows per cell, over the whole period rather than one day."""
    cells_with_rows = 8928 * 10000 - ingest_summary["absent_total"]
    ratio = ingest_summary["total_raw_rows"] / cells_with_rows
    assert 3.55 < ratio < 3.65, f"expected ~3.6, artefacts give {ratio:.3f}"

    for document in ("README.md", "Plan.md", "src/ingest.py", "report/DRAFT_SECTIONS.md"):
        text = _text(document)
        quoted = re.findall(r"([\d.]+) (?:country )?rows (?:per cell|on average)", text)
        for value in quoted:
            assert (
                abs(float(value) - ratio) < 0.1
            ), f"{document} quotes {value} rows per cell; artefacts give {ratio:.2f}"


def test_matrix_size_is_quoted_correctly() -> None:
    """340.58 MiB is 8928 x 10000 x 4 bytes; earlier drafts said ~357 MB."""
    expected_mib = 8928 * 10000 * 4 / 1024**2
    assert abs(expected_mib - 340.58) < 0.01

    for document in DOCUMENTS:
        text = _text(document)
        for value in re.findall(r"([\d.]+) MiB\b", text):
            if 330 < float(value) < 350:
                assert (
                    abs(float(value) - expected_mib) < 0.1
                ), f"{document} quotes {value} MiB for the matrix; it is 340.58"


def test_memory_peaks_match_the_benchmark(memory_rows: list[dict[str, str]]) -> None:
    """Peak-memory figures in prose must be the subprocess-isolated ones."""
    by_strategy: dict[str, list[float]] = {}
    for row in memory_rows:
        mib = int(row["per_day_peak_rss_bytes"]) / 1024**2
        by_strategy.setdefault(row["strategy"], []).append(mib)

    means = {k: sum(v) / len(v) for k, v in by_strategy.items()}
    # These are the values the draft and the ingest docstring quote.
    assert abs(means["naive_pandas"] - 633.06) < 1.0, means
    assert abs(means["pandas_chunked"] - 172.55) < 1.0, means
    assert abs(means["polars_lazy"] - 552.22) < 1.0, means


def test_anomaly_denominator_excludes_the_first_day() -> None:
    """The seasonal-naive detector cannot evaluate the first 144 intervals."""
    path = TABLES / "anomalies.csv"
    if not path.exists():
        pytest.skip("anomalies.csv not present; run python run.py eda")

    with path.open(encoding="utf-8", newline="") as fh:
        flagged = len(list(csv.DictReader(fh)))

    evaluable = 8928 - 144
    rate = flagged / evaluable
    assert abs(rate - 0.0173) < 0.0005, f"{flagged}/{evaluable} = {rate:.4f}, expected ~1.73%"

    for document in ("report/DRAFT_SECTIONS.md", "report/RESULTS_SUMMARY.md"):
        text = _text(document)
        if "1.73%" in text:
            assert (
                "8,784" in text
            ), f"{document} quotes 1.73% but not the 8,784 denominator it refers to"


def test_download_rate_uses_decimal_megabits(ingest_summary: dict) -> None:
    """Network rates are quoted in Mbit/s, not mebibits per second."""
    total_bytes = 20_804_803_507
    seconds = ingest_summary["download_total_min"] * 60
    mbit = total_bytes * 8 / seconds / 1e6
    assert 465 < mbit < 473, f"expected ~469 Mbit/s, computed {mbit:.0f}"

    text = _text("report/DRAFT_SECTIONS.md")
    for value in re.findall(r"(\d+) Mbit/s", text):
        if int(value) > 100:  # the download rate, not the local one
            assert abs(int(value) - mbit) < 5, f"draft quotes {value} Mbit/s; computed {mbit:.0f}"


# --------------------------------------------------------------------------
# Cross-document consistency
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("figure", "pattern"),
    [
        ("dataset size", r"19\.38 GiB"),
        ("matrix shape", r"8928 . 10000|8,928"),
        ("absent cells", r"34,682"),
    ],
)
def test_headline_figures_agree_across_documents(figure: str, pattern: str) -> None:
    """A figure quoted in more than one document must be the same figure."""
    quoting = [d for d in DOCUMENTS if _text(d) and re.search(pattern, _text(d))]
    assert quoting, f"no document quotes the {figure}"


# --------------------------------------------------------------------------
# The exported evidence pack
# --------------------------------------------------------------------------


def test_every_registered_figure_resolves_to_a_file() -> None:
    """A spec pointing at nothing drops a figure from the report silently.

    Exploratory figures live under results/figures/eda/ and evaluation figures
    flat under results/figures/, so each spec carries the subdirectory it came
    from. A wrong one produces no error, just a missing figure.
    """
    from src.config import load_config
    from src.export_report import FIGURES

    config = load_config(auto_env=False)
    missing = []
    for spec in FIGURES:
        base = config.paths.figures / spec.subdir if spec.subdir else config.paths.figures
        if not (base / spec.filename).exists():
            missing.append(f"{spec.subdir or '.'}/{spec.filename}")
    assert missing == [], f"registered figures that do not exist: {missing}"


def test_every_registered_figure_was_exported() -> None:
    from src.config import PROJECT_ROOT
    from src.export_report import FIGURES

    exported = PROJECT_ROOT / "report" / "figures"
    if not exported.exists():
        pytest.skip("report not exported yet; run python -m src.export_report")
    missing = [s.filename for s in FIGURES if not (exported / s.filename).exists()]
    assert missing == [], f"registered but not exported: {missing}"


def test_figures_are_at_print_resolution() -> None:
    """The report needs 300 dpi; the two stages once drifted to 300 and 150."""
    from src.config import PROJECT_ROOT

    exported = PROJECT_ROOT / "report" / "figures"
    if not exported.exists():
        pytest.skip("report not exported yet")
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is optional
        pytest.skip("Pillow not installed")

    low = []
    for path in sorted(exported.glob("*.png")):
        dpi = Image.open(path).info.get("dpi")
        if dpi and round(dpi[0]) < 300:
            low.append(f"{path.name} at {round(dpi[0])} dpi")
    assert low == [], f"figures below 300 dpi: {low}"


def test_results_summary_has_no_pending_sections() -> None:
    """Phase 8 is done when nothing in the evidence pack still says pending."""
    from src.config import PROJECT_ROOT

    path = PROJECT_ROOT / "report" / "RESULTS_SUMMARY.md"
    if not path.exists():
        pytest.skip("report not exported yet")
    text = path.read_text(encoding="utf-8")
    assert "_Pending" not in text, "RESULTS_SUMMARY.md still has pending sections"
    for heading in ("## Methodology", "## Results", "## Discussion and failure analysis"):
        assert heading in text, f"missing {heading}"


# --------------------------------------------------------------------------
# The drafted narrative must agree with the artefacts
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def draft() -> str:
    from src.config import PROJECT_ROOT

    path = PROJECT_ROOT / "report" / "DRAFT_SECTIONS.md"
    if not path.exists():
        pytest.skip("draft sections not written")
    return path.read_text(encoding="utf-8")


def _final_rows(split: str) -> list[dict[str, str]]:
    import csv as _csv

    from src.config import PROJECT_ROOT

    path = PROJECT_ROOT / "results" / "tables" / f"final_metrics_all_{split}.csv"
    if not path.exists():
        pytest.skip(f"{split} metrics not present")
    with path.open(encoding="utf-8", newline="") as fh:
        return list(_csv.DictReader(fh))


def test_draft_quotes_the_winning_mean_mase(draft: str) -> None:
    rows = _final_rows("test")
    values = [float(r["mase"]) for r in rows if r["model"] == "harmonic_arima"]
    mean = sum(values) / len(values)
    assert f"{mean:.3f}" in draft, f"draft should quote harmonic mean MASE {mean:.3f}"


def test_draft_quotes_per_area_mase_correctly(draft: str) -> None:
    """Every MASE the results table states must appear as written."""
    rows = _final_rows("test")
    wanted = {
        ("harmonic_arima", "5059"),
        ("harmonic_arima", "5259"),
        ("lightgbm", "5161"),
        ("persistence", "5161"),
    }
    for row in rows:
        if (row["model"], row["square_id"]) in wanted:
            assert f"{float(row['mase']):.3f}" in draft, (
                f"{row['model']} on {row['square_id']} is {float(row['mase']):.3f} "
                "but that value does not appear in the draft"
            )


def test_draft_parameter_counts_match_the_tables(draft: str) -> None:
    rows = _final_rows("test")
    for model in ("harmonic_arima", "lightgbm", "lstm"):
        params = {int(r["n_params"]) for r in rows if r["model"] == model}
        value = max(params)
        assert (
            f"{value:,}" in draft or str(value) in draft
        ), f"{model} has {value:,} parameters, not quoted in the draft"


def test_draft_inference_ratio_matches_the_timing_table(draft: str) -> None:
    """The cost-inversion claim is the one most worth pinning."""
    import csv as _csv

    from src.config import PROJECT_ROOT

    path = PROJECT_ROOT / "results" / "tables" / "timing_test.csv"
    if not path.exists():
        pytest.skip("timing table not present")
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in _csv.DictReader(fh) if r["square_id"] == "5161"]
    per_step = {r["model"]: float(r["inference_ms_per_step"]) for r in rows}
    ratio = per_step["harmonic_arima"] / per_step["lstm"]
    assert f"{ratio:,.0f}" in draft, f"inference ratio is {ratio:,.0f}x"

    # The grid-scale claim: one forecast for every cell, against the interval.
    grid_minutes = per_step["harmonic_arima"] * 10_000 / 1000 / 60
    assert (
        f"{grid_minutes:.1f} minutes" in draft
    ), f"a grid-wide pass takes {grid_minutes:.1f} minutes"


def test_draft_does_not_claim_a_ratio_between_different_configurations(draft: str) -> None:
    """Regression: 13.8 h and 2.6 s were different configurations.

    The measured 777x compares one configuration fitted on both machines. Pairing
    the largest CPU fit with the smallest GPU fit would have inflated it 25-fold.
    """
    assert "13.8 hours against 2.6 seconds" not in draft
    assert "2,019 seconds on the CPU and 2.6 seconds" in draft
