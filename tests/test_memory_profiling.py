"""Tests for the memory and timing instrumentation.

These measurements are the evidence behind the report's data-handling section,
so the instrument itself needs to be shown to work: the sampler must actually
catch a transient peak, and timing must survive a callable that raises.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.memory_profiling import (
    MemoryReport,
    deep_size_bytes,
    format_bytes,
    measure,
    rss_bytes,
)
from src.timing import Stopwatch, TimingResult, collect_environment, time_it

# --------------------------------------------------------------------------
# measure()
# --------------------------------------------------------------------------


def test_measure_records_wall_time() -> None:
    with measure("sleep", trace_python_allocs=False) as report:
        time.sleep(0.2)
    assert report.wall_s >= 0.19
    assert report.n_samples > 0


def test_measure_captures_a_transient_peak() -> None:
    """The peak must be caught while it exists, not read after it is freed.

    A 200 MB array is allocated, held long enough for the 50 ms sampler to see
    it, then released before the block exits. Reading RSS only at exit would
    miss it entirely.
    """
    with measure("transient", trace_python_allocs=False) as report:
        block = np.ones(25_000_000, dtype=np.float64)  # ~200 MB
        time.sleep(0.3)
        del block

    assert report.rss_peak_delta > 100 * 1024**2
    # And the peak must exceed the level the process settled back to.
    assert report.rss_peak >= report.rss_after


def test_measure_reports_tracemalloc_when_enabled() -> None:
    with measure("traced", trace_python_allocs=True) as report:
        _ = [object() for _ in range(10_000)]
    assert report.tracemalloc_peak is not None
    assert report.tracemalloc_peak > 0


def test_measure_omits_tracemalloc_when_disabled() -> None:
    with measure("untraced", trace_python_allocs=False) as report:
        pass
    assert report.tracemalloc_peak is None


def test_measure_populates_report_even_when_body_raises() -> None:
    report_holder: list[MemoryReport] = []
    with (
        pytest.raises(RuntimeError),
        measure("failing", trace_python_allocs=False) as report,
    ):
        report_holder.append(report)
        time.sleep(0.1)
        raise RuntimeError("boom")
    assert report_holder[0].wall_s >= 0.09


def test_report_to_row_is_flat_and_csv_safe() -> None:
    with measure("row", trace_python_allocs=False, notes="hello") as report:
        pass
    report.extra["strategy_family"] = "polars"
    row = report.to_row()
    assert row["label"] == "row"
    assert row["notes"] == "hello"
    assert row["strategy_family"] == "polars"
    assert all(not isinstance(v, (dict, list)) for v in row.values())


def test_rss_bytes_is_positive() -> None:
    assert rss_bytes() > 0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def test_deep_size_of_numpy_array() -> None:
    arr = np.zeros(1000, dtype=np.float32)
    assert deep_size_bytes(arr) == 4000


def test_deep_size_returns_none_for_unknown_types() -> None:
    assert deep_size_bytes(object()) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "0.00 B"), (1024, "1.00 KiB"), (1024**2, "1.00 MiB"), (1024**3, "1.00 GiB")],
)
def test_format_bytes(value: int, expected: str) -> None:
    assert format_bytes(value) == expected


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------


def test_time_it_runs_the_requested_repeats() -> None:
    calls = []

    def work() -> int:
        calls.append(1)
        time.sleep(0.05)
        return 42

    result, value = time_it(work, label="work", repeats=3)
    assert len(calls) == 3
    assert value == 42
    assert result.repeats == 3
    assert len(result.times_s) == 3
    assert result.mean_s >= 0.04
    assert result.std_s >= 0.0


def test_time_it_excludes_warmup_from_statistics() -> None:
    calls: list[int] = []

    def work() -> None:
        calls.append(1)

    result, _ = time_it(work, label="warm", repeats=2, warmup=3)
    assert len(calls) == 5
    assert len(result.times_s) == 2


def test_time_it_rejects_zero_repeats() -> None:
    with pytest.raises(ValueError, match="repeats"):
        time_it(lambda: None, repeats=0)


def test_timing_result_std_is_zero_for_single_run() -> None:
    result = TimingResult(label="one", repeats=1, times_s=[0.5])
    assert result.std_s == 0.0
    assert result.mean_s == 0.5


def test_timing_result_row_is_csv_safe() -> None:
    row = TimingResult(label="x", repeats=2, times_s=[0.1, 0.2]).to_row()
    assert row["mean_s"] == pytest.approx(0.15)
    assert isinstance(row["times_s"], str)


def test_stopwatch_measures_a_block() -> None:
    with Stopwatch("block") as sw:
        time.sleep(0.1)
    assert sw.elapsed_s >= 0.09


# --------------------------------------------------------------------------
# environment capture
# --------------------------------------------------------------------------


def test_collect_environment_has_the_fields_the_report_quotes() -> None:
    env = collect_environment()
    assert env["cpu"]["logical_cores"] >= 1
    assert env["memory"]["total_bytes"] > 0
    assert env["python"]["version"].startswith("3.")
    assert "available" in env["gpu"]
    assert "numpy" in env["packages"]


def test_record_hardware_writes_readable_json(tmp_path) -> None:
    import json

    target = tmp_path / "nested" / "environment.json"
    from src.timing import record_hardware

    written = record_hardware(target)
    assert target.exists()
    reloaded = json.loads(target.read_text(encoding="utf-8"))
    assert reloaded["cpu"]["logical_cores"] == written["cpu"]["logical_cores"]
