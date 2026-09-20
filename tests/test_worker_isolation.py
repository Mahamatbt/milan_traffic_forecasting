"""Tests that benchmark workers run correctly from any working directory.

Each memory measurement runs in its own subprocess, which is what keeps
allocator reuse from making whichever strategy runs second look artificially
cheap. That isolation introduced a bug that only appeared on Kaggle: the worker
was launched with the caller's working directory, which there is
``/kaggle/working`` while the repository clone lives in ``/kaggle/working/repo``.
The worker died with ``ModuleNotFoundError: No module named 'src'`` after the
benchmark had already downloaded a 322 MB file.

The parent's ``sys.path`` is no help, because a subprocess does not inherit it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT
from src.memory_report import _measure_isolated, _worker_environment

N_SQUARES = 4
PER_DAY = 6
BASE_MS = 1_383_260_400_000  # 2013-10-31T23:00:00Z == 2013-11-01 00:00 Rome
STEP_MS = 240 * 60 * 1000  # 6 intervals/day


# --------------------------------------------------------------------------
# PYTHONPATH construction
# --------------------------------------------------------------------------


def test_repo_is_first_on_pythonpath(monkeypatch) -> None:
    other = str(Path("some") / "other" / "place")
    monkeypatch.setenv("PYTHONPATH", other)
    entries = _worker_environment()["PYTHONPATH"].split(os.pathsep)

    assert entries[0] == str(PROJECT_ROOT)
    assert other in entries


def test_repo_is_not_duplicated(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", str(PROJECT_ROOT))
    entries = _worker_environment()["PYTHONPATH"].split(os.pathsep)
    assert entries.count(str(PROJECT_ROOT)) == 1


def test_unset_pythonpath_is_handled(monkeypatch) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert _worker_environment()["PYTHONPATH"] == str(PROJECT_ROOT)


def test_existing_environment_is_preserved(monkeypatch) -> None:
    """The worker needs the rest of the environment, not just PYTHONPATH."""
    monkeypatch.setenv("SOME_UNRELATED_VAR", "kept")
    assert _worker_environment()["SOME_UNRELATED_VAR"] == "kept"


# --------------------------------------------------------------------------
# End-to-end worker launch
# --------------------------------------------------------------------------


def _write_tiny_day(directory) -> object:
    """A synthetic day small enough to measure in well under a second."""
    lines = []
    for interval in range(PER_DAY):
        time_ms = BASE_MS + interval * STEP_MS
        for square in range(1, N_SQUARES + 1):
            value = square + interval
            fields = [square, time_ms, 39, 0.1, 0.1, 0.1, 0.1, value]
            lines.append("\t".join(str(f) for f in fields))
    path = directory / "sms-call-internet-mi-2013-11-01.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_tiny_config(directory) -> object:
    """Config overlay shrinking the grid to match the synthetic day."""
    path = directory / "tiny.yaml"
    path.write_text(
        "\n".join(
            [
                "dataset:",
                f"  n_squares: {N_SQUARES}",
                f"  daily_period: {PER_DAY}",
                f"  weekly_period: {PER_DAY * 7}",
                "  interval_minutes: 240",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("strategy", ["polars_lazy", "pandas_chunked"])
def test_worker_runs_from_an_unrelated_working_directory(
    tmp_path, monkeypatch, strategy: str
) -> None:
    """The exact Kaggle condition: cwd is nowhere near the repository root."""
    monkeypatch.chdir(tmp_path)

    day = _write_tiny_day(tmp_path)
    config_file = _write_tiny_config(tmp_path)

    result = _measure_isolated(strategy, day, str(config_file))

    assert result["strategy"] == strategy
    assert result["peak_rss_delta"] >= 0
    assert result["wall_s"] > 0
    expected = sum(
        square + interval for interval in range(PER_DAY) for square in range(1, N_SQUARES + 1)
    )
    assert result["total_internet"] == pytest.approx(expected)


def test_worker_reports_a_relative_path_correctly(tmp_path, monkeypatch) -> None:
    """A relative --path would resolve against the worker's cwd, not the caller's."""
    monkeypatch.chdir(tmp_path)
    _write_tiny_day(tmp_path)
    config_file = _write_tiny_config(tmp_path)

    relative = Path("sms-call-internet-mi-2013-11-01.txt")
    assert not relative.is_absolute()

    result = _measure_isolated("polars_lazy", relative, str(config_file))
    assert result["strategy"] == "polars_lazy"


def test_worker_failure_surfaces_the_stderr(tmp_path, monkeypatch) -> None:
    """A dead worker must report why, not vanish into a generic error."""
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "sms-call-internet-mi-2013-11-01.txt"  # never created

    with pytest.raises(RuntimeError) as excinfo:
        _measure_isolated("polars_lazy", missing, None)

    message = str(excinfo.value)
    assert "polars_lazy" in message
    assert "exited" in message
