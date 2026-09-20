"""Tests for the task runner's subprocess environment.

``run.py`` launches each pipeline stage as ``python -m src.<module>``. That has
the same failure mode the memory benchmark hit on Kaggle: a subprocess does not
inherit the parent's ``sys.path``, so a runner invoked from anywhere other than
the repository root would fail with ``No module named 'src'``.

The benchmark's version of this bug was caught only after it had cost a Kaggle
session. This one is pinned here instead.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

from src.config import PROJECT_ROOT

RUNNER = PROJECT_ROOT / "run.py"


@pytest.fixture(scope="module")
def runner():
    """Import run.py as a module without executing its CLI.

    Registered in sys.modules before execution because @dataclass resolves
    annotations through the module entry, and fails without it.
    """
    spec = importlib.util.spec_from_file_location("project_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["project_runner"] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("project_runner", None)


def test_runner_exists() -> None:
    assert RUNNER.exists()


def test_project_root_is_the_repository(runner) -> None:
    """Anchored on the file's own location, not on the caller's directory."""
    assert runner.PROJECT_ROOT == PROJECT_ROOT


def test_every_pipeline_stage_has_a_task(runner) -> None:
    names = {task.name for task in runner.TASKS}
    for stage in runner.PIPELINE:
        assert stage in names, stage


# Modules a later phase will add. Listed rather than skipped, so the set shrinks
# as phases land and cannot quietly hide a module that was deleted by accident.
# src.train left this set when Phase 5 landed, which is what the companion test
# below enforces.
PENDING_MODULES: set[str] = set()


def test_task_modules_are_importable(runner) -> None:
    """A task pointing at a module that does not exist fails only when run."""
    missing = {
        task.module
        for task in runner.TASKS
        if task.takes_config and importlib.util.find_spec(task.module) is None
    }
    unexpected = missing - PENDING_MODULES
    assert unexpected == set(), f"tasks reference modules that do not exist: {unexpected}"


def test_pending_module_list_does_not_go_stale() -> None:
    """Once a phase lands, its module must leave the pending set."""
    landed = {name for name in PENDING_MODULES if importlib.util.find_spec(name) is not None}
    assert landed == set(), f"{landed} now exist; remove them from PENDING_MODULES in this test"


def test_runner_works_from_an_unrelated_directory(tmp_path) -> None:
    """The exact condition that broke the benchmark workers on Kaggle."""
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "env"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-800:]
    assert "No module named" not in completed.stderr


def test_unknown_task_reports_the_available_ones(tmp_path) -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "no-such-task"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        check=False,
    )
    assert completed.returncode == 2
    assert "unknown task" in completed.stderr
    assert "Available" in completed.stderr


def test_config_is_forwarded_only_to_tasks_that_accept_it(runner) -> None:
    config_task = next(t for t in runner.TASKS if t.name == "eda")
    assert "--config" in runner._task_args(config_task, "config/kaggle.yaml", [])

    test_task = next(t for t in runner.TASKS if t.name == "test")
    assert "--config" not in runner._task_args(test_task, "config/kaggle.yaml", [])


def test_extra_arguments_are_passed_through(runner) -> None:
    task = next(t for t in runner.TASKS if t.name == "ingest")
    args = runner._task_args(task, None, ["--limit", "3"])
    assert args == ["--limit", "3"]
