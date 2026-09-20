#!/usr/bin/env python
"""Cross-platform task runner for the forecasting pipeline.

``make`` is not available on a default Windows install, and this project is
developed on Windows but executed on Kaggle's Linux images. This script is the
canonical entry point on both:

    python run.py env
    python run.py ingest --config config/kaggle.yaml
    python run.py all

The Makefile mirrors these targets for anyone who prefers ``make``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Anchored on this file rather than on the caller's working directory, so
# `python /path/to/run.py <task>` works from anywhere. A subprocess does not
# inherit the parent's sys.path, only PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parent

# Ordered so `all` runs the pipeline end to end.
PIPELINE = (
    "env",
    "download",
    "ingest",
    "matrix",
    "eda",
    "train",
    "final",
    "results",
    "evaluate",
    "report",
    "pdf",
)


@dataclass(frozen=True)
class Task:
    """A named task mapped to a runnable module."""

    name: str
    module: str
    help: str
    takes_config: bool = True


TASKS: tuple[Task, ...] = (
    Task("env", "src.env_report", "Record hardware and library versions"),
    Task("download", "src.download", "Fetch the 62 daily CDR files and the Milano Grid"),
    Task("ingest", "src.ingest", "Convert raw text to per-day blocks, with memory evidence"),
    Task(
        "pipeline",
        "src.pipeline",
        "Stream download -> ingest -> delete, one day at a time (use on Kaggle)",
    ),
    Task("benchmark", "src.memory_report", "Benchmark ingest strategies in isolated processes"),
    Task("matrix", "src.build_matrix", "Assemble the 8928 x 10000 matrix and per-square totals"),
    Task("eda", "src.run_eda", "Produce exploratory figures, statistics and the selected series"),
    Task("train", "src.train", "Train and tune the three models"),
    Task("final", "src.final_runs", "Refit on train+validation and evaluate on the test week"),
    Task("results", "src.run_results", "Phase 6 figures, diagnostics and failure analysis"),
    Task("evaluate", "src.evaluate", "Evaluate on the test week; write metrics and diagnostics"),
    Task("facts", "src.facts", "Regenerate the named fact registry from the artefacts"),
    Task("pdf", "src.export_pdf", "Render report/REPORT.md as a print-ready PDF"),
    Task("report", "src.export_report", "Export figures and tables into report/"),
    Task("test", "pytest", "Run the test suite", takes_config=False),
    Task("lint", "ruff", "Lint src/ and tests/", takes_config=False),
)

_BY_NAME = {t.name: t for t in TASKS}


def _run(module: str, args: list[str]) -> int:
    """Invoke a module with the current interpreter, returning its exit code."""
    cmd = [sys.executable, "-m", module, *args]
    env = dict(os.environ)
    root = str(PROJECT_ROOT)
    existing = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != root]
    env["PYTHONPATH"] = os.pathsep.join([root, *existing])
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=root, env=env)


def _task_args(task: Task, config: str | None, passthrough: list[str]) -> list[str]:
    """Assemble the argument list for one task."""
    if task.name == "test":
        return ["tests/", "-q", *passthrough]
    if task.name == "lint":
        return ["check", "src/", "tests/", *passthrough]
    args = list(passthrough)
    if task.takes_config and config:
        args = ["--config", config, *args]
    return args


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the task runner."""
    names = ", ".join([*(t.name for t in TASKS), "all"])
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="tasks:\n"
        + "\n".join(f"  {t.name:<10} {t.help}" for t in TASKS)
        + "\n  all        Run the full pipeline in order",
    )
    parser.add_argument("task", help=f"One of: {names}")
    parser.add_argument(
        "--config",
        default=None,
        help="Config YAML passed through to the task. Defaults to config/default.yaml, "
        "with config/kaggle.yaml layered on automatically in a Kaggle session.",
    )
    parser.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="Remaining arguments are forwarded verbatim to the task module.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch to one task, or run the whole pipeline for ``all``."""
    args = build_parser().parse_args(argv)

    if args.task == "all":
        for name in PIPELINE:
            task = _BY_NAME[name]
            print(f"\n=== {name}: {task.help} ===", flush=True)
            code = _run(task.module, _task_args(task, args.config, []))
            if code != 0:
                print(f"\ntask '{name}' failed with exit code {code}", file=sys.stderr)
                return code
        return 0

    task = _BY_NAME.get(args.task)
    if task is None:
        print(
            f"unknown task {args.task!r}. Available: " f"{', '.join([*_BY_NAME, 'all'])}",
            file=sys.stderr,
        )
        return 2

    return _run(task.module, _task_args(task, args.config, args.rest))


if __name__ == "__main__":
    raise SystemExit(main())
