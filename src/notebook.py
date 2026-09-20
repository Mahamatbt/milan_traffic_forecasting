"""Notebook bootstrap helpers.

Notebooks in this project render figures and narrate findings; they never hold
model or preprocessing logic. This module exists so the boilerplate every
notebook needs -- import path, config, seeds, plot styling -- is one call rather
than a cell of setup duplicated across three notebooks and drifting apart.

Usage, as the first code cell:

    from src.notebook import setup
    ctx = setup()
    ctx.config.splits.test
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, Config, load_config
from src.seeding import SeedReport, seed_all
from src.timing import describe_environment

__all__ = ["NotebookContext", "setup", "apply_plot_style", "savefig"]

# Figure defaults. 300 dpi is the export requirement for the report; the
# on-screen default stays lower so notebook rendering is not sluggish.
_EXPORT_DPI = 300

_STYLE: dict[str, Any] = {
    "figure.figsize": (11, 4.5),
    "figure.dpi": 110,
    "savefig.dpi": _EXPORT_DPI,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
}


@dataclass(frozen=True)
class NotebookContext:
    """Everything a notebook needs after bootstrapping."""

    config: Config
    seed_report: SeedReport
    environment: str

    def describe(self) -> str:
        """Multi-line summary to print at the top of a notebook."""
        return "\n".join(
            [
                f"environment : {self.config.env}",
                f"config      : {', '.join(str(s) for s in self.config.source_files)}",
                f"seed        : {self.seed_report.seed} (strict={self.seed_report.strict})",
                f"hardware    : {self.environment}",
                f"test week   : {self.config.splits.test}",
            ]
        )


def apply_plot_style() -> None:
    """Apply the project-wide matplotlib style."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(_STYLE)


def setup(
    config_path: str | Path | None = None,
    *,
    seed: int | None = None,
    strict_determinism: bool = False,
    style: bool = True,
) -> NotebookContext:
    """Prepare a notebook session: import path, config, seeds and plot style.

    Safe to call more than once; re-running the first cell will not duplicate
    the ``sys.path`` entry.

    Args:
        config_path: Explicit config file. Defaults to ``config/default.yaml``
            with the Kaggle overrides layered on inside a Kaggle session.
        seed: Seed to apply. Defaults to the first entry of ``config.seeds``.
        strict_determinism: Force deterministic torch kernels. Off by default
            because it slows training and notebooks are exploratory.
        style: Apply the project matplotlib style.

    Returns:
        A :class:`NotebookContext` holding the loaded config and seed record.
    """
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    config = load_config(config_path)
    config.paths.mkdirs()

    seed_report = seed_all(seed if seed is not None else config.seeds[0], strict=strict_determinism)

    if style:
        apply_plot_style()

    return NotebookContext(
        config=config,
        seed_report=seed_report,
        environment=describe_environment(),
    )


def savefig(fig: Any, path: str | Path, *, dpi: int = _EXPORT_DPI) -> Path:
    """Save a figure at report resolution, creating parent directories.

    Args:
        fig: A matplotlib ``Figure``.
        path: Destination file path.
        dpi: Output resolution; defaults to the 300 dpi the report requires.

    Returns:
        The resolved path written to.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    return target
