"""Deterministic seeding across every source of randomness in the study.

Reported metrics must be reproducible to the significant figures quoted in the
report. That requires seeding Python's ``random``, NumPy, and PyTorch together,
and -- for the neural model -- also disabling the cuDNN autotuner, which
otherwise picks convolution algorithms based on runtime benchmarking and can
change results between runs on identical inputs.

Determinism is not free: ``torch.use_deterministic_algorithms`` can select
slower kernels. Because this study reports training times, :func:`seed_all`
takes ``strict`` as an explicit argument so the trade-off is a visible decision
rather than a hidden default.
"""

from __future__ import annotations

import contextlib
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["seed_all", "SeedReport"]


@dataclass(frozen=True)
class SeedReport:
    """Record of what was seeded, for the experiment log."""

    seed: int
    strict: bool
    torch_seeded: bool
    cuda_seeded: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "deterministic_strict": self.strict,
            "torch_seeded": self.torch_seeded,
            "cuda_seeded": self.cuda_seeded,
        }


def seed_all(seed: int, *, strict: bool = False) -> SeedReport:
    """Seed every random source this project uses.

    Args:
        seed: The seed value. Study runs use the seeds listed in
            ``config.seeds``.
        strict: Additionally force deterministic algorithm selection in
            PyTorch and disable cuDNN benchmarking. Use for final reported
            runs; leave off during hyperparameter sweeps where throughput
            matters more than exact reproducibility of intermediate results.

    Returns:
        A :class:`SeedReport` describing what was actually seeded, so the
        experiment log records whether torch and CUDA were present.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch_seeded = False
    cuda_seeded = False
    try:
        import torch
    except ImportError:
        return SeedReport(seed=seed, strict=strict, torch_seeded=False, cuda_seeded=False)

    torch.manual_seed(seed)
    torch_seeded = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cuda_seeded = True

    if strict:
        # cuBLAS needs this set before the handle is created to make matmul
        # reductions deterministic on CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Older torch, or an op with no deterministic implementation.
        with contextlib.suppress(AttributeError, RuntimeError):
            torch.use_deterministic_algorithms(True, warn_only=True)

    return SeedReport(seed=seed, strict=strict, torch_seeded=torch_seeded, cuda_seeded=cuda_seeded)
