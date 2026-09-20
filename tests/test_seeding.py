"""Tests for deterministic seeding.

Reproducibility of the reported metrics rests on these, so each random source
is checked independently rather than assuming one seeding call covers all.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.seeding import seed_all


def test_seeding_makes_python_random_reproducible() -> None:
    seed_all(7)
    first = [random.random() for _ in range(5)]
    seed_all(7)
    assert [random.random() for _ in range(5)] == first


def test_seeding_makes_numpy_reproducible() -> None:
    seed_all(7)
    first = np.random.rand(5)
    seed_all(7)
    np.testing.assert_array_equal(np.random.rand(5), first)


def test_different_seeds_diverge() -> None:
    seed_all(0)
    a = np.random.rand(10)
    seed_all(1)
    b = np.random.rand(10)
    assert not np.allclose(a, b)


def test_seed_report_records_what_was_seeded() -> None:
    report = seed_all(3, strict=False)
    assert report.seed == 3
    assert report.strict is False
    assert set(report.to_dict()) == {
        "seed",
        "deterministic_strict",
        "torch_seeded",
        "cuda_seeded",
    }


def test_torch_is_seeded_when_available() -> None:
    torch = pytest.importorskip("torch")

    report = seed_all(11)
    assert report.torch_seeded is True

    first = torch.randn(5)
    seed_all(11)
    assert torch.equal(torch.randn(5), first)


def test_strict_mode_sets_cudnn_flags() -> None:
    torch = pytest.importorskip("torch")

    seed_all(5, strict=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
