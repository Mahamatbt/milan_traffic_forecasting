"""Timings must be attributable to the machine that produced them.

Requirement IV reports training and inference cost. A number without its
hardware is not a measurement, and the failure mode here is silent: the final
fits run on Kaggle while `results/environment.json` is written locally, so the
repository can end up claiming GPU timings on a machine with no CUDA device.
Nothing else in the suite would notice.
"""

from __future__ import annotations

import csv
import json

import pytest

from src.config import PROJECT_ROOT

TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS = PROJECT_ROOT / "results"


def _timing_devices() -> set[str]:
    devices: set[str] = set()
    for path in TABLES.glob("timing_*.csv"):
        with path.open(encoding="utf-8", newline="") as fh:
            devices.update(row["device"] for row in csv.DictReader(fh) if row.get("device"))
    return devices


def _environment_records() -> list[dict]:
    """Every committed hardware record, whatever it is named."""
    records = []
    for path in sorted(RESULTS.glob("environment*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def test_a_hardware_record_exists() -> None:
    assert _environment_records(), "no results/environment*.json; run `run.py` to record hardware"


def test_gpu_timings_have_a_gpu_machine_on_record() -> None:
    """A cuda row with no CUDA machine recorded is an unattributable number."""
    devices = _timing_devices()
    if not devices:
        pytest.skip("no timing tables yet")
    if not any(d.startswith("cuda") for d in devices):
        return

    records = _environment_records()
    with_gpu = [r for r in records if r.get("gpu", {}).get("available")]
    assert with_gpu, (
        "timing_*.csv reports cuda rows, but no committed results/environment*.json "
        "describes a machine with a CUDA device. The final fits ran on Kaggle; "
        "download results/environment.json from that session's Output tab and commit "
        "it (as environment_final_runs.json if it should not replace the local one). "
        f"Devices in the timing tables: {sorted(devices)}."
    )


def test_cpu_timings_have_a_cpu_machine_on_record() -> None:
    devices = _timing_devices()
    if not devices:
        pytest.skip("no timing tables yet")
    if not any(d == "cpu" for d in devices):
        return
    assert any(
        r.get("cpu", {}).get("physical_cores") for r in _environment_records()
    ), "timing tables report cpu rows with no CPU description on record"
