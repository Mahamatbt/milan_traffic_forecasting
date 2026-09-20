"""Tests for the streaming download-ingest-delete pipeline.

The pipeline exists because Kaggle scratch is discarded at session end, so the
properties that matter are about interruption, not throughput: completed days
must be skipped, the raw file must survive until its block is safely written,
and a time budget must stop between days rather than mid-write.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from src.config import load_config
from src.download import DataverseFile, GuestbookResponse
from src.pipeline import PipelineResult, _remaining_days, run_streaming

N_SQUARES = 3
PER_DAY = 4
BASE_MS = 1_383_260_400_000  # 2013-10-31T23:00:00Z == 2013-11-01 00:00 Rome
STEP_MS = 360 * 60 * 1000

GUESTBOOK = GuestbookResponse(
    name="A Researcher", email="a@example.edu", institution="Uni", position="Student"
)


@pytest.fixture
def config(tmp_path):
    """Miniature 4-day study writing into tmp_path."""
    return load_config(
        auto_env=False,
        overrides={
            "paths": {
                "raw": str(tmp_path / "raw"),
                "interim": str(tmp_path / "interim"),
                "processed": str(tmp_path / "processed"),
                "results": str(tmp_path / "results"),
            },
            "dataset": {
                "start_date": "2013-11-01",
                "end_date": "2013-11-04",
                "n_squares": N_SQUARES,
                "daily_period": PER_DAY,
                "weekly_period": PER_DAY * 7,
                "interval_minutes": 360,
            },
            "splits": {
                "train": ["2013-11-01", "2013-11-01"],
                "validation": ["2013-11-02", "2013-11-02"],
                "test": ["2013-11-03", "2013-11-03"],
                "stress": ["2013-11-04", "2013-11-04"],
            },
            "ingest": {"delete_raw_after_ingest": True},
        },
    )


def _day_text(day_index: int) -> str:
    """Raw TSV for one synthetic day, split across two country rows per cell."""
    lines = []
    for t in range(PER_DAY):
        time_ms = BASE_MS + (day_index * PER_DAY + t) * STEP_MS
        for s in range(1, N_SQUARES + 1):
            value = s * 100 + t
            lines.append(f"{s}\t{time_ms}\t39\t0.1\t0.1\t0.1\t0.1\t{value * 0.5}")
            lines.append(f"{s}\t{time_ms}\t33\t0.1\t0.1\t0.1\t0.1\t{value * 0.5}")
    return "\n".join(lines) + "\n"


def _files() -> list[DataverseFile]:
    """Dataverse listing for the four synthetic days."""
    out = []
    for i in range(4):
        date = (dt.date(2013, 11, 1) + dt.timedelta(days=i)).isoformat()
        out.append(
            DataverseFile(
                file_id=100 + i,
                filename=f"sms-call-internet-mi-{date}.txt",
                filesize=len(_day_text(i).encode()),
                md5="0" * 32,
                content_type="text/plain",
                restricted=False,
            )
        )
    return out


class FakeClient:
    """Serves the synthetic day files and records what was fetched."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.downloaded: list[str] = []

    def list_files(self, doi: str) -> list[DataverseFile]:
        return _files()


@pytest.fixture
def patched_download(monkeypatch):
    """Replace the real transfer with one that writes the synthetic text."""
    calls: list[str] = []

    def fake_download_file(client, entry, dest_dir, guestbook, *, show_progress=True, **kw):
        from src.download import DownloadResult

        calls.append(entry.filename)
        if getattr(client, "fail_on", None) == entry.filename:
            raise RuntimeError("simulated transfer failure")

        dest_dir.mkdir(parents=True, exist_ok=True)
        index = int(entry.filename[-6:-4]) - 1
        path = dest_dir / entry.filename
        path.write_text(_day_text(index), encoding="utf-8")
        return DownloadResult(
            filename=entry.filename,
            file_id=entry.file_id,
            path=path,
            size_bytes=entry.filesize,
            md5=entry.md5,
            verified=True,
            skipped=False,
            resumed_from=0,
            wall_s=0.01,
            attempts=1,
        )

    monkeypatch.setattr("src.pipeline.download_file", fake_download_file)
    monkeypatch.setattr("src.pipeline.DataverseClient", lambda *a, **k: FakeClient())
    return calls


# --------------------------------------------------------------------------
# Day selection
# --------------------------------------------------------------------------


def test_remaining_days_are_sorted_by_date(config) -> None:
    pairs = _remaining_days(_files(), config, force=False)
    assert [date for date, _ in pairs] == [
        "2013-11-01",
        "2013-11-02",
        "2013-11-03",
        "2013-11-04",
    ]


def test_completed_days_are_skipped(config) -> None:
    """Resumption is the whole point: a finished day must not be refetched."""
    interim = config.paths.interim
    interim.mkdir(parents=True, exist_ok=True)
    for suffix, payload in ((".npy", None), (".times.npy", None), (".json", {})):
        if suffix.endswith("npy"):
            np.save(interim / f"2013-11-02{suffix}", np.zeros((1, 1), np.float32))
        else:
            (interim / f"2013-11-02{suffix}").write_text(json.dumps(payload), encoding="utf-8")

    pairs = _remaining_days(_files(), config, force=False)
    assert "2013-11-02" not in [date for date, _ in pairs]
    assert len(pairs) == 3


def test_force_reprocesses_completed_days(config) -> None:
    interim = config.paths.interim
    interim.mkdir(parents=True, exist_ok=True)
    np.save(interim / "2013-11-02.npy", np.zeros((1, 1), np.float32))
    np.save(interim / "2013-11-02.times.npy", np.zeros(1, "datetime64[ns]"))
    (interim / "2013-11-02.json").write_text("{}", encoding="utf-8")

    assert len(_remaining_days(_files(), config, force=True)) == 4


# --------------------------------------------------------------------------
# Streaming behaviour
# --------------------------------------------------------------------------


def test_processes_every_day_and_writes_blocks(config, patched_download) -> None:
    result = run_streaming(config, GUESTBOOK, show_progress=False)

    assert len(result.processed) == 4
    for i in range(4):
        date = (dt.date(2013, 11, 1) + dt.timedelta(days=i)).isoformat()
        block = np.load(config.paths.interim / f"{date}.npy")
        assert block.shape == (PER_DAY, N_SQUARES)
        assert block[0, 0] == pytest.approx(100.0)


def test_raw_is_deleted_after_the_block_is_written(config, patched_download) -> None:
    """Peak disk stays near one file, which is what Kaggle scratch requires."""
    run_streaming(config, GUESTBOOK, show_progress=False)

    assert list(config.paths.raw.glob("*.txt")) == []
    assert len(list(config.paths.interim.glob("*.npy"))) == 8  # values + times per day


def test_raw_is_retained_when_configured(tmp_path, patched_download) -> None:
    config = load_config(
        auto_env=False,
        overrides={
            "paths": {
                "raw": str(tmp_path / "raw"),
                "interim": str(tmp_path / "interim"),
                "processed": str(tmp_path / "processed"),
                "results": str(tmp_path / "results"),
            },
            "dataset": {
                "start_date": "2013-11-01",
                "end_date": "2013-11-04",
                "n_squares": N_SQUARES,
                "daily_period": PER_DAY,
                "weekly_period": PER_DAY * 7,
                "interval_minutes": 360,
            },
            "splits": {
                "train": ["2013-11-01", "2013-11-01"],
                "validation": ["2013-11-02", "2013-11-02"],
                "test": ["2013-11-03", "2013-11-03"],
                "stress": ["2013-11-04", "2013-11-04"],
            },
            "ingest": {"delete_raw_after_ingest": False},
        },
    )
    run_streaming(config, GUESTBOOK, show_progress=False)
    assert len(list(config.paths.raw.glob("*.txt"))) == 4


def test_limit_stops_early(config, patched_download) -> None:
    result = run_streaming(config, GUESTBOOK, limit=2, show_progress=False)
    assert len(result.processed) == 2
    assert len(patched_download) == 2


def test_rerun_after_partial_progress_resumes(config, patched_download) -> None:
    """A resumed run must continue, not restart."""
    run_streaming(config, GUESTBOOK, limit=2, show_progress=False)
    first_pass = list(patched_download)

    second = run_streaming(config, GUESTBOOK, show_progress=False)

    assert len(second.processed) == 2
    assert [c for c in patched_download if c not in first_pass] == [
        "sms-call-internet-mi-2013-11-03.txt",
        "sms-call-internet-mi-2013-11-04.txt",
    ]


def test_time_budget_stops_between_days(config, patched_download) -> None:
    """Stopping between days keeps every written block intact."""
    result = run_streaming(config, GUESTBOOK, show_progress=False, time_budget_s=-1.0)

    assert result.processed == []
    assert patched_download == []


def test_a_failing_day_does_not_destroy_earlier_progress(config, monkeypatch) -> None:
    """If day 3 fails, days 1 and 2 must remain on disk as finished artefacts."""
    from src.download import DownloadResult

    def fake_download_file(client, entry, dest_dir, guestbook, *, show_progress=True, **kw):
        if entry.filename.endswith("2013-11-03.txt"):
            raise RuntimeError("simulated transfer failure")
        dest_dir.mkdir(parents=True, exist_ok=True)
        index = int(entry.filename[-6:-4]) - 1
        path = dest_dir / entry.filename
        path.write_text(_day_text(index), encoding="utf-8")
        return DownloadResult(
            filename=entry.filename,
            file_id=entry.file_id,
            path=path,
            size_bytes=entry.filesize,
            md5=entry.md5,
            verified=True,
            skipped=False,
            resumed_from=0,
            wall_s=0.01,
            attempts=1,
        )

    monkeypatch.setattr("src.pipeline.download_file", fake_download_file)
    monkeypatch.setattr("src.pipeline.DataverseClient", lambda *a, **k: FakeClient())

    with pytest.raises(RuntimeError, match="simulated"):
        run_streaming(config, GUESTBOOK, show_progress=False)

    from src.ingest import day_is_complete

    assert day_is_complete("2013-11-01", config.paths.interim)
    assert day_is_complete("2013-11-02", config.paths.interim)
    assert not day_is_complete("2013-11-03", config.paths.interim)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_summary_handles_a_no_op_run() -> None:
    assert "nothing to do" in PipelineResult().summary()


def test_summary_reports_totals(config, patched_download) -> None:
    result = run_streaming(config, GUESTBOOK, show_progress=False)
    summary = result.summary()
    assert "days processed   : 4" in summary
    assert "absent cells" in summary
