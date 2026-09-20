"""Tests for the Dataverse downloader.

These run offline against a fake client. The behaviours worth protecting are
the ones that only show up when a 19 GiB transfer goes wrong: resuming from a
partial file, refusing to trust a corrupt one, and coping with a server that
ignores a Range header and restarts the body.

The real API shape is pinned in ``REAL_ENTRY``, captured from the live endpoint,
so a change upstream fails here rather than mid-download.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.download import (
    CHUNK_SIZE,
    DataverseFile,
    DownloadError,
    GuestbookResponse,
    download_file,
    ensure_disk_space,
    load_dotenv,
    verify_md5,
    write_manifest,
)

# Captured verbatim from
# GET /api/datasets/:persistentId/versions/:latest/files?persistentId=doi:10.7910/DVN/EGZHFV
REAL_ENTRY = {
    "description": "",
    "label": "sms-call-internet-mi-2013-11-01.txt",
    "restricted": False,
    "version": 1,
    "datasetVersionId": 181170,
    "dataFile": {
        "id": 2674255,
        "persistentId": "doi:10.7910/DVN/EGZHFV/AKDGBV",
        "filename": "sms-call-internet-mi-2013-11-01.txt",
        "contentType": "text/plain; charset=US-ASCII",
        "filesize": 322874887,
        "md5": "658C49F8E3C7910BD3BB2BAFBF62CDCE",
        "checksum": {"type": "MD5", "value": "658c49f8e3c7910bd3bb2bafbf62cdce"},
        "tabularData": False,
    },
}

GUESTBOOK = GuestbookResponse(
    name="A Researcher", email="a@example.edu", institution="Uni", position="Student"
)


@pytest.fixture(autouse=True)
def _no_backoff_sleeping(monkeypatch) -> None:
    """Neutralise retry backoff so exercising the retry paths stays instant."""
    monkeypatch.setattr("src.download._backoff_sleep", lambda attempt: None)


# --------------------------------------------------------------------------
# API parsing
# --------------------------------------------------------------------------


def test_parses_the_real_api_entry_shape() -> None:
    f = DataverseFile.from_entry(REAL_ENTRY)
    assert f.file_id == 2674255
    assert f.filename == "sms-call-internet-mi-2013-11-01.txt"
    assert f.filesize == 322874887
    assert f.restricted is False


def test_md5_is_normalised_to_lowercase() -> None:
    """The API returns an uppercase md5; comparisons must not be case-sensitive."""
    assert DataverseFile.from_entry(REAL_ENTRY).md5 == "658c49f8e3c7910bd3bb2bafbf62cdce"


def test_entry_without_datafile_is_rejected() -> None:
    with pytest.raises(DownloadError, match="no 'dataFile' key"):
        DataverseFile.from_entry({"label": "x"})


def test_entry_missing_a_required_field_is_rejected() -> None:
    entry = {"dataFile": {"id": 1, "filename": "x.txt"}}
    with pytest.raises(DownloadError, match="missing"):
        DataverseFile.from_entry(entry)


# --------------------------------------------------------------------------
# Guestbook
# --------------------------------------------------------------------------


def test_guestbook_payload_shape() -> None:
    payload = GUESTBOOK.to_payload()
    assert set(payload["guestbookResponse"]) == {"name", "email", "institution", "position"}
    assert payload["guestbookResponse"]["email"] == "a@example.edu"


def test_guestbook_rejects_blank_fields() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        GuestbookResponse(name="  ", email="a@b.c", institution="U", position="S")


def test_guestbook_resolution_names_every_missing_field(monkeypatch) -> None:
    for var in ("NAME", "EMAIL", "INSTITUTION", "POSITION"):
        monkeypatch.delenv(f"DATAVERSE_GB_{var}", raising=False)
    args = SimpleNamespace(gb_name=None, gb_email=None, gb_institution=None, gb_position=None)
    with pytest.raises(DownloadError) as excinfo:
        GuestbookResponse.from_args_or_env(args)
    message = str(excinfo.value)
    for flag in ("--gb-name", "--gb-email", "--gb-institution", "--gb-position"):
        assert flag in message


def test_guestbook_reads_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("DATAVERSE_GB_NAME", "Env Name")
    monkeypatch.setenv("DATAVERSE_GB_EMAIL", "env@example.edu")
    monkeypatch.setenv("DATAVERSE_GB_INSTITUTION", "Env Uni")
    monkeypatch.setenv("DATAVERSE_GB_POSITION", "Researcher")
    args = SimpleNamespace(gb_name=None, gb_email=None, gb_institution=None, gb_position=None)
    assert GuestbookResponse.from_args_or_env(args).name == "Env Name"


def test_cli_flag_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv("DATAVERSE_GB_NAME", "Env Name")
    monkeypatch.setenv("DATAVERSE_GB_EMAIL", "env@example.edu")
    monkeypatch.setenv("DATAVERSE_GB_INSTITUTION", "Env Uni")
    monkeypatch.setenv("DATAVERSE_GB_POSITION", "Researcher")
    args = SimpleNamespace(
        gb_name="Flag Name", gb_email=None, gb_institution=None, gb_position=None
    )
    assert GuestbookResponse.from_args_or_env(args).name == "Flag Name"


# --------------------------------------------------------------------------
# .env loading
# --------------------------------------------------------------------------


def test_dotenv_sets_unset_variables(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DATAVERSE_GB_NAME", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(["# a comment", "", "DATAVERSE_GB_NAME=From File"]), encoding="utf-8"
    )

    assert load_dotenv(env_file) == {"DATAVERSE_GB_NAME": "From File"}
    assert os.environ["DATAVERSE_GB_NAME"] == "From File"


def test_real_environment_wins_over_dotenv(tmp_path, monkeypatch) -> None:
    """Kaggle Secrets must override the checked-out file, not the reverse."""
    monkeypatch.setenv("DATAVERSE_GB_NAME", "From Environment")
    env_file = tmp_path / ".env"
    env_file.write_text("DATAVERSE_GB_NAME=From File", encoding="utf-8")

    assert load_dotenv(env_file) == {}
    assert os.environ["DATAVERSE_GB_NAME"] == "From Environment"


def test_dotenv_strips_quotes_and_ignores_junk(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DATAVERSE_GB_INSTITUTION", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(["no_equals_sign_here", 'DATAVERSE_GB_INSTITUTION="Some University"']),
        encoding="utf-8",
    )
    assert load_dotenv(env_file)["DATAVERSE_GB_INSTITUTION"] == "Some University"


def test_dotenv_absent_is_not_an_error(tmp_path) -> None:
    assert load_dotenv(tmp_path / "nope.env") == {}


# --------------------------------------------------------------------------
# Client configuration
# --------------------------------------------------------------------------


def test_client_replaces_the_blocked_default_user_agent() -> None:
    """Harvard's WAF answers the default python-requests agent with HTTP 403.

    requests.Session populates User-Agent itself, so setdefault silently fails
    to displace it -- a bug that only shows up against the live API.
    """
    import requests

    from src.download import USER_AGENT, DataverseClient

    session = requests.Session()
    assert session.headers["User-Agent"] == requests.utils.default_user_agent()

    DataverseClient("https://example.org", session=session)
    assert session.headers["User-Agent"] == USER_AGENT
    assert "python-requests" not in session.headers["User-Agent"].split()[0]


def test_client_preserves_a_caller_supplied_user_agent() -> None:
    import requests

    from src.download import DataverseClient

    session = requests.Session()
    session.headers["User-Agent"] = "my-own-agent/2.0"
    DataverseClient("https://example.org", session=session)
    assert session.headers["User-Agent"] == "my-own-agent/2.0"


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeResponse:
    """Minimal stand-in for a streaming ``requests.Response``."""

    def __init__(self, body: bytes, status_code: int) -> None:
        self._body = body
        self.status_code = status_code

    def iter_content(self, chunk_size: int):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def close(self) -> None:
        pass


class FakeClient:
    """Serves a fixed payload, with configurable range and failure behaviour."""

    def __init__(
        self,
        payload: bytes,
        *,
        honour_range: bool = True,
        fail_first: int = 0,
        truncate_to: int | None = None,
    ) -> None:
        self.payload = payload
        self.honour_range = honour_range
        self.fail_first = fail_first
        self.truncate_to = truncate_to
        self.signed_url_calls = 0
        self.stream_calls: list[int] = []

    def signed_url(self, file_id: int, guestbook: GuestbookResponse) -> str:
        self.signed_url_calls += 1
        return f"https://signed.example/{file_id}?token={self.signed_url_calls}"

    def open_stream(self, url: str, *, start_byte: int = 0):
        self.stream_calls.append(start_byte)
        if self.fail_first > 0:
            self.fail_first -= 1
            raise DownloadError("simulated transient network failure")
        if start_byte and self.honour_range:
            body = self.payload[start_byte:]
            status = 206
        else:
            body = self.payload
            status = 200
        if self.truncate_to is not None:
            body = body[: self.truncate_to]
        return FakeResponse(body, status), status == 206


@pytest.fixture
def payload() -> bytes:
    # Larger than one chunk so the streaming loop actually iterates.
    return b"milan-traffic-" * (CHUNK_SIZE // 8)


@pytest.fixture
def spec(payload: bytes) -> DataverseFile:
    return DataverseFile(
        file_id=42,
        filename="day.txt",
        filesize=len(payload),
        md5=hashlib.md5(payload).hexdigest(),
        content_type="text/plain",
        restricted=False,
    )


# --------------------------------------------------------------------------
# Transfer behaviour
# --------------------------------------------------------------------------


def test_downloads_and_verifies(tmp_path: Path, payload: bytes, spec: DataverseFile) -> None:
    result = download_file(FakeClient(payload), spec, tmp_path, GUESTBOOK, show_progress=False)
    assert result.verified and not result.skipped
    assert (tmp_path / "day.txt").read_bytes() == payload
    assert not (tmp_path / "day.txt.part").exists()


def test_rerun_skips_a_verified_file(tmp_path: Path, payload: bytes, spec: DataverseFile) -> None:
    """Re-running the command must be a no-op, not a re-download."""
    client = FakeClient(payload)
    download_file(client, spec, tmp_path, GUESTBOOK, show_progress=False)
    before = client.signed_url_calls

    result = download_file(client, spec, tmp_path, GUESTBOOK, show_progress=False)
    assert result.skipped and result.verified
    assert client.signed_url_calls == before  # no further network access


def test_resumes_from_a_partial_file(tmp_path: Path, payload: bytes, spec: DataverseFile) -> None:
    partial = len(payload) // 3
    (tmp_path / "day.txt.part").write_bytes(payload[:partial])

    client = FakeClient(payload)
    result = download_file(client, spec, tmp_path, GUESTBOOK, show_progress=False)

    assert result.resumed_from == partial
    assert client.stream_calls == [partial]
    assert (tmp_path / "day.txt").read_bytes() == payload


def test_restarts_when_server_ignores_range(
    tmp_path: Path, payload: bytes, spec: DataverseFile
) -> None:
    """A 200 in reply to a Range request means the whole body is coming.

    Appending it to the partial file would silently corrupt the download, so
    the writer must truncate and start over.
    """
    (tmp_path / "day.txt.part").write_bytes(payload[: len(payload) // 2])

    client = FakeClient(payload, honour_range=False)
    result = download_file(client, spec, tmp_path, GUESTBOOK, show_progress=False)

    assert result.verified
    assert (tmp_path / "day.txt").read_bytes() == payload


def test_corrupt_existing_file_is_replaced(
    tmp_path: Path, payload: bytes, spec: DataverseFile
) -> None:
    """A file of the right length but wrong content must not be trusted."""
    (tmp_path / "day.txt").write_bytes(b"x" * len(payload))

    result = download_file(FakeClient(payload), spec, tmp_path, GUESTBOOK, show_progress=False)
    assert result.verified and not result.skipped
    assert (tmp_path / "day.txt").read_bytes() == payload


def test_md5_mismatch_discards_the_part_file(
    tmp_path: Path, payload: bytes, spec: DataverseFile
) -> None:
    """A bad transfer must not be left behind for a later run to resume from."""
    wrong = DataverseFile(**{**spec.__dict__, "md5": "0" * 32})
    with pytest.raises(DownloadError, match="MD5 mismatch"):
        download_file(FakeClient(payload), wrong, tmp_path, GUESTBOOK, show_progress=False)
    assert not (tmp_path / "day.txt.part").exists()
    assert not (tmp_path / "day.txt").exists()


def test_retries_transient_failures(tmp_path: Path, payload: bytes, spec: DataverseFile) -> None:
    client = FakeClient(payload, fail_first=2)
    result = download_file(client, spec, tmp_path, GUESTBOOK, show_progress=False)
    assert result.verified
    assert result.attempts == 3


def test_connection_dropped_mid_stream_recovers_by_resuming(
    tmp_path: Path, payload: bytes, spec: DataverseFile
) -> None:
    """A body cut short is not a failure: the next attempt resumes and finishes it."""
    client = FakeClient(payload, truncate_to=len(payload) // 2)
    result = download_file(client, spec, tmp_path, GUESTBOOK, show_progress=False)

    assert result.verified
    assert result.attempts == 2
    assert client.stream_calls == [0, len(payload) // 2]
    assert (tmp_path / "day.txt").read_bytes() == payload


def test_transfer_that_never_progresses_eventually_fails(
    tmp_path: Path, payload: bytes, spec: DataverseFile
) -> None:
    """A server returning an empty body forever must not loop indefinitely."""
    client = FakeClient(payload, truncate_to=0)
    with pytest.raises(DownloadError, match="failed to download"):
        download_file(client, spec, tmp_path, GUESTBOOK, show_progress=False)


def test_oversized_part_file_is_discarded(
    tmp_path: Path, payload: bytes, spec: DataverseFile
) -> None:
    """A sidecar longer than the expected size is from a different version."""
    (tmp_path / "day.txt.part").write_bytes(payload + b"extra-bytes")

    client = FakeClient(payload)
    result = download_file(client, spec, tmp_path, GUESTBOOK, show_progress=False)
    assert result.resumed_from == 0
    assert (tmp_path / "day.txt").read_bytes() == payload


def test_each_attempt_requests_a_fresh_signed_url(
    tmp_path: Path, payload: bytes, spec: DataverseFile
) -> None:
    """Signed URLs expire after about an hour, so they cannot be cached."""
    client = FakeClient(payload, fail_first=1)
    download_file(client, spec, tmp_path, GUESTBOOK, show_progress=False)
    assert client.signed_url_calls == 2


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def test_plan_listing_does_not_repeat_files(capsys) -> None:
    """With six or fewer files the head and tail slices used to overlap.

    A --limit 1 dry run printed the same file twice, which looks like a
    duplicate download rather than a display quirk.
    """
    from src.config import load_config
    from src.download import _print_plan

    files = [
        DataverseFile(
            file_id=i,
            filename=f"sms-call-internet-mi-2013-11-{i:02d}.txt",
            filesize=1000,
            md5="0" * 32,
            content_type="text/plain",
            restricted=False,
        )
        for i in range(1, 4)
    ]
    config = load_config(auto_env=False)

    _print_plan(files, config, config.paths.raw)
    printed = capsys.readouterr().out
    for entry in files:
        assert printed.count(entry.filename) == 1, entry.filename


def test_plan_listing_elides_the_middle_of_a_long_list(capsys) -> None:
    from src.config import load_config
    from src.download import _print_plan

    files = [
        DataverseFile(
            file_id=i,
            filename=f"sms-call-internet-mi-2013-11-{i:02d}.txt",
            filesize=1000,
            md5="0" * 32,
            content_type="text/plain",
            restricted=False,
        )
        for i in range(1, 21)
    ]
    config = load_config(auto_env=False)

    _print_plan(files, config, config.paths.raw)
    printed = capsys.readouterr().out
    assert "more ..." in printed
    assert printed.count("sms-call-internet") == 6


def test_verify_md5(tmp_path: Path) -> None:
    target = tmp_path / "f.bin"
    target.write_bytes(b"hello")
    digest = hashlib.md5(b"hello").hexdigest()
    assert verify_md5(target, digest) == (True, digest)
    assert verify_md5(target, "0" * 32)[0] is False


def test_verify_md5_accepts_uppercase_expected(tmp_path: Path) -> None:
    target = tmp_path / "f.bin"
    target.write_bytes(b"hello")
    assert verify_md5(target, hashlib.md5(b"hello").hexdigest().upper())[0] is True


def test_disk_space_check_rejects_impossible_transfer(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="insufficient disk space"):
        ensure_disk_space(tmp_path, required_bytes=10**15)


def test_disk_space_check_passes_for_a_small_transfer(tmp_path: Path) -> None:
    ensure_disk_space(tmp_path, required_bytes=1024, margin=0)


def test_manifest_merges_across_runs(tmp_path: Path) -> None:
    """A --limit run must not erase the record of previously fetched files."""
    manifest = tmp_path / "manifest.json"
    write_manifest(
        manifest,
        [{"filename": "a.txt", "size_bytes": 10, "verified": True}],
        extra={"doi_traffic": "doi:x"},
    )
    write_manifest(
        manifest,
        [{"filename": "b.txt", "size_bytes": 20, "verified": True}],
        extra={"doi_traffic": "doi:x"},
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert [f["filename"] for f in payload["files"]] == ["a.txt", "b.txt"]
    assert payload["n_files"] == 2
    assert payload["total_bytes"] == 30


def test_manifest_updates_an_existing_record(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, [{"filename": "a.txt", "size_bytes": 10, "verified": False}], extra={})
    write_manifest(manifest, [{"filename": "a.txt", "size_bytes": 10, "verified": True}], extra={})

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["n_files"] == 1
    assert payload["files"][0]["verified"] is True
