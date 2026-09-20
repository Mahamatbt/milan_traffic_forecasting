"""Download the Telecom Italia CDR dataset from Harvard Dataverse.

The dataset is ~19.4 GiB across 62 daily files, so this module is built around
the assumption that a download will be interrupted at least once:

* Every file streams to a ``.part`` sidecar and is renamed only after its MD5
  matches the checksum Dataverse publishes. A half-written file can therefore
  never be mistaken for a complete one.
* Resume uses HTTP ``Range``. The S3 backend advertises ``Accept-Ranges: bytes``
  and returns ``206 Partial Content``, so an interrupted transfer continues from
  the byte it stopped at rather than restarting.
* The file list is discovered from the Dataverse API rather than hardcoded, so
  a change upstream surfaces as a mismatch instead of a silent gap.

Guestbook
---------
This dataset has a Dataverse Guestbook attached (``guestbookId`` 96), so a plain
``GET`` on the access endpoint returns HTTP 400. The documented route is to POST
a guestbook response to the same endpoint, which returns a short-lived signed
URL. No API token is required, but a response *is* recorded against the
repository each time -- it is the data provider's record of who used their data.
Identity therefore comes from the caller (CLI flags or environment variables)
and is never invented or defaulted to a placeholder.

Signed URLs expire after roughly an hour, so each retry requests a fresh one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from src.config import PROJECT_ROOT, Config, load_config

__all__ = [
    "DataverseFile",
    "GuestbookResponse",
    "DataverseClient",
    "download_file",
    "verify_md5",
    "load_dotenv",
]

# Read in 8 MiB chunks: large enough that per-chunk overhead is negligible on a
# 350 MB file, small enough that memory stays flat regardless of file size.
CHUNK_SIZE = 8 * 1024 * 1024

# Refuse to start unless the full dataset plus this much slack will fit.
DISK_MARGIN_BYTES = 5 * 1024**3

# Harvard Dataverse sits behind a WAF that rejects the default
# "python-requests/x.y.z" User-Agent with HTTP 403. A descriptive agent is both
# what gets through and what politely identifies this client to the repository.
USER_AGENT = "milan_traffic_forecasting/0.1 (academic coursework; python-requests)"

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 5
_BACKOFF_BASE_S = 2.0


def _backoff_sleep(attempt: int) -> None:
    """Wait before retry ``attempt`` (1-based), doubling each time.

    Routed through one function so the test suite can neutralise it; otherwise
    exercising the retry paths would cost 30 s of real sleeping.
    """
    time.sleep(_BACKOFF_BASE_S * (2 ** (attempt - 1)))


class DownloadError(RuntimeError):
    """Raised when a file cannot be retrieved or fails verification."""


# --------------------------------------------------------------------------
# Value types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DataverseFile:
    """One file as described by the Dataverse file-listing API."""

    file_id: int
    filename: str
    filesize: int
    md5: str
    content_type: str
    restricted: bool

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> DataverseFile:
        """Build from one element of the API's ``data`` array.

        The nesting is verified rather than assumed: the payload puts the
        identity fields under ``dataFile`` while ``restricted`` sits on the
        outer object.
        """
        try:
            data_file = entry["dataFile"]
        except KeyError as exc:
            raise DownloadError(
                f"unexpected API entry shape, no 'dataFile' key: {sorted(entry)}"
            ) from exc

        missing = [k for k in ("id", "filename", "filesize", "md5") if k not in data_file]
        if missing:
            raise DownloadError(
                f"API entry for {data_file.get('filename', '?')} is missing {missing}"
            )

        return cls(
            file_id=int(data_file["id"]),
            filename=str(data_file["filename"]),
            filesize=int(data_file["filesize"]),
            md5=str(data_file["md5"]).lower(),
            content_type=str(data_file.get("contentType", "")),
            restricted=bool(entry.get("restricted", False)),
        )


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Load ``KEY=value`` pairs from a local ``.env`` into the environment.

    Guestbook identity is personal data, so it belongs in a gitignored file
    rather than in the repository, a shell history or a committed config.
    Variables already present in the environment win, which lets Kaggle Secrets
    or a CI secret store override the file without editing it.

    Args:
        path: The file to read. Defaults to ``.env`` beside the project root.

    Returns:
        The keys that were actually set from the file.
    """
    env_path = path or (PROJECT_ROOT / ".env")
    applied: dict[str, str] = {}
    if not env_path.exists():
        return applied

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


@dataclass(frozen=True)
class GuestbookResponse:
    """Identity submitted to the dataset's Guestbook.

    Dataverse records one response per download request. These values are the
    data provider's record of who used their data, so they are supplied
    explicitly by the caller rather than defaulted to a placeholder.
    """

    name: str
    email: str
    institution: str
    position: str

    def __post_init__(self) -> None:
        blank = [
            f for f in ("name", "email", "institution", "position") if not getattr(self, f).strip()
        ]
        if blank:
            raise ValueError(f"guestbook response fields must not be blank: {blank}")

    def to_payload(self) -> dict[str, Any]:
        """Render as the JSON body the access endpoint expects."""
        return {"guestbookResponse": asdict(self)}

    @classmethod
    def from_args_or_env(cls, args: argparse.Namespace) -> GuestbookResponse:
        """Resolve from CLI flags, falling back to ``DATAVERSE_GB_*`` env vars.

        Raises:
            DownloadError: If any field is unset, with the flag and variable
                names that would satisfy it.
        """
        fields = {
            "name": (args.gb_name, "DATAVERSE_GB_NAME", "--gb-name"),
            "email": (args.gb_email, "DATAVERSE_GB_EMAIL", "--gb-email"),
            "institution": (args.gb_institution, "DATAVERSE_GB_INSTITUTION", "--gb-institution"),
            "position": (args.gb_position, "DATAVERSE_GB_POSITION", "--gb-position"),
        }
        resolved: dict[str, str] = {}
        unset: list[str] = []
        for key, (cli_value, env_var, flag) in fields.items():
            value = cli_value or os.environ.get(env_var, "")
            if not value.strip():
                unset.append(f"{flag} (or ${env_var})")
            resolved[key] = value.strip()

        if unset:
            raise DownloadError(
                "This dataset requires a Guestbook response before files can be "
                "downloaded, and the response is recorded by the repository.\n"
                "Supply your real details via:\n  " + "\n  ".join(unset)
            )
        return cls(**resolved)


@dataclass
class DownloadResult:
    """Outcome of one file transfer."""

    filename: str
    file_id: int
    path: Path
    size_bytes: int
    md5: str
    verified: bool
    skipped: bool
    resumed_from: int
    wall_s: float
    attempts: int

    def to_record(self) -> dict[str, Any]:
        """Flatten for the manifest."""
        record = asdict(self)
        record["path"] = str(self.path)
        record["downloaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return record


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class DataverseClient:
    """Minimal Dataverse client covering file listing and guestbook-gated access."""

    def __init__(
        self,
        base_url: str,
        *,
        session: requests.Session | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        # setdefault would be a no-op here: requests.Session populates
        # User-Agent itself, so the blocked default has to be displaced. A
        # caller that deliberately set its own agent keeps it.
        current = self.session.headers.get("User-Agent", "")
        if not current or current == requests.utils.default_user_agent():
            self.session.headers["User-Agent"] = USER_AGENT

    def list_files(self, doi: str) -> list[DataverseFile]:
        """List every file in the latest published version of a dataset.

        Args:
            doi: Persistent identifier, e.g. ``doi:10.7910/DVN/EGZHFV``.

        Returns:
            One :class:`DataverseFile` per entry, in API order.

        Raises:
            DownloadError: On a non-OK response or an unexpected payload shape.
        """
        url = f"{self.base_url}/api/datasets/:persistentId/versions/:latest/files"
        response = self._get_with_retry(url, params={"persistentId": doi})
        payload = response.json()

        if payload.get("status") != "OK":
            raise DownloadError(f"Dataverse returned status {payload.get('status')!r} for {doi}")
        entries = payload.get("data")
        if not isinstance(entries, list):
            raise DownloadError(
                f"expected 'data' to be a list for {doi}, got {type(entries).__name__}"
            )
        return [DataverseFile.from_entry(entry) for entry in entries]

    def signed_url(self, file_id: int, guestbook: GuestbookResponse) -> str:
        """Exchange a guestbook response for a short-lived download URL.

        Args:
            file_id: Numeric Dataverse datafile id.
            guestbook: Identity recorded against the download.

        Returns:
            A signed URL valid for roughly one hour.

        Raises:
            DownloadError: If the endpoint does not return a signed URL.
        """
        url = f"{self.base_url}/api/access/datafile/{file_id}"
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self.session.post(
                    url,
                    json=guestbook.to_payload(),
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"},
                )
            except requests.RequestException as exc:
                if attempt == _MAX_ATTEMPTS:
                    raise DownloadError(f"signed URL request failed for {file_id}: {exc}") from exc
                _backoff_sleep(attempt)
                continue

            if response.status_code in _RETRY_STATUS and attempt < _MAX_ATTEMPTS:
                _backoff_sleep(attempt)
                continue

            if response.status_code != 200:
                raise DownloadError(
                    f"signed URL request for file {file_id} returned "
                    f"HTTP {response.status_code}: {response.text[:300]}"
                )

            signed = response.json().get("data", {}).get("signedUrl")
            if not signed:
                raise DownloadError(f"no signedUrl in response for file {file_id}")
            return str(signed)

        raise DownloadError(f"exhausted retries requesting signed URL for file {file_id}")

    def _get_with_retry(self, url: str, **kwargs: Any) -> requests.Response:
        """GET with exponential backoff on transient failures."""
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self.session.get(url, timeout=self.timeout, **kwargs)
                if response.status_code in _RETRY_STATUS and attempt < _MAX_ATTEMPTS:
                    _backoff_sleep(attempt)
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt == _MAX_ATTEMPTS:
                    break
                _backoff_sleep(attempt)
        raise DownloadError(f"GET {url} failed after {_MAX_ATTEMPTS} attempts: {last_error}")

    def open_stream(self, url: str, *, start_byte: int = 0) -> tuple[requests.Response, bool]:
        """Open a streaming GET, optionally resuming from ``start_byte``.

        Returns:
            The response and whether the server honoured the range request. A
            server that ignores ``Range`` returns 200 with the whole body, in
            which case the caller must restart from zero rather than append.
        """
        headers = {"Range": f"bytes={start_byte}-"} if start_byte else {}
        response = self.session.get(url, stream=True, timeout=self.timeout, headers=headers)
        response.raise_for_status()
        honoured = response.status_code == 206
        return response, honoured


# --------------------------------------------------------------------------
# Transfer
# --------------------------------------------------------------------------


def verify_md5(path: Path, expected: str, *, chunk_size: int = CHUNK_SIZE) -> tuple[bool, str]:
    """Hash a file and compare against an expected MD5.

    Args:
        path: File to hash.
        expected: Lowercase hex digest from the Dataverse metadata.
        chunk_size: Read size; keeps memory flat on multi-hundred-MB files.

    Returns:
        ``(matches, actual_digest)``.
    """
    digest = hashlib.md5()  # noqa: S324 - matching the checksum Dataverse publishes
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    actual = digest.hexdigest()
    return actual == expected.lower(), actual


def _hash_existing(path: Path, chunk_size: int = CHUNK_SIZE) -> hashlib._Hash:
    """Seed an MD5 accumulator with the bytes already on disk."""
    digest = hashlib.md5()  # noqa: S324
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest


def ensure_disk_space(
    destination: Path, required_bytes: int, margin: int = DISK_MARGIN_BYTES
) -> None:
    """Abort before starting if the transfer cannot possibly fit.

    Raises:
        DownloadError: If free space is below ``required_bytes + margin``.
    """
    destination.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination).free
    needed = required_bytes + margin
    if free < needed:
        raise DownloadError(
            f"insufficient disk space at {destination}: "
            f"{free / 1024**3:.1f} GiB free, need {needed / 1024**3:.1f} GiB "
            f"({required_bytes / 1024**3:.1f} GiB of data plus "
            f"{margin / 1024**3:.1f} GiB margin)"
        )


def download_file(
    client: DataverseClient,
    file: DataverseFile,
    dest_dir: Path,
    guestbook: GuestbookResponse,
    *,
    chunk_size: int = CHUNK_SIZE,
    show_progress: bool = True,
) -> DownloadResult:
    """Fetch one file with resume, verification and retry.

    A completed file present on disk with a matching MD5 is skipped, so
    re-running the command is a no-op. A partial ``.part`` sidecar is resumed
    from its current length.

    Args:
        client: Configured Dataverse client.
        file: File metadata from :meth:`DataverseClient.list_files`.
        dest_dir: Directory to write into.
        guestbook: Identity recorded against each access request.
        chunk_size: Streaming read size.
        show_progress: Render a per-file tqdm bar.

    Returns:
        A :class:`DownloadResult` describing what happened.

    Raises:
        DownloadError: If the file cannot be retrieved or fails verification.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / file.filename
    part_path = dest_dir / f"{file.filename}.part"
    started = time.perf_counter()

    if final_path.exists() and final_path.stat().st_size == file.filesize:
        matches, actual = verify_md5(final_path, file.md5)
        if matches:
            return DownloadResult(
                filename=file.filename,
                file_id=file.file_id,
                path=final_path,
                size_bytes=file.filesize,
                md5=actual,
                verified=True,
                skipped=True,
                resumed_from=0,
                wall_s=time.perf_counter() - started,
                attempts=0,
            )
        # Present but corrupt: discard rather than trust it.
        final_path.unlink()

    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        resume_from = part_path.stat().st_size if part_path.exists() else 0
        if resume_from > file.filesize:  # sidecar from a different version
            part_path.unlink()
            resume_from = 0

        try:
            written = _stream_to_part(
                client=client,
                file=file,
                guestbook=guestbook,
                part_path=part_path,
                resume_from=resume_from,
                chunk_size=chunk_size,
                show_progress=show_progress,
            )
        except (requests.RequestException, DownloadError) as exc:
            last_error = exc
            if attempt == _MAX_ATTEMPTS:
                break
            _backoff_sleep(attempt)
            continue

        if written != file.filesize:
            last_error = DownloadError(
                f"{file.filename}: wrote {written} bytes, expected {file.filesize}"
            )
            if attempt == _MAX_ATTEMPTS:
                break
            continue

        matches, actual = verify_md5(part_path, file.md5)
        if not matches:
            # A corrupt transfer must not be resumed; start clean next attempt.
            part_path.unlink()
            last_error = DownloadError(
                f"{file.filename}: MD5 mismatch (expected {file.md5}, got {actual})"
            )
            if attempt == _MAX_ATTEMPTS:
                break
            continue

        part_path.replace(final_path)
        return DownloadResult(
            filename=file.filename,
            file_id=file.file_id,
            path=final_path,
            size_bytes=file.filesize,
            md5=actual,
            verified=True,
            skipped=False,
            resumed_from=resume_from,
            wall_s=time.perf_counter() - started,
            attempts=attempt,
        )

    raise DownloadError(f"failed to download {file.filename}: {last_error}")


def _stream_to_part(
    *,
    client: DataverseClient,
    file: DataverseFile,
    guestbook: GuestbookResponse,
    part_path: Path,
    resume_from: int,
    chunk_size: int,
    show_progress: bool,
) -> int:
    """Stream one file to its ``.part`` sidecar, returning the total bytes on disk.

    A fresh signed URL is requested on every call because signed URLs expire
    after about an hour, which is comfortably shorter than a full run.
    """
    url = client.signed_url(file.file_id, guestbook)
    response, ranged = client.open_stream(url, start_byte=resume_from)

    # If the server ignored Range it is sending the whole body, so appending
    # would corrupt the file. Truncate and take it from the top.
    mode = "ab" if (resume_from and ranged) else "wb"
    if resume_from and not ranged:
        resume_from = 0

    progress = _progress_bar(file, resume_from, enabled=show_progress)
    written = resume_from
    try:
        with part_path.open(mode) as fh:
            for block in response.iter_content(chunk_size=chunk_size):
                if not block:
                    continue
                fh.write(block)
                written += len(block)
                if progress is not None:
                    progress.update(len(block))
    finally:
        if progress is not None:
            progress.close()
        response.close()
    return written


def _progress_bar(file: DataverseFile, initial: int, *, enabled: bool) -> Any:
    """Build a tqdm bar, or None when progress is suppressed or tqdm is absent."""
    if not enabled:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(
        total=file.filesize,
        initial=initial,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=file.filename,
        leave=False,
    )


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def write_manifest(path: Path, records: list[dict[str, Any]], *, extra: dict[str, Any]) -> None:
    """Write the download manifest, merging with any existing records.

    Merging matters because a run limited with ``--limit`` must not erase the
    record of files fetched earlier.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            existing = {r["filename"]: r for r in previous.get("files", [])}
        except (json.JSONDecodeError, KeyError, TypeError):
            existing = {}

    for record in records:
        existing[record["filename"]] = record

    merged = sorted(existing.values(), key=lambda r: r["filename"])
    payload = {
        **extra,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_files": len(merged),
        "total_bytes": sum(r["size_bytes"] for r in merged),
        "files": merged,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Download the Milan CDR dataset and Milano Grid from Harvard Dataverse.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The dataset has a Guestbook, so identity details are required and are\n"
            "recorded by the repository. Set them once as environment variables:\n"
            "  DATAVERSE_GB_NAME, DATAVERSE_GB_EMAIL,\n"
            "  DATAVERSE_GB_INSTITUTION, DATAVERSE_GB_POSITION"
        ),
    )
    parser.add_argument("--config", default=None, help="Path to a config YAML.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be fetched and the total bytes, then exit.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Fetch at most N traffic files (for testing)."
    )
    parser.add_argument("--skip-grid", action="store_true", help="Do not fetch the Milano Grid.")
    parser.add_argument(
        "--no-progress", action="store_true", help="Suppress per-file progress bars."
    )
    parser.add_argument("--gb-name", default=None, help="Guestbook: your full name.")
    parser.add_argument("--gb-email", default=None, help="Guestbook: your email address.")
    parser.add_argument("--gb-institution", default=None, help="Guestbook: your institution.")
    parser.add_argument("--gb-position", default=None, help="Guestbook: your position/role.")
    return parser


def _print_plan(files: list[DataverseFile], config: Config, destination: Path) -> int:
    """Render the dry-run plan; returns the total bytes to fetch."""
    total = sum(f.filesize for f in files)
    present = sum(
        1
        for f in files
        if (destination / f.filename).exists()
        and (destination / f.filename).stat().st_size == f.filesize
    )
    free = shutil.disk_usage(destination if destination.exists() else destination.parent).free

    print(f"dataset      : {config.dataset.doi_traffic}")
    print(f"destination  : {destination}")
    print(f"files        : {len(files)}  ({present} already complete on disk)")
    print(f"total size   : {total:,} bytes ({total / 1024**3:.2f} GiB)")
    print(f"free space   : {free / 1024**3:.2f} GiB")
    print(f"restricted   : {[f.filename for f in files if f.restricted] or 'none'}")
    print()

    def show(entry: DataverseFile) -> None:
        print(f"  {entry.file_id:<9} {entry.filename:<40} {entry.filesize:>13,} B  md5={entry.md5}")

    # Elide the middle only when there is a middle to elide; a short list
    # (say --limit 1) must not print the same file as both head and tail.
    if len(files) <= 6:
        for f in files:
            show(f)
    else:
        for f in files[:3]:
            show(f)
        print(f"  ... {len(files) - 6} more ...")
        for f in files[-3:]:
            show(f)
    return total


def main(argv: list[str] | None = None) -> int:
    """Discover, verify and download the dataset."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    destination = config.paths.raw
    grid_dir = destination / "grid"

    client = DataverseClient(config.dataset.dataverse_base)
    print(f"listing files for {config.dataset.doi_traffic} ...", flush=True)
    files = sorted(client.list_files(config.dataset.doi_traffic), key=lambda f: f.filename)
    if args.limit:
        files = files[: args.limit]

    expected = config.dataset.n_days
    if args.limit is None and len(files) != expected:
        print(f"WARNING: expected {expected} files, API listed {len(files)}")

    total = _print_plan(files, config, destination)
    if args.dry_run:
        print("\n(dry run: nothing downloaded)")
        return 0

    load_dotenv()
    guestbook = GuestbookResponse.from_args_or_env(args)
    outstanding = sum(
        f.filesize
        for f in files
        if not (
            (destination / f.filename).exists()
            and (destination / f.filename).stat().st_size == f.filesize
        )
    )
    ensure_disk_space(destination, outstanding)

    print(f"\nguestbook    : {guestbook.name} <{guestbook.email}>, {guestbook.institution}")
    print(f"to fetch     : {outstanding / 1024**3:.2f} GiB of {total / 1024**3:.2f} GiB\n")

    records: list[dict[str, Any]] = []
    for index, file in enumerate(files, start=1):
        result = download_file(
            client, file, destination, guestbook, show_progress=not args.no_progress
        )
        status = "skipped (verified)" if result.skipped else f"ok in {result.wall_s:.1f}s"
        if result.resumed_from:
            status += f", resumed from {result.resumed_from / 1024**2:.0f} MiB"
        print(f"[{index:>2}/{len(files)}] {file.filename}  {status}", flush=True)
        records.append(result.to_record())

    if not args.skip_grid:
        records.extend(_download_grid(client, config, grid_dir, guestbook, args))

    manifest_path = destination / "manifest.json"
    write_manifest(
        manifest_path,
        records,
        extra={
            "doi_traffic": config.dataset.doi_traffic,
            "doi_grid": config.dataset.doi_grid,
            "dataverse_base": config.dataset.dataverse_base,
        },
    )
    verified = sum(1 for r in records if r["verified"])
    print(f"\n{verified}/{len(records)} files verified; manifest at {manifest_path}")
    return 0 if verified == len(records) else 1


def _download_grid(
    client: DataverseClient,
    config: Config,
    grid_dir: Path,
    guestbook: GuestbookResponse,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Fetch the Milano Grid GeoJSON used for the choropleth in Phase 3."""
    print(f"\nlisting files for {config.dataset.doi_grid} ...", flush=True)
    grid_files = client.list_files(config.dataset.doi_grid)
    records = []
    for file in grid_files:
        result = download_file(
            client, file, grid_dir, guestbook, show_progress=not args.no_progress
        )
        print(f"[grid] {file.filename}  {'skipped' if result.skipped else 'ok'}", flush=True)
        records.append(result.to_record())
    return records


if __name__ == "__main__":
    raise SystemExit(main())
