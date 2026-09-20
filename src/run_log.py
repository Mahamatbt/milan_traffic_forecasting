"""Extract the per-day ingest log from an executed notebook into a table.

The streaming pipeline prints one line per day as it runs. On Kaggle that
output is the only surviving record of how long the 19.4 GiB download took and
how much of each day was missing, because the scratch directory is discarded
when the session ends.

Parsing it back into a CSV rather than transcribing figures by hand keeps the
report's numbers tied to the run that produced them, and makes a mis-copied
digit impossible.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = ["DayRecord", "parse_notebook", "write_log", "summarise"]

# [ 1/62] 2013-11-01  dl    5.7s  ingest   1.6s  4,842,625 rows  absent  18  peak   574 MiB
_DAY_LINE = re.compile(
    r"\[\s*(?P<n>\d+)/(?P<total>\d+)\]\s+"
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"dl\s+(?P<dl>[\d.]+)s\s+"
    r"ingest\s+(?P<ingest>[\d.]+)s\s+"
    r"(?P<rows>[\d,]+)\s+rows\s+"
    r"absent\s+(?P<absent>\d+)\s+"
    r"peak\s+(?P<peak>\d+)\s+MiB"
)

CELLS_PER_DAY = 1_440_000  # 144 intervals x 10,000 squares


@dataclass(frozen=True)
class DayRecord:
    """One day's line from the pipeline log."""

    date: str
    download_s: float
    ingest_s: float
    raw_rows: int
    absent_cells: int
    peak_rss_mib: int

    @property
    def absent_fraction(self) -> float:
        return self.absent_cells / CELLS_PER_DAY

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["absent_fraction"] = round(self.absent_fraction, 8)
        row["total_s"] = round(self.download_s + self.ingest_s, 2)
        return row


def _notebook_text(path: Path) -> str:
    """Concatenate every stream output in an executed notebook."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        for output in cell.get("outputs", []):
            if "text" in output:
                chunks.append("".join(output["text"]))
            elif "data" in output and "text/plain" in output["data"]:
                chunks.append("".join(output["data"]["text/plain"]))
    return "".join(chunks)


def parse_notebook(path: Path) -> list[DayRecord]:
    """Recover every per-day record from an executed notebook.

    Args:
        path: An executed ``.ipynb`` containing the pipeline's output.

    Returns:
        Records in date order.

    Raises:
        ValueError: If no day lines are found, which means the notebook was
            never run or its outputs were stripped.
    """
    matches = list(_DAY_LINE.finditer(_notebook_text(path)))
    if not matches:
        raise ValueError(
            f"{path} contains no pipeline day lines; was it executed with outputs kept?"
        )
    records = [
        DayRecord(
            date=m["date"],
            download_s=float(m["dl"]),
            ingest_s=float(m["ingest"]),
            raw_rows=int(m["rows"].replace(",", "")),
            absent_cells=int(m["absent"]),
            peak_rss_mib=int(m["peak"]),
        )
        for m in matches
    ]
    return sorted(records, key=lambda r: r.date)


def summarise(records: list[DayRecord]) -> dict[str, Any]:
    """Aggregate the run into the figures the report quotes."""
    import statistics

    download = [r.download_s for r in records]
    ingest = [r.ingest_s for r in records]
    absent = [r.absent_cells for r in records]

    return {
        "n_days": len(records),
        "first_day": records[0].date,
        "last_day": records[-1].date,
        "total_raw_rows": sum(r.raw_rows for r in records),
        "download_total_min": round(sum(download) / 60, 2),
        "download_mean_s": round(statistics.fmean(download), 2),
        "download_std_s": round(statistics.stdev(download), 2) if len(download) > 1 else 0.0,
        "ingest_total_min": round(sum(ingest) / 60, 2),
        "ingest_mean_s": round(statistics.fmean(ingest), 2),
        "ingest_std_s": round(statistics.stdev(ingest), 2) if len(ingest) > 1 else 0.0,
        "peak_rss_max_mib": max(r.peak_rss_mib for r in records),
        "peak_rss_mean_mib": round(statistics.fmean([r.peak_rss_mib for r in records]), 1),
        "absent_total": sum(absent),
        "absent_mean_per_day": round(statistics.fmean(absent), 1),
        "absent_max_per_day": max(absent),
        "absent_max_day": max(records, key=lambda r: r.absent_cells).date,
    }


def write_log(records: list[DayRecord], destination: Path) -> Path:
    """Write the per-day table as CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_row() for r in records]
    with destination.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "notebook",
        nargs="?",
        default="notebooks/runs/00_kaggle_ingest.executed.ipynb",
        help="Executed notebook to parse.",
    )
    parser.add_argument("--config", default=None, help="Path to a config YAML.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse an executed notebook and write the run tables."""
    from src.config import load_config

    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    config.paths.mkdirs()

    records = parse_notebook(Path(args.notebook))
    log_path = write_log(records, config.paths.tables / "ingest_log.csv")
    summary = summarise(records)

    summary_path = config.paths.tables / "ingest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for key, value in summary.items():
        print(f"  {key:<22}: {value}")
    print(f"\nwrote {log_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
