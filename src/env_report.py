"""CLI entry point: record the execution environment.

Run at the start of any experimental session so the timings reported later can
be attributed to specific hardware:

    python -m src.env_report --config config/default.yaml
"""

from __future__ import annotations

import argparse
import json

from src.config import load_config
from src.timing import describe_environment, record_hardware


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a config YAML. Defaults to config/default.yaml, with "
        "config/kaggle.yaml layered on automatically inside a Kaggle session.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the full environment snapshot rather than the one-line summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Record hardware and library versions, then report where they were written."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    config.paths.mkdirs()

    env = record_hardware(config.paths.environment_json)

    print(f"environment: {config.env}  (config: {', '.join(str(s) for s in config.source_files)})")
    if args.print_json:
        print(json.dumps(env, indent=2))
    else:
        print(describe_environment(env))
    print(f"written to {config.paths.environment_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
