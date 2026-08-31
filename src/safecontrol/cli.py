from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .pipeline import check_reproduction, reproduce
from .provenance import verify_repository


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safecontrol",
        description="Reproduce and verify the scalar robust neural-control certificate.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    reproduce_parser = subparsers.add_parser(
        "reproduce", help="generate canonical data, artifacts, and results"
    )
    reproduce_parser.add_argument("--root", default=".")
    reproduce_parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in a temporary directory and compare exact bytes",
    )
    verify_parser = subparsers.add_parser(
        "verify", help="verify hashes, bindings, benchmark gates, and certificate"
    )
    verify_parser.add_argument("--root", default=".")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = Path(args.root).resolve()
        if args.command == "reproduce":
            result = check_reproduction(root) if args.check else reproduce(root)
        else:
            config = load_config(root / "config" / "benchmark.json")
            result = verify_repository(root, config)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
