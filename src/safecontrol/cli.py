from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .pipeline import check_reproduction, check_semantic_reproduction, reproduce
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
        help="regenerate in a temporary directory and compare exact bytes (canonical environment)",
    )
    reproduce_parser.add_argument(
        "--semantic-check",
        action="store_true",
        help="regenerate and compare scientific values within the cross-platform tolerance",
    )
    verify_parser = subparsers.add_parser(
        "verify", help="verify hashes, bindings, benchmark gates, and certificate"
    )
    verify_parser.add_argument("--root", default=".")
    sweep_parser = subparsers.add_parser("rho-sweep", help="run the fixed-radius contraction study")
    sweep_parser.add_argument("--root", default=".")
    sweep_parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = Path(args.root).resolve()
        if args.command == "reproduce":
            if args.check and args.semantic_check:
                raise ValueError("--check and --semantic-check are mutually exclusive")
            if args.check:
                result = check_reproduction(root)
            elif args.semantic_check:
                result = check_semantic_reproduction(root)
            else:
                result = reproduce(root)
        elif args.command == "rho-sweep":
            from .rho_sweep import check_sweep, run_sweep
            result = check_sweep(root) if args.check else run_sweep(root)
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
