"""Unified entry point for the complete education QA pipeline."""
from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import sys
import httpx

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eduqa", description="教育问答：数据、微调、评测、部署与压测"
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("data", "training", "service", "evaluation", "benchmark", "deployment"):
        importlib.import_module(f"eduqa.{name}").add_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.func(args)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        return result if isinstance(result, int) else 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError, RuntimeError, OSError, httpx.InvalidURL) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
