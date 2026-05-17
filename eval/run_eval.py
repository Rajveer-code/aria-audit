"""Direct runner for eval suites. Use `aria-audit run` for full pipeline."""

from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_eval",
        description=__doc__,
    )
    p.add_argument(
        "--model",
        required=True,
        help="Ollama model tag (e.g. qwen3:8b-q4_K_M).",
    )
    p.add_argument(
        "--eval",
        dest="eval_suite",
        required=True,
        help="Path to the JSONL eval suite file, or a suite name under eval/suites/.",
    )
    p.add_argument(
        "--out",
        default="eval/results",
        help="Output directory for SQLite DB and CSV summary (default: eval/results).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from aria_audit.orchestrator import run_eval  # noqa: PLC0415

    return run_eval(
        model=args.model,
        eval_suite=args.eval_suite,
        out_dir=args.out,
    )


if __name__ == "__main__":
    sys.exit(main())
