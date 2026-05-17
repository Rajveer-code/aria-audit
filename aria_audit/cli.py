"""Command-line entry for aria-audit.

Subcommands:
  benchmark   — Phase 0 VRAM gate test + tokens/sec benchmark
  run         — execute audit over an eval suite, emit SQLite + summary CSV
"""

from __future__ import annotations

import argparse
import logging
import sys


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from aria_audit.bench.vram_gate import run_vram_gate
    return run_vram_gate(model=args.model, sample_n=args.samples)


def _cmd_run(args: argparse.Namespace) -> int:
    from aria_audit.orchestrator import run_eval
    return run_eval(model=args.model, eval_suite=args.eval, out_dir=args.out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aria-audit", description=__doc__)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("benchmark", help="Phase 0 VRAM gate test + tokens/sec")
    pb.add_argument("--model", default="qwen3:8b-q4_K_M")
    pb.add_argument("--samples", type=int, default=100)
    pb.set_defaults(func=_cmd_benchmark)

    pr = sub.add_parser("run", help="Run audit over an eval suite")
    pr.add_argument("--model", required=True)
    pr.add_argument("--eval", required=True, help="Path or name of suite under eval/suites/")
    pr.add_argument("--out", default="eval/results")
    pr.set_defaults(func=_cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
