"""Baseline comparison script for ARIA-Audit.

Runs four scorers (ARIA, Granite Guardian, RAGAS, LlamaGuard) over an eval
suite JSONL file and writes two CSVs plus a headline summary table.

Usage
-----
    python eval/compare_baselines.py --suite bbq_subset [--out eval/results] \\
        [--model qwen3:8b-q4_K_M]

Suite file format (one JSON object per line)::

    {
        "item_id":       "bbq_001",
        "prompt":        "…",
        "response":      "…",
        "context":       "…",
        "group":         "gender"
    }

Output files
------------
``{out_dir}/headline_table.csv``
    Mean scores per axis per scorer — the comparison "figure" for the paper.

``{out_dir}/per_item_scores.csv``
    One row per eval item with all raw scores for post-hoc analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(vals: list[float]) -> float | None:
    """Return mean of vals, ignoring sentinel −1.0 entries, or None if empty."""
    valid = [v for v in vals if v >= 0.0]
    return sum(valid) / len(valid) if valid else None


def _fmt(v: float | None, decimals: int = 4) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def _load_suite(suite_path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with suite_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed JSON on line {lineno}: {exc}", file=sys.stderr)
    return items


def _resolve_suite(suite: str) -> Path:
    """Accept a bare suite name (e.g. 'bbq_subset') or a full path."""
    p = Path(suite)
    if p.exists():
        return p
    # Try eval/suites/<suite>.jsonl relative to the script's repo root
    # (two levels up from eval/)
    script_dir = Path(__file__).parent
    alt = script_dir / "suites" / (suite if suite.endswith(".jsonl") else suite + ".jsonl")
    if alt.exists():
        return alt
    raise FileNotFoundError(
        f"Eval suite not found: {suite!r}  (also tried {alt})"
    )


# ---------------------------------------------------------------------------
# Per-scorer wrappers with timing
# ---------------------------------------------------------------------------

def _run_aria(
    prompt: str,
    response: str,
    context: str,
    model_name: str,
) -> tuple[dict[str, float], float]:
    """Return (scores_dict, latency_ms). scores contain -1.0 on failure."""
    fallback: dict[str, float] = {
        "composite": -1.0,
        "faithfulness": -1.0,
        "equity_di": -1.0,
        "calibration_conf": -1.0,
        "attribution_jaccard": -1.0,
    }
    t0 = time.perf_counter()
    try:
        from aria_audit.orchestrator import audit  # noqa: PLC0415

        retrieved_chunks: list[tuple[str, str]] | None = None
        if context.strip():
            retrieved_chunks = [("ctx", context)]

        # Offline eval mode: generate_fn just returns the pre-recorded response.
        def _offline_generate(_p: str) -> str:
            return response

        envelope = audit(
            prompt=prompt,
            response=response,
            model_name=model_name,
            generate_fn=_offline_generate,
            retrieved_chunks=retrieved_chunks,
            run_calibration=True,
            run_faithfulness=True,
            run_consistency=True,
            run_equity=False,   # expensive; skip for batch comparison
            run_attribution=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "composite": envelope.composite_score,
            "faithfulness": (
                envelope.faithfulness.hhem_score if envelope.faithfulness else -1.0
            ),
            "equity_di": (
                envelope.equity.disparate_impact if envelope.equity else -1.0
            ),
            "calibration_conf": (
                envelope.calibration.confidence if envelope.calibration else -1.0
            ),
            "attribution_jaccard": (
                envelope.attribution.jaccard_at_k if envelope.attribution else -1.0
            ),
        }, latency_ms
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[WARN] ARIA audit raised: {exc}", file=sys.stderr)
        return fallback, latency_ms


def _run_granite_guardian(
    prompt: str,
    response: str,
    context: str,
) -> tuple[dict[str, float], float]:
    fallback: dict[str, float] = {"safety": -1.0, "groundedness": -1.0}
    t0 = time.perf_counter()
    try:
        from aria_audit.baselines.granite_guardian import score as gg_score  # noqa: PLC0415
        ctx_arg = context if context.strip() else None
        result = gg_score(prompt, response, context=ctx_arg)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "safety": result.get("safety", -1.0),
            "groundedness": result.get("groundedness", -1.0),
        }, latency_ms
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[WARN] Granite Guardian raised: {exc}", file=sys.stderr)
        return fallback, latency_ms


def _run_ragas(
    prompt: str,
    response: str,
    context: str,
) -> tuple[dict[str, float], float]:
    fallback: dict[str, float] = {"faithfulness": -1.0, "answer_relevancy": -1.0}
    t0 = time.perf_counter()
    try:
        from aria_audit.baselines.ragas_wrapper import score as ragas_score  # noqa: PLC0415
        result = ragas_score(response, context, question=prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "faithfulness": result.get("faithfulness", -1.0),
            "answer_relevancy": result.get("answer_relevancy", -1.0),
        }, latency_ms
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[WARN] RAGAS raised: {exc}", file=sys.stderr)
        return fallback, latency_ms


def _run_llamaguard(
    prompt: str,
    response: str,
) -> tuple[dict[str, float], float]:
    fallback: dict[str, float] = {"safety": -1.0}
    t0 = time.perf_counter()
    try:
        from aria_audit.baselines.llamaguard import score as lg_score  # noqa: PLC0415
        result = lg_score(prompt, response)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "safety": result.get("safety", -1.0),
        }, latency_ms
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[WARN] LlamaGuard raised: {exc}", file=sys.stderr)
        return fallback, latency_ms


# ---------------------------------------------------------------------------
# Headline table printer
# ---------------------------------------------------------------------------

def _print_headline_table(
    aria_rows: list[dict[str, float]],
    gg_rows: list[dict[str, float]],
    ragas_rows: list[dict[str, float]],
    lg_rows: list[dict[str, float]],
    latencies: dict[str, list[float]],
) -> None:
    aria_faith = _mean([r["faithfulness"] for r in aria_rows])
    gg_ground  = _mean([r["groundedness"] for r in gg_rows])
    gg_safe    = _mean([r["safety"] for r in gg_rows])
    ragas_faith = _mean([r["faithfulness"] for r in ragas_rows])
    ragas_rel   = _mean([r["answer_relevancy"] for r in ragas_rows])
    lg_safe     = _mean([r["safety"] for r in lg_rows])
    aria_ece    = _mean([r["calibration_conf"] for r in aria_rows])
    aria_di     = _mean([r["equity_di"] for r in aria_rows])
    aria_attr   = _mean([r["attribution_jaccard"] for r in aria_rows])

    col_w = [28, 10, 10, 10, 12]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"

    def _row(*cells: str) -> str:
        parts = []
        for cell, w in zip(cells, col_w):
            parts.append(f" {cell:<{w - 2}} ")
        return "|" + "|".join(parts) + "|"

    print()
    print("=" * 72)
    print("  ARIA-Audit vs Baselines — Headline Comparison Table")
    print("=" * 72)
    print(sep)
    print(_row("Axis", "ARIA", "GG", "RAGAS", "LlamaGuard"))
    print(sep)
    print(_row(
        "Faithfulness",
        _fmt(aria_faith), _fmt(gg_ground, 4) + "*", _fmt(ragas_faith), "N/A",
    ))
    print(_row(
        "Safety/Groundedness",
        "N/A", _fmt(gg_safe), "N/A", _fmt(lg_safe),
    ))
    print(_row(
        "Calibration ECE†",
        _fmt(aria_ece), "N/A", "N/A", "N/A",
    ))
    print(_row(
        "Equity DI†",
        _fmt(aria_di), "N/A", "N/A", "N/A",
    ))
    print(_row(
        "Attribution Jaccard†",
        _fmt(aria_attr), "N/A", "N/A", "N/A",
    ))
    print(_row(
        "Answer Relevancy",
        "N/A", "N/A", _fmt(ragas_rel), "N/A",
    ))
    print(sep)
    print()
    print("  * GG groundedness used as proxy for faithfulness column.")
    print("  † Axes ARIA covers that no baseline covers: Calibration ECE,")
    print("    Equity Disparate Impact, Attribution Jaccard stability.")
    print()

    # Latency comparison
    print("-" * 44)
    print(f"  {'Scorer':<20} {'Mean latency (ms)':>20}")
    print("-" * 44)
    for scorer, lats in latencies.items():
        mean_lat = sum(lats) / len(lats) if lats else 0.0
        print(f"  {scorer:<20} {mean_lat:>20.1f}")
    print("-" * 44)
    print()


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

_PER_ITEM_FIELDS = [
    "item_id", "group",
    "aria_composite", "aria_faithfulness", "aria_equity_di",
    "aria_calibration_conf", "aria_attribution_jaccard",
    "gg_safety", "gg_groundedness",
    "ragas_faithfulness", "ragas_relevancy",
    "llamaguard_safety",
]

_HEADLINE_FIELDS = [
    "axis", "aria", "granite_guardian", "ragas", "llamaguard",
]


def _write_per_item_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_PER_ITEM_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_headline_csv(
    path: Path,
    aria_rows: list[dict[str, float]],
    gg_rows: list[dict[str, float]],
    ragas_rows: list[dict[str, float]],
    lg_rows: list[dict[str, float]],
) -> None:
    aria_faith  = _fmt(_mean([r["faithfulness"] for r in aria_rows]))
    aria_ece    = _fmt(_mean([r["calibration_conf"] for r in aria_rows]))
    aria_di     = _fmt(_mean([r["equity_di"] for r in aria_rows]))
    aria_attr   = _fmt(_mean([r["attribution_jaccard"] for r in aria_rows]))
    gg_ground   = _fmt(_mean([r["groundedness"] for r in gg_rows]))
    gg_safe     = _fmt(_mean([r["safety"] for r in gg_rows]))
    ragas_faith = _fmt(_mean([r["faithfulness"] for r in ragas_rows]))
    ragas_rel   = _fmt(_mean([r["answer_relevancy"] for r in ragas_rows]))
    lg_safe     = _fmt(_mean([r["safety"] for r in lg_rows]))

    headline_rows = [
        {"axis": "Faithfulness",         "aria": aria_faith,  "granite_guardian": gg_ground, "ragas": ragas_faith, "llamaguard": "N/A"},
        {"axis": "Safety/Groundedness",  "aria": "N/A",       "granite_guardian": gg_safe,   "ragas": "N/A",       "llamaguard": lg_safe},
        {"axis": "Calibration ECE",      "aria": aria_ece,    "granite_guardian": "N/A",     "ragas": "N/A",       "llamaguard": "N/A"},
        {"axis": "Equity DI",            "aria": aria_di,     "granite_guardian": "N/A",     "ragas": "N/A",       "llamaguard": "N/A"},
        {"axis": "Attribution Jaccard",  "aria": aria_attr,   "granite_guardian": "N/A",     "ragas": "N/A",       "llamaguard": "N/A"},
        {"axis": "Answer Relevancy",     "aria": "N/A",       "granite_guardian": "N/A",     "ragas": ragas_rel,   "llamaguard": "N/A"},
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_HEADLINE_FIELDS)
        writer.writeheader()
        writer.writerows(headline_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare ARIA-Audit against Granite Guardian, RAGAS, and LlamaGuard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--suite",  required=True,
                        help="Suite name (e.g. bbq_subset) or path to .jsonl file.")
    parser.add_argument("--out",    default="eval/results",
                        help="Output directory for CSV files.")
    parser.add_argument("--model",  default="qwen3:8b-q4_K_M",
                        help="Model name forwarded to ARIA audit (for logging).")
    args = parser.parse_args(argv)

    # Resolve suite path
    try:
        suite_path = _resolve_suite(args.suite)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    # Load items
    items = _load_suite(suite_path)
    if not items:
        print(f"[ERROR] No valid items found in {suite_path}", file=sys.stderr)
        return 1

    print(f"Loaded {len(items)} items from {suite_path}")

    # Prepare output dir
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accumulators
    per_item_csv_rows: list[dict[str, Any]] = []
    aria_all:  list[dict[str, float]] = []
    gg_all:    list[dict[str, float]] = []
    ragas_all: list[dict[str, float]] = []
    lg_all:    list[dict[str, float]] = []
    latencies: dict[str, list[float]] = {
        "ARIA": [], "Granite Guardian": [], "RAGAS": [], "LlamaGuard": []
    }

    n = len(items)
    for idx, item in enumerate(items, 1):
        item_id  = str(item.get("item_id", idx))
        group    = str(item.get("group", ""))
        prompt   = str(item.get("prompt", ""))
        response = str(item.get("response", ""))
        context  = str(item.get("context", ""))

        print(f"  [{idx:>3}/{n}] {item_id} ({group})", end="  ", flush=True)

        # --- ARIA ---
        aria_scores, aria_lat = _run_aria(prompt, response, context, args.model)
        latencies["ARIA"].append(aria_lat)
        aria_all.append(aria_scores)
        print(f"ARIA {aria_lat:.0f}ms", end="  ", flush=True)

        # --- Granite Guardian ---
        gg_scores, gg_lat = _run_granite_guardian(prompt, response, context)
        latencies["Granite Guardian"].append(gg_lat)
        gg_all.append(gg_scores)
        print(f"GG {gg_lat:.0f}ms", end="  ", flush=True)

        # --- RAGAS ---
        ragas_scores, ragas_lat = _run_ragas(prompt, response, context)
        latencies["RAGAS"].append(ragas_lat)
        ragas_all.append(ragas_scores)
        print(f"RAGAS {ragas_lat:.0f}ms", end="  ", flush=True)

        # --- LlamaGuard ---
        lg_scores, lg_lat = _run_llamaguard(prompt, response)
        latencies["LlamaGuard"].append(lg_lat)
        lg_all.append(lg_scores)
        print(f"LG {lg_lat:.0f}ms", flush=True)

        # Assemble per-item row
        def _s(v: float) -> str:
            return "N/A" if v < 0.0 else f"{v:.4f}"

        per_item_csv_rows.append({
            "item_id":                item_id,
            "group":                  group,
            "aria_composite":         _s(aria_scores["composite"]),
            "aria_faithfulness":      _s(aria_scores["faithfulness"]),
            "aria_equity_di":         _s(aria_scores["equity_di"]),
            "aria_calibration_conf":  _s(aria_scores["calibration_conf"]),
            "aria_attribution_jaccard": _s(aria_scores["attribution_jaccard"]),
            "gg_safety":              _s(gg_scores["safety"]),
            "gg_groundedness":        _s(gg_scores["groundedness"]),
            "ragas_faithfulness":     _s(ragas_scores["faithfulness"]),
            "ragas_relevancy":        _s(ragas_scores["answer_relevancy"]),
            "llamaguard_safety":      _s(lg_scores["safety"]),
        })

    # --- Print headline table ---
    _print_headline_table(aria_all, gg_all, ragas_all, lg_all, latencies)

    # --- Write CSVs ---
    headline_path  = out_dir / "headline_table.csv"
    per_item_path  = out_dir / "per_item_scores.csv"

    _write_headline_csv(headline_path, aria_all, gg_all, ragas_all, lg_all)
    _write_per_item_csv(per_item_path, per_item_csv_rows)

    print(f"Saved: {headline_path}")
    print(f"Saved: {per_item_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
