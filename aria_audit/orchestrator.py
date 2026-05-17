"""Single-pass shared-sample audit pipeline.

One set of N=3 samples feeds calibration + consistency + attribution simultaneously.
Then HHEM 2.1 loads once for faithfulness. Equity runs separately (it needs generate_fn).

Sub-second overhead budget: calibration is pure-numpy, consistency reuses HHEM load,
attribution is pure-python, equity is CPU-only (sentiment/refusal). Faithfulness is the
only guaranteed GPU load.

Phase 1 full implementation.
"""

from __future__ import annotations

import csv
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from aria_audit.core import (
    AuditEnvelope,
    CalibrationResult,
    DriftSignal,
)
from aria_audit.drift import PageHinkley

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level drift trackers — one per axis, keyed by axis name.
# These persist for the lifetime of the process so that per-call drift
# detection is meaningful across a session of requests.
# ---------------------------------------------------------------------------
_drift_trackers: dict[str, PageHinkley] = {
    "calibration": PageHinkley(),
    "faithfulness": PageHinkley(),
    "consistency": PageHinkley(),
    "equity": PageHinkley(),
    "attribution": PageHinkley(),
}


# ---------------------------------------------------------------------------
# Ollama generate helper — used by run_eval when no external generate_fn is
# supplied.  Deferred import so the module is importable without `requests`.
# ---------------------------------------------------------------------------

def _make_ollama_generate_fn(model: str) -> Callable[[str], str]:
    """Return a synchronous generate_fn that calls the local Ollama HTTP API."""

    def _generate(prompt: str) -> str:
        import requests  # noqa: PLC0415

        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return str(data.get("response", ""))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama generate failed for model=%r: %s", model, exc)
            return ""

    return _generate


# ---------------------------------------------------------------------------
# Per-axis drift helper
# ---------------------------------------------------------------------------

def _push_drift(axis: str, score: float, t: float) -> DriftSignal:
    """Push *score* to the axis tracker; return a DriftSignal snapshot."""
    tracker = _drift_trackers.setdefault(axis, PageHinkley())
    alarmed = tracker.update(score, t)
    sig = tracker.snapshot(axis, t)
    # snapshot() always returns alarmed=False — override with live result.
    return DriftSignal(
        axis=sig.axis,
        cumsum=sig.cumsum,
        alarmed=alarmed,
        last_reset_at=sig.last_reset_at,
    )


# ---------------------------------------------------------------------------
# Main audit entry-point
# ---------------------------------------------------------------------------

def audit(
    prompt: str,
    response: str,
    model_name: str,
    generate_fn: Callable[[str], str],
    retrieved_chunks: list[tuple[str, str]] | None = None,
    run_calibration: bool = True,
    run_faithfulness: bool = True,
    run_consistency: bool = True,
    run_equity: bool = False,   # expensive — off by default in online serving
    run_attribution: bool = True,
    equity_axis: str = "gender",
    copilot_fn: Callable[[str], str] | None = None,
    request_id: str = "",
    db_logger=None,             # EnvelopeLogger instance or None
    suite_name: str = "",
    suite_item_id: str = "",
) -> AuditEnvelope:
    """Run a full single-pass five-axis audit for one (prompt, response) pair.

    Shared samples
    --------------
    Consistency generates N=3 paraphrased responses from ``generate_fn``.
    Attribution re-uses the first of those paraphrased responses as
    ``response_b`` rather than calling ``generate_fn`` a second time.

    Per-response calibration
    ------------------------
    Calibration here extracts the verbalized confidence from the response
    and stores it as a scalar.  ECE (which requires a batch) is set to 0.0 —
    batch ECE is computed in ``run_eval`` after accumulating all items.

    Parameters
    ----------
    prompt:
        The user prompt that produced *response*.
    response:
        The model's output text to audit.
    model_name:
        Identifier string for the model (used for logging only).
    generate_fn:
        Callable ``(str) -> str``.  Called up to N=3 times for consistency
        sampling, and once per counterfactual variant for equity.
    retrieved_chunks:
        Sequence of ``(chunk_id, chunk_text)`` pairs for RAG contexts.
        Attribution is skipped when ``None`` or empty.
    run_calibration / run_faithfulness / run_consistency / run_equity / run_attribution:
        Feature flags.  Equity is off by default (expensive).
    equity_axis:
        Demographic axis passed to :func:`~aria_audit.axes.equity.score`.
    copilot_fn:
        Optional Phi-4-mini callable for claim extraction and paraphrase.
    request_id:
        Caller-supplied idempotency key; auto-generated UUID4 when empty.
    db_logger:
        :class:`~aria_audit.storage.sqlite_logger.EnvelopeLogger` instance.
        When provided the finished envelope is written to SQLite.
    suite_name / suite_item_id:
        Forwarded to ``db_logger.log`` for eval-suite provenance.

    Returns
    -------
    AuditEnvelope
    """
    if not request_id:
        request_id = str(uuid.uuid4())

    audit_start = time.perf_counter()
    now = time.time()
    drift_signals: list[DriftSignal] = []

    # -- 1. Calibration (pure-numpy; no GPU) --------------------------------
    calibration_result = None
    if run_calibration:
        try:
            from aria_audit.axes.calibration import parse_verbalized_confidence  # noqa: PLC0415
            conf, label = parse_verbalized_confidence(response)
            # Per-response ECE: no ground truth, use distance from neutral (0.5)
            # as single-response overconfidence proxy.
            # conf=0.5 (default, no hedging) → ECE=0.0 (treated as calibrated)
            # conf=0.99 (overconfident)      → ECE=0.49 (penalised)
            single_ece = round(abs(conf - 0.5), 4)
            calibration_result = CalibrationResult(
                ece_overall=single_ece,
                ece_per_group={},
                confidence=conf,
                confidence_bin=label,
            )
            drift_signals.append(_push_drift("calibration", conf, now))
            logger.debug(
                "audit calibration: request_id=%s conf=%.3f label=%r",
                request_id, conf, label,
            )
        except Exception:
            logger.exception("audit: calibration axis failed — skipping.")

    # -- 2. Consistency (N=3 shared samples; GPU via HHEM / TF-IDF fallback) -
    consistency_result = None
    _paraphrased_responses: list[str] = []  # reused by attribution

    if run_consistency:
        try:
            from aria_audit.axes import consistency as _consistency_mod  # noqa: PLC0415

            # Capture the intermediate paraphrased responses for attribution
            # by monkey-patching generate_fn to record calls during this run.
            _recorded_responses: list[str] = []

            def _recording_generate(p: str) -> str:
                r = generate_fn(p)
                _recorded_responses.append(r)
                return r

            consistency_result = _consistency_mod.score(
                prompt,
                _recording_generate,
                n_samples=3,
                copilot_fn=copilot_fn,
            )
            _paraphrased_responses = _recorded_responses

            # Drift: higher semantic entropy = less consistent = lower score
            entropy_score = consistency_result.semantic_entropy
            drift_signals.append(_push_drift("consistency", entropy_score, now))
            logger.debug(
                "audit consistency: request_id=%s H=%.4f n_clusters=%d",
                request_id,
                consistency_result.semantic_entropy,
                consistency_result.n_meaning_clusters,
            )
        except Exception:
            logger.exception("audit: consistency axis failed — skipping.")

    # -- 3. Faithfulness (HHEM 2.1; single GPU load) -------------------------
    faithfulness_result = None
    if run_faithfulness:
        context = ""
        if retrieved_chunks:
            context = "\n".join(chunk_text for _, chunk_text in retrieved_chunks)
        try:
            from aria_audit.axes import faithfulness as _faith_mod  # noqa: PLC0415
            faithfulness_result = _faith_mod.score(
                response,
                context,
                copilot_fn=copilot_fn,
            )
            drift_signals.append(
                _push_drift("faithfulness", faithfulness_result.hhem_score, now)
            )
            logger.debug(
                "audit faithfulness: request_id=%s hhem=%.4f claims=%d",
                request_id,
                faithfulness_result.hhem_score,
                faithfulness_result.claims_total,
            )
        except Exception:
            logger.exception("audit: faithfulness axis failed — skipping.")

    # -- 4. Attribution (pure-python; reuses _paraphrased_responses) ---------
    attribution_result = None
    if run_attribution and retrieved_chunks:
        try:
            from aria_audit.axes import attribution as _attr_mod  # noqa: PLC0415
            from aria_audit.axes.consistency import paraphrase_prompt  # noqa: PLC0415

            # Prefer to reuse the first response already generated for
            # consistency (saves one generate_fn call).
            if _paraphrased_responses:
                response_b = _paraphrased_responses[0]
            else:
                # Consistency was skipped — generate one paraphrase now.
                paraphrases = paraphrase_prompt(prompt, n=1, copilot_fn=copilot_fn)
                para = paraphrases[0] if paraphrases else prompt
                try:
                    response_b = generate_fn(para)
                except Exception:  # noqa: BLE001
                    logger.warning("audit: attribution paraphrase generate_fn failed; using original response.")
                    response_b = response

            attribution_result = _attr_mod.score(
                response,
                response_b,
                retrieved_chunks[:5],
                retrieved_chunks[:5],
            )
            drift_signals.append(
                _push_drift("attribution", attribution_result.jaccard_at_k, now)
            )
            logger.debug(
                "audit attribution: request_id=%s jaccard@%d=%.4f",
                request_id,
                attribution_result.k,
                attribution_result.jaccard_at_k,
            )
        except Exception:
            logger.exception("audit: attribution axis failed — skipping.")

    # -- 5. Equity (CPU-only; expensive due to many generate_fn calls) -------
    equity_result = None
    if run_equity:
        try:
            from aria_audit.axes import equity as _equity_mod  # noqa: PLC0415
            equity_result = _equity_mod.score(
                prompt,
                generate_fn,
                axis=equity_axis,
            )
            # Drift: DI=1.0 is perfect parity; distance from 1.0 is the signal.
            di_drift_score = abs(1.0 - equity_result.disparate_impact)
            drift_signals.append(_push_drift("equity", di_drift_score, now))
            logger.debug(
                "audit equity: request_id=%s axis=%r DI=%.4f EOD=%.4f",
                request_id,
                equity_axis,
                equity_result.disparate_impact,
                equity_result.equalized_odds_gap,
            )
        except Exception:
            logger.exception("audit: equity axis failed — skipping.")

    # -- Build envelope ------------------------------------------------------
    audit_end = time.perf_counter()
    latency_ms_audit = (audit_end - audit_start) * 1000.0

    # Collect VRAM peak (0.0 on CPU-only environments)
    try:
        from aria_audit.gpu_manager import GPUManager  # noqa: PLC0415
        peak_vram = GPUManager.peak_vram_gb()
    except Exception:  # noqa: BLE001
        peak_vram = 0.0

    chunk_ids = [cid for cid, _ in retrieved_chunks] if retrieved_chunks else []

    envelope = AuditEnvelope(
        request_id=request_id,
        model_name=model_name,
        prompt=prompt,
        response=response,
        retrieved_chunk_ids=chunk_ids,
        calibration=calibration_result,
        faithfulness=faithfulness_result,
        consistency=consistency_result,
        equity=equity_result,
        attribution=attribution_result,
        drift=drift_signals,
        latency_ms_generation=0.0,   # generation latency is measured outside audit()
        latency_ms_audit=latency_ms_audit,
        peak_vram_gb=peak_vram,
        timestamp=now,
    )

    # -- Persist to SQLite if a logger was supplied --------------------------
    if db_logger is not None:
        try:
            db_logger.log(envelope, suite_name=suite_name, suite_item_id=suite_item_id)
        except Exception:
            logger.exception("audit: db_logger.log failed — envelope not persisted.")

    logger.info(
        "audit complete: request_id=%s model=%r latency_ms=%.1f composite=%.1f",
        request_id,
        model_name,
        latency_ms_audit,
        envelope.composite_score,
    )

    return envelope


# ---------------------------------------------------------------------------
# Eval-suite runner
# ---------------------------------------------------------------------------

def run_eval(model: str, eval_suite: str, out_dir: str) -> int:
    """Run the full audit pipeline over every item in an eval suite JSONL file.

    Suite file format (one JSON object per line)::

        {"prompt": "...", "response": "...", "context": "...", "group": "...", "item_id": "..."}

    After processing all items:
    - Batch ECE is computed using ``(confidence, correctness)`` pairs grouped
      by the ``group`` field.  Correctness is proxied as
      ``faithfulness.hhem_score > 0.5`` when faithfulness was run.
    - A CSV summary is written to ``{out_dir}/envelope_summary.csv``.
    - A SQLite database is written to ``{out_dir}/audit_results.db``.

    Parameters
    ----------
    model:
        Ollama model tag (e.g. ``"qwen3:8b-q4_K_M"``).
    eval_suite:
        Path to a JSONL suite file, or a suite name under ``eval/suites/``.
    out_dir:
        Directory for output files.  Created if it does not exist.

    Returns
    -------
    int
        0 on success, 1 on error.
    """
    import sys  # noqa: PLC0415

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Resolve suite path
    suite_path = Path(eval_suite)
    if not suite_path.exists():
        # Try under eval/suites/ relative to project root
        alt = Path("eval") / "suites" / eval_suite
        if alt.exists():
            suite_path = alt
        else:
            logger.error("Eval suite not found: %s (also tried %s)", eval_suite, alt)
            return 1

    # Load suite items
    items: list[dict] = []
    try:
        with suite_path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSON on line %d: %s", lineno, exc)
    except OSError as exc:
        logger.error("Cannot open eval suite %s: %s", suite_path, exc)
        return 1

    if not items:
        logger.error("Eval suite %s contains no valid items.", suite_path)
        return 1

    logger.info("Running eval: model=%r suite=%s n_items=%d", model, suite_path, len(items))

    # Open SQLite logger
    db_path = out_path / "audit_results.db"
    try:
        from aria_audit.storage.sqlite_logger import EnvelopeLogger  # noqa: PLC0415
        db_logger = EnvelopeLogger(db_path)
    except Exception:
        logger.exception("Failed to open SQLite logger at %s", db_path)
        return 1

    # Build generate_fn once — shared across all items
    generate_fn = _make_ollama_generate_fn(model)
    suite_name = suite_path.stem

    # Accumulators for batch ECE
    # per_group_data[group] = list of (confidence, correctness) tuples
    per_group_data: dict[str, list[tuple[float, float]]] = {}

    # Per-axis score accumulators for summary table
    axis_scores: dict[str, list[float]] = {
        "calibration_confidence": [],
        "faithfulness_hhem": [],
        "consistency_entropy": [],
        "equity_di": [],
        "attribution_jaccard": [],
    }
    total_latency_ms = 0.0
    envelopes: list[AuditEnvelope] = []

    suite_start = time.perf_counter()

    for idx, item in enumerate(items):
        prompt = str(item.get("prompt", ""))
        response = str(item.get("response", ""))
        context_text = str(item.get("context", ""))
        group = str(item.get("group", "default"))
        item_id = str(item.get("item_id", str(idx)))

        # Build retrieved_chunks from flat context text (single synthetic chunk)
        retrieved_chunks: list[tuple[str, str]] | None = None
        if context_text.strip():
            retrieved_chunks = [(f"chunk_{item_id}_0", context_text)]

        try:
            env = audit(
                prompt=prompt,
                response=response,
                model_name=model,
                generate_fn=generate_fn,
                retrieved_chunks=retrieved_chunks,
                run_calibration=True,
                run_faithfulness=True,
                run_consistency=True,
                run_equity=False,
                run_attribution=True,
                suite_name=suite_name,
                suite_item_id=item_id,
                db_logger=db_logger,
            )
        except Exception:
            logger.exception("audit() raised for item_id=%s — skipping.", item_id)
            continue

        envelopes.append(env)
        total_latency_ms += env.latency_ms_audit

        # Accumulate calibration data for batch ECE
        if env.calibration is not None:
            conf = env.calibration.confidence
            correctness = float(
                env.faithfulness.hhem_score > 0.5 if env.faithfulness else 0.5
            )
            per_group_data.setdefault(group, []).append((conf, correctness))
            axis_scores["calibration_confidence"].append(conf)

        if env.faithfulness is not None:
            axis_scores["faithfulness_hhem"].append(env.faithfulness.hhem_score)
        if env.consistency is not None:
            axis_scores["consistency_entropy"].append(env.consistency.semantic_entropy)
        if env.equity is not None:
            axis_scores["equity_di"].append(env.equity.disparate_impact)
        if env.attribution is not None:
            axis_scores["attribution_jaccard"].append(env.attribution.jaccard_at_k)

        logger.debug(
            "item %d/%d item_id=%s composite=%.1f latency=%.0f ms",
            idx + 1, len(items), item_id, env.composite_score, env.latency_ms_audit,
        )

    suite_elapsed = (time.perf_counter() - suite_start) * 1000.0

    # -- Batch ECE computation -----------------------------------------------
    try:
        from aria_audit.axes.calibration import expected_calibration_error  # noqa: PLC0415

        ece_per_group: dict[str, float] = {}
        all_confs: list[float] = []
        all_correct: list[float] = []
        for group, pairs in per_group_data.items():
            confs_g = np.array([p[0] for p in pairs], dtype=float)
            correct_g = np.array([p[1] for p in pairs], dtype=float)
            all_confs.extend(confs_g.tolist())
            all_correct.extend(correct_g.tolist())
            if len(confs_g) >= 2:
                ece_per_group[group] = expected_calibration_error(confs_g, correct_g)

        if all_confs:
            ece_overall = expected_calibration_error(
                np.array(all_confs, dtype=float),
                np.array(all_correct, dtype=float),
            )
        else:
            ece_overall = float("nan")

        logger.info(
            "Batch ECE: overall=%.4f per_group=%s",
            ece_overall if not (ece_overall != ece_overall) else float("nan"),
            {k: f"{v:.4f}" for k, v in ece_per_group.items()},
        )
    except Exception:
        logger.exception("Batch ECE computation failed.")
        ece_overall = float("nan")
        ece_per_group = {}

    # -- Summary table -------------------------------------------------------
    n_items = len(envelopes)
    print("\n" + "=" * 72)
    print(f"ARIA-Audit eval summary — model={model!r}  suite={suite_name!r}")
    print(f"Items processed: {n_items}/{len(items)}  "
          f"Total latency: {suite_elapsed:.0f} ms  "
          f"Avg/item: {(suite_elapsed / n_items if n_items else 0.0):.0f} ms")
    print("=" * 72)

    _fmt_mean = lambda vals: f"{sum(vals)/len(vals):.4f}" if vals else "—"

    rows = [
        ("calibration_confidence", axis_scores["calibration_confidence"]),
        ("faithfulness_hhem", axis_scores["faithfulness_hhem"]),
        ("consistency_entropy", axis_scores["consistency_entropy"]),
        ("equity_di", axis_scores["equity_di"]),
        ("attribution_jaccard", axis_scores["attribution_jaccard"]),
    ]
    print(f"{'Axis':<28} {'Mean':>10}  {'N':>6}")
    print("-" * 48)
    for axis_name, vals in rows:
        print(f"{axis_name:<28} {_fmt_mean(vals):>10}  {len(vals):>6}")
    if all_confs:
        ece_str = f"{ece_overall:.4f}" if ece_overall == ece_overall else "nan"
        print(f"{'ece_overall (batch)':<28} {ece_str:>10}  {len(all_confs):>6}")
    print("=" * 72 + "\n")

    # -- Save CSV summary ----------------------------------------------------
    csv_path = out_path / "envelope_summary.csv"
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "item_idx",
                "item_id",
                "group",
                "request_id",
                "composite_score",
                "calibration_confidence",
                "calibration_ece_overall",
                "faithfulness_hhem",
                "faithfulness_raga",
                "consistency_entropy",
                "consistency_n_clusters",
                "equity_di",
                "equity_eod",
                "attribution_jaccard",
                "latency_ms_audit",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, (item, env) in enumerate(zip(items[:len(envelopes)], envelopes)):
                def _f(v, default=""):
                    return v if v is not None else default

                writer.writerow({
                    "item_idx": i,
                    "item_id": item.get("item_id", i),
                    "group": item.get("group", "default"),
                    "request_id": env.request_id,
                    "composite_score": f"{env.composite_score:.4f}",
                    "calibration_confidence": (
                        f"{env.calibration.confidence:.4f}" if env.calibration else ""
                    ),
                    "calibration_ece_overall": (
                        f"{ece_overall:.4f}"
                        if all_confs and ece_overall == ece_overall
                        else ""
                    ),
                    "faithfulness_hhem": (
                        f"{env.faithfulness.hhem_score:.4f}" if env.faithfulness else ""
                    ),
                    "faithfulness_raga": (
                        f"{env.faithfulness.raga_faithfulness:.4f}" if env.faithfulness else ""
                    ),
                    "consistency_entropy": (
                        f"{env.consistency.semantic_entropy:.4f}" if env.consistency else ""
                    ),
                    "consistency_n_clusters": (
                        env.consistency.n_meaning_clusters if env.consistency else ""
                    ),
                    "equity_di": (
                        f"{env.equity.disparate_impact:.4f}" if env.equity else ""
                    ),
                    "equity_eod": (
                        f"{env.equity.equalized_odds_gap:.4f}" if env.equity else ""
                    ),
                    "attribution_jaccard": (
                        f"{env.attribution.jaccard_at_k:.4f}" if env.attribution else ""
                    ),
                    "latency_ms_audit": f"{env.latency_ms_audit:.1f}",
                })
        logger.info("CSV summary written to %s", csv_path)
    except OSError:
        logger.exception("Failed to write CSV summary to %s", csv_path)
        db_logger.close()
        return 1

    db_logger.close()
    print(f"Results: SQLite → {db_path}  CSV → {csv_path}")
    return 0
