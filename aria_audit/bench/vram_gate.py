"""Phase 0 blocking VRAM gate test.

Sequence (per plan v3):
  1. Load Qwen3 8B Q4_K_M (anchor) → measure VRAM
  2. Load BGE-M3 (batch_size=8), embed `sample_n` chunks → measure peak VRAM → unload
  3. Load HHEM 2.1, score `sample_n` (response, context) pairs → measure peak VRAM → unload
  4. Print latency/memory table

Pass criteria: peak VRAM during steps 2 and 3 stays under 7.6 GB.
Failure path: fall back to Qwen2.5-7B Q4_K_M, re-run.

This file is intentionally light on hard deps — it raises clear errors if
Ollama or transformers are not installed so the user can fix and re-run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict

from aria_audit.gpu_manager import (
    BGE_M3,
    HHEM_21,
    QWEN3_8B,
    GPUManager,
    VRAMExceeded,
    get_manager,
)

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    qwen_resident_gb: float
    bge_peak_gb: float
    hhem_peak_gb: float
    qwen_tokens_per_sec: float
    bge_chunks_per_sec: float
    hhem_pairs_per_sec: float
    passed: bool
    notes: str = ""

    def pretty(self) -> str:
        d = asdict(self)
        return "\n".join(f"  {k:<26s} {v}" for k, v in d.items())


def _load_qwen_via_ollama(model: str) -> object:
    """Warm up the Ollama model. Returns a thin client object."""
    try:
        import httpx
    except ImportError as e:
        raise RuntimeError("httpx required: pip install httpx") from e

    client = httpx.Client(base_url="http://localhost:11434", timeout=120.0)
    # Trigger model load
    r = client.post("/api/generate", json={"model": model, "prompt": "ok", "stream": False})
    r.raise_for_status()
    return client


def _unload_noop(_obj: object) -> None:
    pass


def _load_bge_m3() -> object:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError("sentence-transformers required: pip install sentence-transformers") from e
    return SentenceTransformer("BAAI/bge-m3")


def _unload_bge_m3(obj: object) -> None:
    del obj
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _load_hhem() -> object:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise RuntimeError("transformers required: pip install transformers") from e
    model = AutoModelForSequenceClassification.from_pretrained(
        "vectara/hallucination_evaluation_model", trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained("vectara/hallucination_evaluation_model")
    return (model, tok)


def _unload_hhem(obj: object) -> None:
    del obj
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_vram_gate(model: str = "qwen3:8b-q4_K_M", sample_n: int = 100) -> int:
    """Returns 0 on pass, 1 on fail (caller exits with this)."""
    mgr: GPUManager = get_manager()
    mgr.reset_peak_stats()

    chunks = [f"This is sample chunk number {i} about a placeholder topic." for i in range(sample_n)]
    pairs = [(c, c) for c in chunks]

    notes: list[str] = []
    passed = True

    # Step 1: Qwen3 8B anchor
    try:
        with mgr.acquire(QWEN3_8B, lambda: _load_qwen_via_ollama(model), _unload_noop) as client:
            t0 = time.perf_counter()
            r = client.post(
                "/api/generate",
                json={"model": model, "prompt": "Count from 1 to 50.", "stream": False, "options": {"num_predict": 256}},
            )
            r.raise_for_status()
            dt = time.perf_counter() - t0
            tokens = r.json().get("eval_count", 256)
            qwen_tps = tokens / dt if dt > 0 else 0.0
            qwen_resident = mgr.peak_vram_gb()
            logger.info("Qwen3 resident=%.2f GB, %.1f tok/s", qwen_resident, qwen_tps)
    except Exception as e:
        return _fail(f"Qwen3 anchor load failed: {e}")

    # Step 2: BGE-M3
    bge_peak = 0.0
    bge_tps = 0.0
    try:
        mgr.reset_peak_stats()
        with mgr.acquire(BGE_M3, _load_bge_m3, _unload_bge_m3) as bge:
            t0 = time.perf_counter()
            _ = bge.encode(chunks, batch_size=8, show_progress_bar=False)
            dt = time.perf_counter() - t0
            bge_tps = sample_n / dt if dt > 0 else 0.0
            bge_peak = mgr.peak_vram_gb()
            logger.info("BGE-M3 peak=%.2f GB, %.1f chunks/s", bge_peak, bge_tps)
    except VRAMExceeded as e:
        passed = False
        notes.append(f"BGE-M3 VRAMExceeded: {e}")
    except Exception as e:
        passed = False
        notes.append(f"BGE-M3 load failed: {e}")

    # Step 3: HHEM 2.1
    hhem_peak = 0.0
    hhem_tps = 0.0
    try:
        mgr.reset_peak_stats()
        with mgr.acquire(HHEM_21, _load_hhem, _unload_hhem) as obj:
            model_obj, _tok = obj
            t0 = time.perf_counter()
            scores = model_obj.predict(pairs)  # vectara model exposes .predict
            _ = list(scores) if hasattr(scores, "__iter__") else scores
            dt = time.perf_counter() - t0
            hhem_tps = sample_n / dt if dt > 0 else 0.0
            hhem_peak = mgr.peak_vram_gb()
            logger.info("HHEM peak=%.2f GB, %.1f pairs/s", hhem_peak, hhem_tps)
    except VRAMExceeded as e:
        passed = False
        notes.append(f"HHEM VRAMExceeded: {e}")
    except Exception as e:
        passed = False
        notes.append(f"HHEM load failed: {e}")

    # Pass gate: each aux peak < 7.6 GB
    if bge_peak >= 7.6 or hhem_peak >= 7.6:
        passed = False
        notes.append(f"Peak exceeded 7.6 GB ceiling: bge={bge_peak:.2f} hhem={hhem_peak:.2f}")

    result = GateResult(
        qwen_resident_gb=qwen_resident,
        bge_peak_gb=bge_peak,
        hhem_peak_gb=hhem_peak,
        qwen_tokens_per_sec=qwen_tps,
        bge_chunks_per_sec=bge_tps,
        hhem_pairs_per_sec=hhem_tps,
        passed=passed,
        notes="; ".join(notes),
    )
    print("=" * 60)
    print("ARIA Phase 0 VRAM Gate Test")
    print("=" * 60)
    print(result.pretty())
    print("=" * 60)
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("Fallback: pull qwen2.5:7b-q4_K_M and re-run.")
    return 0 if passed else 1


def _fail(msg: str) -> int:
    print(f"VRAM gate FAIL: {msg}")
    print("Fallback: pull qwen2.5:7b-q4_K_M and re-run.")
    return 1
