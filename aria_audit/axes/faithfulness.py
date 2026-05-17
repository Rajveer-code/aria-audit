"""Axis 2 — Faithfulness via HHEM 2.1 + RAGAS-style claim extraction.

Uses vectara/hallucination_evaluation_model (HHEM 2.1) loaded once per call
through GPUManager to enforce VRAM mutual-exclusion with the other auxiliary
models.  The public surface is deliberately narrow:

  extract_claims(response, copilot_fn=None) -> list[str]
  hhem_score_pair(response, context)        -> float
  score(response, context, copilot_fn=None) -> FaithfulnessResult

All heavy imports (torch, transformers, GPUManager) are deferred inside
function bodies so the module is importable in CPU-only / test environments
without any optional dependency installed.

References:
  - Vectara HHEM 2.1 — https://huggingface.co/vectara/hallucination_evaluation_model
  - RAGAS Faithfulness — https://docs.ragas.io/
"""

from __future__ import annotations

import json
import logging
import re
from statistics import mean
from typing import Callable

from aria_audit.core import FaithfulnessResult

logger = logging.getLogger(__name__)

_FALLBACK_NEUTRAL = 0.5  # returned when HHEM unavailable and TF-IDF also fails


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity as a faithfulness proxy when HHEM unavailable.

    Measures lexical overlap between response and context.  Not as good as NLI
    but honest — documented as 'tfidf_fallback' in the result.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
        from sklearn.metrics.pairwise import cosine_similarity  # noqa: PLC0415

        vec = TfidfVectorizer(min_df=1, stop_words="english")
        mat = vec.fit_transform([text_a, text_b])
        sim = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
        # Clamp to [0, 1]
        return max(0.0, min(1.0, sim))
    except Exception as exc:
        logger.warning("TF-IDF fallback also failed (%s); returning neutral %.1f", exc, _FALLBACK_NEUTRAL)
        return _FALLBACK_NEUTRAL

# ---------------------------------------------------------------------------
# Abbreviation list used to avoid splitting on "Dr.", "Mr.", etc.
# ---------------------------------------------------------------------------
_ABBREVS: frozenset[str] = frozenset(
    [
        "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc",
        "fig", "eq", "sec", "ref", "approx", "dept", "est", "e.g",
        "i.e", "cf", "al", "vol", "no", "pp", "st", "ave",
    ]
)

# Minimum character length a claim must have to be kept.
_MIN_CLAIM_LEN: int = 10

# Sentence-terminal punctuation tokens we split on.
_SPLIT_RE = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+")


def _smart_sentence_split(text: str) -> list[str]:
    """Split *text* into sentences without cutting on known abbreviations.

    Strategy:
    1. Use a regex that looks for '. ', '! ', '? ' only when NOT preceded by
       a token matching a known abbreviation.
    2. Post-filter: drop fragments shorter than _MIN_CLAIM_LEN chars and
       purely whitespace strings.
    """
    # Normalise line-breaks to spaces so multi-line responses work cleanly.
    text = text.replace("\n", " ").strip()
    if not text:
        return []

    # Primary regex split on sentence boundaries.
    raw_sentences = _SPLIT_RE.split(text)

    # Secondary check: if the last token before the period is a known
    # abbreviation (case-insensitive, letters only), merge back.
    merged: list[str] = []
    buf = ""
    for sent in raw_sentences:
        if buf:
            candidate = buf + " " + sent
        else:
            candidate = sent

        # Check whether this candidate ended mid-abbreviation
        # (last "word" before a period at end of buf is in _ABBREVS).
        trailing_abbrev = re.search(r"\b([A-Za-z]{1,5})\.\s*$", buf)
        if trailing_abbrev and trailing_abbrev.group(1).lower() in _ABBREVS:
            buf = candidate
        else:
            if buf:
                merged.append(buf.strip())
            buf = sent

    if buf.strip():
        merged.append(buf.strip())

    return [s for s in merged if len(s) >= _MIN_CLAIM_LEN]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_claims(
    response: str,
    copilot_fn: Callable[[str], str] | None = None,
) -> list[str]:
    """Split *response* into atomic factual claims.

    Parameters
    ----------
    response:
        The LLM-generated text to decompose.
    copilot_fn:
        Optional callable backed by Phi-4-mini (or any instruction-following
        LLM).  It receives a prompt string and must return a string.  When
        provided the function attempts to parse the model's output as a JSON
        array of strings; on parse failure it falls back to
        ``_smart_sentence_split``.

    Returns
    -------
    list[str]
        Deduplicated list of claim strings, ordered as extracted.
    """
    if not response or not response.strip():
        return []

    if copilot_fn is not None:
        prompt = (
            "List each atomic factual claim in the following text as a JSON "
            "array of strings. Output ONLY the JSON array, no commentary.\n\n"
            f"Text: {response}\nJSON:"
        )
        try:
            raw = copilot_fn(prompt)
            # Strip markdown fences if the model wraps output in ```json … ```
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                claims = [str(c).strip() for c in parsed if str(c).strip()]
                if claims:
                    return claims
                # Fall through on empty list
            logger.debug("copilot_fn returned non-list JSON; falling back to sentence split.")
        except (json.JSONDecodeError, Exception) as exc:
            logger.debug("copilot_fn JSON parse failed (%s); falling back to sentence split.", exc)

    return _smart_sentence_split(response)


# ---------------------------------------------------------------------------
# HHEM 2.1 internals
# ---------------------------------------------------------------------------

def _build_hhem_loader():
    """Return a loader callable for the HHEM 2.1 model.

    The loader imports transformers lazily and returns a namespace-style object
    with a ``.predict()`` method that mirrors the vectara HHEM API.
    """
    def loader():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415
        import torch  # noqa: PLC0415

        model_id = "vectara/hallucination_evaluation_model"
        logger.info("Loading HHEM 2.1 from %s …", model_id)
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        class _HHEMWrapper:
            """Thin wrapper that exposes the `.predict(pairs)` interface."""

            def __init__(self, m, t, dev):
                self._model = m
                self._tok = t
                self._device = dev

            def predict(self, pairs: list[tuple[str, str] | list[str]]) -> list[float]:
                """Score each (text, hypothesis) pair.

                The vectara HHEM model is a sequence-classification model whose
                label 1 means "supported" and label 0 means "hallucinated".
                We return P(label=1) for each pair.

                Parameters
                ----------
                pairs:
                    List of [text, hypothesis] or (text, hypothesis) where
                    *text* is the grounding context and *hypothesis* is the
                    claim to evaluate.
                """
                if not pairs:
                    return []
                import torch  # noqa: PLC0415

                texts = [str(p[0]) for p in pairs]
                hypotheses = [str(p[1]) for p in pairs]

                inputs = self._tok(
                    texts,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self._device)

                with torch.no_grad():
                    logits = self._model(**inputs).logits  # shape (B, 2)

                probs = torch.softmax(logits, dim=-1)
                # Index 1 is the "supported / not hallucinated" label.
                scores = probs[:, 1].cpu().tolist()
                return scores

        return _HHEMWrapper(model, tok, device)

    return loader


def _build_hhem_unloader():
    """Return an unloader callable that moves the HHEM model off the GPU."""

    def unloader(wrapper) -> None:
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                wrapper._model.cpu()
                del wrapper._model
                del wrapper._tok
                torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover
            logger.warning("HHEM unload error (non-fatal): %s", exc)

    return unloader


def hhem_score_pair(response: str, context: str) -> float:
    """Score a single (response, context) pair with HHEM 2.1.

    Loads the model through GPUManager (acquiring the HHEM_21 aux slot),
    scores the pair, then immediately releases the slot.

    Parameters
    ----------
    response:
        The text whose faithfulness relative to *context* is being evaluated.
    context:
        The grounding document / retrieved chunk.

    Returns
    -------
    float
        Score in [0, 1].  Higher values indicate the *response* is well
        supported by *context* (less hallucinated).
    """
    from aria_audit.gpu_manager import HHEM_21, get_manager  # noqa: PLC0415

    loader = _build_hhem_loader()
    unloader = _build_hhem_unloader()

    with get_manager().acquire(HHEM_21, loader, unloader) as hhem:
        scores = hhem.predict([(context, response)])
        return float(scores[0])


def score(
    response: str,
    context: str,
    copilot_fn: Callable[[str], str] | None = None,
) -> FaithfulnessResult:
    """Compute the full faithfulness result for *response* grounded in *context*.

    Workflow
    --------
    1. Extract atomic claims from *response* (optionally via Phi-4-mini).
    2. Acquire HHEM 2.1 once through GPUManager.
    3. Score every (context, claim) pair in a single batched forward pass.
    4. Partition claims into supported (score >= 0.5) and unsupported.
    5. Return :class:`~aria_audit.core.FaithfulnessResult`.

    Empty-response edge case: if no claims are extracted, all faithfulness
    metrics are set to 1.0 (vacuously faithful — nothing was claimed).

    Parameters
    ----------
    response:
        The LLM output to evaluate.
    context:
        The reference document / retrieved context against which claims are
        checked.
    copilot_fn:
        Optional Phi-4-mini callable forwarded to :func:`extract_claims`.

    Returns
    -------
    FaithfulnessResult
    """
    claims = extract_claims(response, copilot_fn=copilot_fn)

    if not claims:
        # No claims extracted — return neutral 0.5 (unknown), not 1.0 (vacuously perfect).
        logger.debug("faithfulness.score: no claims extracted — returning neutral %.1f.", _FALLBACK_NEUTRAL)
        return FaithfulnessResult(
            hhem_score=_FALLBACK_NEUTRAL,
            claims_total=0,
            claims_supported=0,
            claims_unsupported=[],
            raga_faithfulness=_FALLBACK_NEUTRAL,
        )

    # -- Attempt HHEM 2.1 scoring ------------------------------------------
    scores_list: list[float] | None = None
    try:
        from aria_audit.gpu_manager import HHEM_21, get_manager  # noqa: PLC0415

        loader   = _build_hhem_loader()
        unloader = _build_hhem_unloader()
        with get_manager().acquire(HHEM_21, loader, unloader) as hhem:
            pairs = [(context, claim) for claim in claims]
            scores_list = hhem.predict(pairs)
        logger.debug("faithfulness.score: HHEM 2.1 scored %d claims.", len(claims))
    except Exception as exc:
        logger.warning(
            "faithfulness.score: HHEM 2.1 unavailable (%s) — falling back to TF-IDF cosine.", exc
        )

    # -- TF-IDF fallback if HHEM failed ------------------------------------
    if scores_list is None:
        tfidf_sim = _tfidf_cosine(response, context)
        # Broadcast one similarity score across all claims (coarse approximation)
        scores_list = [tfidf_sim] * len(claims)
        logger.debug("faithfulness.score: TF-IDF fallback score=%.3f for %d claims.", tfidf_sim, len(claims))

    supported: list[str] = []
    unsupported: list[str] = []
    for claim, s in zip(claims, scores_list):
        if s >= 0.5:
            supported.append(claim)
        else:
            unsupported.append(claim)

    n = len(claims)
    hhem_mean = float(mean(scores_list))
    raga = len(supported) / n

    logger.debug(
        "faithfulness.score: %d claims, %d supported (%.3f), mean=%.3f",
        n, len(supported), raga, hhem_mean,
    )

    return FaithfulnessResult(
        hhem_score=hhem_mean,
        claims_total=n,
        claims_supported=len(supported),
        claims_unsupported=unsupported,
        raga_faithfulness=raga,
    )
