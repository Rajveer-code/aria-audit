"""Axis 3 — Semantic entropy under sampled paraphrase.

References:
  - Kuhn, Gal & Farquhar 2023 "Semantic Uncertainty" arXiv:2302.09664
  - Wang et al. 2022 "Self-Consistency" arXiv:2203.11171

Sample-based only. Semantic-Entropy-Probes (Kossen 2024) require hidden states,
which Ollama does not expose.

Implementation notes:
  - HHEM 2.1 is loaded once per `cluster_by_entailment` call via GPUManager,
    scoring all pairs in a single model-resident window before unloading.
  - When HHEM is unavailable (no GPU / import failure), falls back to TF-IDF
    cosine similarity with threshold 0.7 as an approximate entailment proxy.
  - All heavy imports (torch, transformers, sklearn) are deferred inside
    functions so the module is importable on any CPU-only environment.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Callable

from aria_audit.core import ConsistencyResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paraphrase generation
# ---------------------------------------------------------------------------

_PARAPHRASE_PREFIXES = [
    "Please answer: ",
    "Can you explain: ",
    "I'd like to know: ",
    "Could you tell me: ",
    "Help me understand: ",
    "Would you clarify: ",
    "In your own words: ",
    "Describe briefly: ",
]

_PARAPHRASE_SUFFIXES = [
    " Please be concise.",
    " Explain clearly.",
    " Provide detail.",
    " Keep it short.",
    " Be thorough.",
    " Summarise if possible.",
    " Give a direct answer.",
    " Include key points.",
]


def _template_paraphrases(prompt: str, n: int) -> list[str]:
    """Produce exactly n template-based paraphrases of `prompt`.

    Cycles through prefix and suffix variations; falls back to the original
    prompt (with minor whitespace change) if more are needed than templates.
    """
    results: list[str] = []
    for i in range(n):
        prefix = _PARAPHRASE_PREFIXES[i % len(_PARAPHRASE_PREFIXES)]
        suffix = _PARAPHRASE_SUFFIXES[i % len(_PARAPHRASE_SUFFIXES)]
        results.append(f"{prefix}{prompt}{suffix}")
    return results


def paraphrase_prompt(
    prompt: str,
    n: int = 3,
    copilot_fn: Callable[[str], str] | None = None,
) -> list[str]:
    """Generate n semantically-equivalent paraphrases of `prompt`.

    Parameters
    ----------
    prompt:
        The original question / instruction to paraphrase.
    n:
        Exact number of paraphrases to return.
    copilot_fn:
        Optional callable (e.g. Phi-4-mini) that accepts a text prompt and
        returns a text response.  When provided the function attempts to
        obtain paraphrases via an LLM; template fallback is used if the
        model response cannot be parsed as a JSON array.

    Returns
    -------
    list[str]
        Exactly n paraphrased prompts.  Original prompt is used to pad if
        fewer than n could be generated.
    """
    if n <= 0:
        return []

    results: list[str] = []

    if copilot_fn is not None:
        instruction = (
            f"Rephrase the following question in {n} different ways while "
            "preserving meaning. Return as JSON array.\n"
            f"Question: {prompt}\nJSON:"
        )
        try:
            raw = copilot_fn(instruction)
            # Isolate the JSON array — the model may emit extra prose
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(raw[start : end + 1])
                if isinstance(parsed, list):
                    results = [str(p) for p in parsed if str(p).strip()]
        except Exception as exc:  # noqa: BLE001
            logger.debug("copilot_fn paraphrase parse failed (%s); using templates", exc)
            results = []

    # Supplement / replace with templates if needed
    if len(results) < n:
        templates = _template_paraphrases(prompt, n)
        # Use templates only for the slots not already filled by copilot_fn
        needed = n - len(results)
        results.extend(templates[:needed])

    # Pad with original if we still don't have enough (shouldn't happen)
    while len(results) < n:
        results.append(prompt)

    return results[:n]


# ---------------------------------------------------------------------------
# Union-Find helper for greedy clustering
# ---------------------------------------------------------------------------

class _UnionFind:
    """Minimal union-find over integer indices."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def clusters(self) -> list[list[int]]:
        groups: dict[int, list[int]] = {}
        for i in range(len(self._parent)):
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return list(groups.values())


# ---------------------------------------------------------------------------
# HHEM 2.1 loading helpers (lazy, GPU-managed)
# ---------------------------------------------------------------------------

def _load_hhem():
    """Load HHEM 2.1 pipeline and return it.  Called inside GPUManager.acquire."""
    from transformers import pipeline  # noqa: PLC0415

    return pipeline(
        "text-classification",
        model="vectara/hallucination_evaluation_model",
        trust_remote_code=True,
    )


def _unload_hhem(pipe) -> None:
    """Release HHEM weights from GPU memory."""
    try:
        import torch  # noqa: PLC0415

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _score_pairs_hhem(
    pipe,
    pairs: list[tuple[str, str]],
) -> list[float]:
    """Score all (hypothesis, premise) pairs with one batched forward pass.

    HHEM 2.1 uses `text-classification` with inputs formatted as
    ``"<hypothesis> [SEP] <premise>"``.  The model emits a label
    ``"consistent"`` / ``"inconsistent"``; we return P(consistent).
    """
    if not pairs:
        return []

    # Format as HHEM expects: hypothesis SEP premise
    inputs = [f"{h} [SEP] {p}" for h, p in pairs]
    raw = pipe(inputs, truncation=True, max_length=512)

    scores: list[float] = []
    for result in raw:
        label = result.get("label", "").lower()
        prob = float(result.get("score", 0.5))
        # HHEM label may be "consistent" (positive) or "inconsistent" (negative)
        if "inconsistent" in label:
            scores.append(1.0 - prob)
        else:
            scores.append(prob)
    return scores


# ---------------------------------------------------------------------------
# TF-IDF fallback scoring
# ---------------------------------------------------------------------------

def _score_pairs_tfidf(texts: list[str], pairs: list[tuple[int, int]]) -> list[float]:
    """Cosine similarity of TF-IDF vectors as a CPU fallback for entailment."""
    if not pairs:
        return []

    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
    from sklearn.metrics.pairwise import cosine_similarity  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    vectorizer = TfidfVectorizer(sublinear_tf=True).fit(texts)
    vecs = vectorizer.transform(texts).toarray().astype(np.float32)

    results: list[float] = []
    for i, j in pairs:
        sim = float(cosine_similarity(vecs[i : i + 1], vecs[j : j + 1])[0, 0])
        results.append(sim)
    return results


# ---------------------------------------------------------------------------
# Core clustering
# ---------------------------------------------------------------------------

def cluster_by_entailment(
    responses: list[str],
    hhem_threshold: float = 0.5,
) -> list[list[int]]:
    """Cluster responses into meaning-equivalent groups via bidirectional NLI.

    Two responses i and j are placed in the same cluster when both
    HHEM(responses[i], responses[j]) >= hhem_threshold
    AND HHEM(responses[j], responses[i]) >= hhem_threshold,
    i.e. mutual entailment in both directions (Kuhn 2023 §3.1).

    A single HHEM instance is loaded once for the entire call; all pairs are
    scored in one batched window before the model is unloaded.

    When HHEM is unavailable (no GPU / missing transformers), falls back to
    TF-IDF cosine similarity >= 0.7 as a symmetric approximate proxy.

    Parameters
    ----------
    responses:
        List of text responses to cluster.
    hhem_threshold:
        Minimum bidirectional entailment score to merge two responses.

    Returns
    -------
    list[list[int]]
        Each inner list contains the indices of responses sharing a meaning
        cluster.  Every index 0 … len(responses)-1 appears exactly once.
    """
    n = len(responses)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    uf = _UnionFind(n)

    # Build all directed pairs (i→j) and (j→i) for i < j
    directed_pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            directed_pairs.append((i, j))
            directed_pairs.append((j, i))

    # Attempt HHEM scoring via GPUManager
    hhem_available = False
    pair_scores: list[float] = []

    try:
        from aria_audit.gpu_manager import get_manager, HHEM_21  # noqa: PLC0415

        manager = get_manager()
        hhem_text_pairs = [(responses[i], responses[j]) for i, j in directed_pairs]

        with manager.acquire(HHEM_21, _load_hhem, _unload_hhem) as pipe:
            pair_scores = _score_pairs_hhem(pipe, hhem_text_pairs)
        hhem_available = True

    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "HHEM unavailable (%s); falling back to TF-IDF cosine similarity.", exc
        )

    if not hhem_available:
        # TF-IDF fallback: symmetric by construction; use threshold 0.7
        tfidf_threshold = 0.7
        index_pairs = [(i, j) for i, j in directed_pairs]
        tfidf_scores = _score_pairs_tfidf(responses, index_pairs)

        # For TF-IDF the similarity is symmetric, so i→j == j→i.
        # We still iterate all directed pairs so the logic below is uniform.
        pair_scores = tfidf_scores
        hhem_threshold = tfidf_threshold

    # Build a mapping from directed pair index to score
    score_map: dict[tuple[int, int], float] = {}
    for k, (i, j) in enumerate(directed_pairs):
        score_map[(i, j)] = pair_scores[k]

    # Union indices that satisfy bidirectional entailment
    for i in range(n):
        for j in range(i + 1, n):
            forward = score_map.get((i, j), 0.0)
            backward = score_map.get((j, i), 0.0)
            if forward >= hhem_threshold and backward >= hhem_threshold:
                uf.union(i, j)

    return uf.clusters()


# ---------------------------------------------------------------------------
# Semantic entropy
# ---------------------------------------------------------------------------

def semantic_entropy(clusters: list[list[int]], n_total: int) -> float:
    """Shannon entropy over meaning clusters (Kuhn 2023 eq. 3).

    H = -∑ (|C_i| / n) * log(|C_i| / n)

    Parameters
    ----------
    clusters:
        Output of `cluster_by_entailment`.
    n_total:
        Total number of responses (equals sum of cluster sizes).

    Returns
    -------
    float
        Semantic entropy in nats.  Returns 0.0 when n_total == 0 or there
        is only one cluster containing all responses.
    """
    if n_total == 0 or not clusters:
        return 0.0

    h = 0.0
    for cluster in clusters:
        p = len(cluster) / n_total
        if p > 0.0:
            h -= p * math.log(p)
    return h


# ---------------------------------------------------------------------------
# Public scoring entry-point
# ---------------------------------------------------------------------------

def score(
    prompt: str,
    generate_fn: Callable[[str], str],
    n_samples: int = 3,
    copilot_fn: Callable[[str], str] | None = None,
) -> ConsistencyResult:
    """Compute semantic entropy for a prompt by sampling paraphrased responses.

    Algorithm (Kuhn 2023):
      1. Generate n_samples paraphrases of `prompt`.
      2. Obtain one response per paraphrase via `generate_fn`.
      3. Cluster responses by bidirectional NLI entailment.
      4. Compute semantic entropy H over the cluster distribution.

    Parameters
    ----------
    prompt:
        The original prompt to audit.
    generate_fn:
        Callable ``(prompt: str) -> str`` — invoked once per paraphrased
        prompt.  Typically wraps an Ollama or OpenAI-compatible endpoint.
    n_samples:
        Number of paraphrased prompts (and therefore responses) to generate.
    copilot_fn:
        Optional Phi-4-mini callable for LLM-based paraphrase generation.
        Falls back to template-based paraphrases when None.

    Returns
    -------
    ConsistencyResult
        ``semantic_entropy``   — Shannon entropy in nats (0 = maximally consistent)
        ``n_samples``          — number of responses actually collected
        ``n_meaning_clusters`` — number of distinct meaning clusters
        ``majority_share``     — fraction of responses in the largest cluster
    """
    # Step 1: generate paraphrases
    paraphrases = paraphrase_prompt(prompt, n=n_samples, copilot_fn=copilot_fn)

    # Step 2: collect one response per paraphrase
    responses: list[str] = []
    for para in paraphrases:
        try:
            resp = generate_fn(para)
            responses.append(str(resp))
        except Exception as exc:  # noqa: BLE001
            logger.warning("generate_fn raised for paraphrase %r: %s", para[:60], exc)

    n_collected = len(responses)

    if n_collected == 0:
        logger.error("No responses collected; returning zero-entropy result.")
        return ConsistencyResult(
            semantic_entropy=0.0,
            n_samples=0,
            n_meaning_clusters=0,
            majority_share=0.0,
        )

    # Step 3: cluster by bidirectional entailment
    clusters = cluster_by_entailment(responses)

    # Step 4: semantic entropy
    H = semantic_entropy(clusters, n_total=n_collected)

    # Majority cluster share
    max_cluster_size = max(len(c) for c in clusters) if clusters else 0
    majority_share = max_cluster_size / n_collected if n_collected > 0 else 0.0

    return ConsistencyResult(
        semantic_entropy=H,
        n_samples=n_collected,
        n_meaning_clusters=len(clusters),
        majority_share=majority_share,
    )
