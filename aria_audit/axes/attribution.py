"""Axis 5 — Retrieval-attribution Jaccard@k under paraphrase.

HONEST REFRAME of CPFE Axis 5 (Jaccard of feature vocabularies). Original
CPFE pipeline used Captum integrated-gradients on classifier features.
Ollama exposes neither gradients nor a fixed feature vocabulary, so for
generative LLMs we measure:

  For each atomic claim in the response, identify the subset of top-k
  retrieved chunks that support it (semantic match + string overlap).
  Then compute Jaccard of supporting-chunk-id sets across two paraphrased
  prompts producing equivalent responses.

This is honestly weaker than feature-saliency attribution and must be
documented as such in the paper.

References:
  - FASS (Feature Attribution Stability Suite) arXiv:2604.02532
  - Slack et al. NeurIPS 2020 "Fooling LIME/SHAP"
"""

from __future__ import annotations

import re

from aria_audit.core import AttributionResult

# ---------------------------------------------------------------------------
# Stop-words (manually defined; 30 words — no external deps)
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "to", "of", "and", "or", "in",
        "on", "at",
    ]
)

# Sentence-ending punctuation pattern (local copy; avoids circular import with faithfulness.py)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words, remove stop-words."""
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in _STOPWORDS]


def _bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    """Return list of adjacent word-pair tuples."""
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_claims_simple(response: str) -> list[str]:
    """Split *response* into atomic claims via sentence-boundary detection.

    Replicates the logic in faithfulness.py without importing it (avoids
    circular import).  Fragments shorter than 4 words are discarded.
    """
    # Split on sentence-ending punctuation followed by whitespace
    raw_sentences = _SENTENCE_END_RE.split(response.strip())
    claims: list[str] = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Also split on newlines so multi-paragraph responses are handled
        for part in sentence.splitlines():
            part = part.strip().rstrip(".,!?;:")
            if not part:
                continue
            # Require at least 3 words to be a meaningful claim
            if len(part.split()) >= 3:
                claims.append(part)
    return claims


def claim_chunk_overlap(claim: str, chunk_text: str) -> float:
    """Combined lexical similarity between *claim* and *chunk_text*.

    Score = 0.6 * word_jaccard + 0.4 * bigram_jaccard
    Both components use lowercased, stop-word-stripped token sets.
    """
    claim_words = set(_tokenize(claim))
    chunk_words = set(_tokenize(chunk_text))

    # Word-level Jaccard
    if claim_words or chunk_words:
        word_jaccard = (
            len(claim_words & chunk_words) / len(claim_words | chunk_words)
        )
    else:
        word_jaccard = 1.0

    # Bigram-level Jaccard
    claim_bigrams = set(_bigrams(_tokenize(claim)))
    chunk_bigrams = set(_bigrams(_tokenize(chunk_text)))
    if claim_bigrams or chunk_bigrams:
        bigram_jaccard = (
            len(claim_bigrams & chunk_bigrams) / len(claim_bigrams | chunk_bigrams)
        )
    else:
        bigram_jaccard = 1.0 if (not claim_bigrams and not chunk_bigrams) else 0.0

    return 0.6 * word_jaccard + 0.4 * bigram_jaccard


def support_set(
    claim: str,
    chunks: list[tuple[str, str]],
    threshold: float = 0.6,
) -> set[str]:
    """Return chunk ids whose overlap with *claim* meets or exceeds *threshold*.

    Args:
        claim:     An atomic factual claim extracted from a model response.
        chunks:    Sequence of ``(chunk_id, chunk_text)`` pairs.
        threshold: Minimum combined overlap score required for a chunk to be
                   considered supporting.  Defaults to 0.6.

    Returns:
        Set of chunk_id strings whose ``claim_chunk_overlap`` with *claim*
        is >= *threshold*.
    """
    supporting: set[str] = set()
    for chunk_id, chunk_text in chunks:
        if claim_chunk_overlap(claim, chunk_text) >= threshold:
            supporting.add(chunk_id)
    return supporting


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets of chunk ids."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score(
    response_a: str,
    response_b: str,
    chunks_a: list[tuple[str, str]],
    chunks_b: list[tuple[str, str]],
    k: int = 5,
) -> AttributionResult:
    """Compute Retrieval-Attribution Jaccard@k between two equivalent responses.

    Algorithm:
      1. Extract atomic claims from each response.
      2. For every claim in *response_a*, find its supporting chunks inside
         ``chunks_a[:k]``.  Union all per-claim support sets → ``supporting_a``.
      3. Same for *response_b* / ``chunks_b[:k]`` → ``supporting_b``.
      4. Jaccard score = |supporting_a ∩ supporting_b| / |supporting_a ∪ supporting_b|.

    Args:
        response_a: Model output for the original prompt.
        response_b: Model output for a paraphrased prompt.
        chunks_a:   Retrieved chunks for the original prompt (ordered by rank).
        chunks_b:   Retrieved chunks for the paraphrased prompt (ordered by rank).
        k:          Number of top-ranked chunks to consider.

    Returns:
        :class:`~aria_audit.core.AttributionResult` with Jaccard@k score and
        the flat lists of supporting chunk ids for each response.
    """
    top_a = chunks_a[:k]
    top_b = chunks_b[:k]

    claims_a = extract_claims_simple(response_a)
    claims_b = extract_claims_simple(response_b)

    supporting_a: set[str] = set()
    for claim in claims_a:
        supporting_a |= support_set(claim, top_a)

    supporting_b: set[str] = set()
    for claim in claims_b:
        supporting_b |= support_set(claim, top_b)

    jaccard_score = jaccard(supporting_a, supporting_b)

    return AttributionResult(
        jaccard_at_k=jaccard_score,
        k=k,
        supporting_chunks_a=sorted(supporting_a),
        supporting_chunks_b=sorted(supporting_b),
    )


def score_single(
    response: str,
    chunks: list[tuple[str, str]],
    k: int = 5,
) -> dict[str, list[str]]:
    """Return ``{claim: [supporting_chunk_ids]}`` for dashboard display.

    Convenience wrapper around :func:`support_set` that processes a single
    response against its retrieved chunks.  Used by ``AuditEnvelopePanel``
    to render per-claim attribution breakdowns without needing a paired
    paraphrased response.

    Args:
        response: A single model response to analyse.
        chunks:   Retrieved chunks associated with the response (ordered by rank).
        k:        Number of top-ranked chunks to consider.

    Returns:
        Ordered dict mapping each extracted claim to its list of supporting
        chunk ids (empty list when no chunk meets the threshold).
    """
    top = chunks[:k]
    claims = extract_claims_simple(response)
    return {claim: sorted(support_set(claim, top)) for claim in claims}
