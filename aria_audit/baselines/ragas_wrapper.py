"""RAGAS baseline — CPU-only approximation of Faithfulness + Answer Relevancy.

DISCLAIMER / APPROXIMATION NOTE
---------------------------------
The published RAGAS framework (Es et al. 2023, arXiv:2309.15217) uses an LLM
judge to (a) decompose the response into atomic claims, (b) verify each claim
against the retrieved context via NLI, and (c) score answer relevancy by having
the LLM generate back-questions and comparing their embeddings to the original.

This module is a *CPU-only proxy* that avoids any LLM call:

  - **Faithfulness proxy**: TF-IDF cosine similarity between each response
    sentence and the full context string; the average sentence-level similarity
    is returned as `faithfulness`.  This approximates claim-level entailment
    without an NLI model.  It will over-score fluent hallucinations and
    under-score paraphrased-but-faithful claims — acceptable for baseline
    comparison but not a replacement for the full pipeline.

  - **Answer relevancy proxy**: TF-IDF cosine similarity between the full
    response and the question.  When no question is provided the metric is
    undefined; we return 0.5 as a neutral sentinel.

Use these scores only to *rank* models relative to each other (i.e. as a
comparative baseline), not as absolute faithfulness guarantees.  The full RAGAS
pipeline (with an LLM judge) is the Phase 3 target.

sklearn (TfidfVectorizer + cosine_similarity) is loaded lazily so that the
rest of ARIA-Audit can import this module even in environments without sklearn.
If sklearn is unavailable, both metrics fall back to word-overlap Jaccard
similarity.

References
----------
- Es et al. (2023). "RAGAS: Automated Evaluation of Retrieval Augmented
  Generation." arXiv:2309.15217
- Faithfulness / claim-level NLI conceptual background: Maynez et al. (2020).
  "On Faithfulness and Factuality in Abstractive Summarization." ACL 2020.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Naively split *text* into sentences on terminal punctuation."""
    # Split on '.', '!', '?' followed by whitespace or end-of-string.
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _jaccard(a: str, b: str) -> float:
    """Word-overlap Jaccard similarity as a sklearn-free fallback."""
    tokens_a = set(re.findall(r"\b\w+\b", a.lower()))
    tokens_b = set(re.findall(r"\b\w+\b", b.lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _tfidf_cosine(texts_a: list[str], texts_b: list[str]) -> list[float]:
    """Return pairwise cosine similarities between corresponding pairs.

    texts_a[i] is compared to texts_b[i].  All texts are fitted together so
    that the vocabulary is shared.

    Raises ImportError transparently — callers should catch and fall back.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # lazy import
    from sklearn.metrics.pairwise import cosine_similarity  # lazy import

    all_texts = texts_a + texts_b
    vec = TfidfVectorizer(min_df=1)
    tfidf = vec.fit_transform(all_texts)

    n = len(texts_a)
    vecs_a = tfidf[:n]
    vecs_b = tfidf[n:]

    scores: list[float] = []
    for i in range(n):
        sim = cosine_similarity(vecs_a[i], vecs_b[i])[0, 0]
        scores.append(float(sim))
    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score(
    response: str,
    context: str,
    question: str | None = None,
) -> dict[str, float]:
    """Compute approximate RAGAS Faithfulness and Answer Relevancy scores.

    Parameters
    ----------
    response:
        The model-generated response to evaluate.
    context:
        The retrieved context passage(s) that the response should be grounded
        in (concatenate multiple chunks with a newline before passing).
    question:
        The original user question.  When omitted, ``answer_relevancy`` is
        returned as ``0.5`` (undefined / neutral sentinel).

    Returns
    -------
    dict with keys:
        ``"faithfulness"`` — float in [0, 1].  Average sentence-level TF-IDF
        cosine similarity between each response sentence and the full context.
        Higher = more faithful.

        ``"answer_relevancy"`` — float in [0, 1].  TF-IDF cosine similarity
        between the full response and the question.  Higher = more on-topic.

    Notes
    -----
    See module docstring for a full explanation of the approximation.
    """
    # --- faithfulness ---
    sentences = _split_sentences(response)
    if not sentences:
        # Empty response: no claims to check → treat as 0 faithfulness
        faithfulness = 0.0
    else:
        try:
            context_repeated = [context] * len(sentences)
            sims = _tfidf_cosine(sentences, context_repeated)
            faithfulness = sum(sims) / len(sims)
        except ImportError:
            # sklearn not available — fall back to Jaccard per sentence
            sims = [_jaccard(sent, context) for sent in sentences]
            faithfulness = sum(sims) / len(sims)

    # --- answer relevancy ---
    if question is None:
        answer_relevancy = 0.5
    else:
        try:
            ar_list = _tfidf_cosine([response], [question])
            answer_relevancy = ar_list[0]
        except ImportError:
            answer_relevancy = _jaccard(response, question)

    return {
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(answer_relevancy, 4),
    }
