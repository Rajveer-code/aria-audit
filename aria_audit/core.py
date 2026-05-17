"""Core data structures for ARIA-Audit.

`AuditEnvelope` is the structured telemetry vector emitted alongside every
LLM response. Each axis maps 1:1 to a CPFE-paper construct, honestly
reframed where the original definition does not transfer to free-form
generative outputs (see plan v3 for details).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class CalibrationResult:
    """Group-conditional ECE on verbalized confidence (Tian 2023; QA-Calibration ICLR'25)."""

    ece_overall: float
    ece_per_group: dict[str, float]
    confidence: float
    confidence_bin: str
    n_bins: int = 10


@dataclass(frozen=True)
class FaithfulnessResult:
    """HHEM 2.1 NLI score + RAGAS-style claim breakdown."""

    hhem_score: float          # P(supported) from HHEM 2.1
    claims_total: int
    claims_supported: int
    claims_unsupported: list[str]
    raga_faithfulness: float   # claims_supported / claims_total


@dataclass(frozen=True)
class ConsistencyResult:
    """Semantic entropy under sampled paraphrase (Kuhn 2023)."""

    semantic_entropy: float
    n_samples: int
    n_meaning_clusters: int
    majority_share: float


@dataclass(frozen=True)
class EquityResult:
    """Counterfactual DI + EOD on demographic substitution (HolisticBias-style)."""

    disparate_impact: float        # P(positive|A) / P(positive|B)
    equalized_odds_gap: float      # max TPR gap across groups
    groups_tested: list[str]
    response_property: str         # 'sentiment' | 'refusal' | 'recommendation_valence'
    counterfactual_pairs_n: int


@dataclass(frozen=True)
class AttributionResult:
    """Retrieval-attribution Jaccard@k under paraphrase.

    NOTE: This is the honest reframe of CPFE Axis 5. The original Jaccard-of-feature-
    vocabularies + Captum integrated-gradients does not work through Ollama (no
    gradient access). We measure: for each claim in the response, which of the
    top-k retrieved chunks support it, then take Jaccard of supporting-chunk-sets
    across paraphrased prompts.
    """

    jaccard_at_k: float
    k: int
    supporting_chunks_a: list[str]
    supporting_chunks_b: list[str]


@dataclass(frozen=True)
class DriftSignal:
    """Page-Hinkley / CUSUM streaming-drift alarm state per axis."""

    axis: str
    cumsum: float
    alarmed: bool
    last_reset_at: float


@dataclass
class AuditEnvelope:
    """Structured five-axis telemetry vector emitted per response.

    This is the artifact: it is logged, joined, plotted, and reported. Every field
    here corresponds to a column in the eval-results CSV and a panel in the demo
    dashboard. Schema changes are breaking — bump the version when modifying.
    """

    SCHEMA_VERSION: str = "0.1.0"

    request_id: str = ""
    model_name: str = ""
    prompt: str = ""
    response: str = ""
    retrieved_chunk_ids: list[str] = field(default_factory=list)

    calibration: CalibrationResult | None = None
    faithfulness: FaithfulnessResult | None = None
    consistency: ConsistencyResult | None = None
    equity: EquityResult | None = None
    attribution: AttributionResult | None = None

    drift: list[DriftSignal] = field(default_factory=list)

    latency_ms_generation: float = 0.0
    latency_ms_audit: float = 0.0
    peak_vram_gb: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = self.SCHEMA_VERSION
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @property
    def composite_score(self) -> float:
        """Weighted 0–100 score for dashboard display.

        Weights are deliberate, not tuned: equity and faithfulness are the
        load-bearing axes for safety, so each gets 30%. Calibration 20%,
        consistency 10%, attribution 10%.
        """
        parts: list[tuple[float, float]] = []
        if self.calibration is not None:
            parts.append((0.20, max(0.0, 1.0 - self.calibration.ece_overall)))
        if self.faithfulness is not None:
            parts.append((0.30, self.faithfulness.hhem_score))
        if self.consistency is not None:
            parts.append((0.10, max(0.0, 1.0 - self.consistency.semantic_entropy)))
        if self.equity is not None:
            di = self.equity.disparate_impact
            equity_score = 1.0 - abs(1.0 - di) if di > 0 else 0.0
            parts.append((0.30, max(0.0, equity_score)))
        if self.attribution is not None:
            parts.append((0.10, self.attribution.jaccard_at_k))
        if not parts:
            return 0.0
        total_w = sum(w for w, _ in parts)
        return 100.0 * sum(w * v for w, v in parts) / total_w
