"""ARIA-Audit: runtime five-axis fairness audit for locally-deployed conversational LLMs.

Operationalizes the CPFE (Cross-Platform Fairness Evaluation) framework on deployed
inference, closing the audit-runtime gap between offline evaluation harnesses
(HELM, RAGAS) and inline policy guardrails (Granite Guardian, NeMo).

Axes:
  - Calibration:   group-conditional ECE on verbalized confidence
  - Faithfulness:  HHEM 2.1 + RAGAS-style claim extraction
  - Consistency:   semantic entropy via sampled paraphrase + NLI clustering
  - Equity:        DI + EOD via counterfactual demographic substitution
  - Attribution:   retrieval-attribution Jaccard@k under paraphrase
"""

from aria_audit.core import AuditEnvelope

__version__ = "0.1.0"
__author__ = "Rajveer Singh Pall"

__all__ = ["AuditEnvelope", "__version__"]
