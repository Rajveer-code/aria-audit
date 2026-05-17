"""Axis 1 — Group-conditional ECE on verbalized confidence.

Phase 1 implementation target. Skeleton here to lock the public API.

References:
  - Tian et al. 2023 "Just Ask for Calibration" arXiv:2305.14975
  - QA-Calibration ICLR 2025
  - Kadavath et al. 2022 "Language Models (Mostly) Know What They Know" arXiv:2207.05221
"""

from __future__ import annotations

import numpy as np

from aria_audit.core import CalibrationResult

VERBAL_BIN_MAP: dict[str, float] = {
    "very low": 0.10, "low": 0.25, "medium": 0.50,
    "high": 0.75, "very high": 0.90, "certain": 0.99,
}


def parse_verbalized_confidence(text: str) -> tuple[float, str]:
    """Extract a verbalized confidence label + numeric centerpoint from response text.

    Phase 1 implementation; placeholder pattern-match here.
    """
    lower = text.lower()
    for label, val in sorted(VERBAL_BIN_MAP.items(), key=lambda kv: -len(kv[0])):
        if label in lower:
            return val, label
    return 0.5, "medium"  # default if no verbal confidence emitted


def expected_calibration_error(
    confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10
) -> float:
    """Standard ECE with M equal-width bins. Reused from CPFE pipeline (calibration_comparison.py)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if not mask.any():
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correctness[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def score(
    confidences: np.ndarray,
    correctness: np.ndarray,
    groups: np.ndarray | None = None,
    n_bins: int = 10,
) -> CalibrationResult:
    """Compute scalar + per-group ECE.

    Phase 1 will wire this into `orchestrator.run_eval`. The function itself is
    correct and unit-testable now.
    """
    overall = expected_calibration_error(confidences, correctness, n_bins)
    per_group: dict[str, float] = {}
    if groups is not None:
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() < n_bins:
                continue
            per_group[str(g)] = expected_calibration_error(
                confidences[mask], correctness[mask], n_bins
            )
    return CalibrationResult(
        ece_overall=overall,
        ece_per_group=per_group,
        confidence=float(confidences.mean()) if len(confidences) else 0.0,
        confidence_bin="batch",
        n_bins=n_bins,
    )
