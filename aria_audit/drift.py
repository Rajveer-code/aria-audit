"""Streaming-drift detection over per-axis time series.

Replaces the "per-response statistical significance" axis from the original
CPFE classifier setting, which is statistically awkward for a single LLM call.
Page-Hinkley is a one-sided sequential CUSUM-style test for change-point
detection on streaming univariate data.

Reference:
  - Page, E. S. (1954). "Continuous Inspection Schemes." Biometrika 41 (1/2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aria_audit.core import DriftSignal


@dataclass
class PageHinkley:
    """One-sided Page-Hinkley test.

    Triggers when cumulative deviation from running mean exceeds `lambda_`.
    Magnitude `delta` is the minimum change considered worth detecting.
    """

    delta: float = 0.005
    lambda_: float = 0.05
    alpha: float = 0.999  # running-mean forgetting factor

    n: int = 0
    mean: float = 0.0
    cumsum: float = 0.0
    min_cumsum: float = 0.0
    last_reset_t: float = field(default=0.0)

    def update(self, x: float, t: float = 0.0) -> bool:
        """Push observation; return True if drift alarm fires this step."""
        self.n += 1
        self.mean = self.alpha * self.mean + (1 - self.alpha) * x if self.n > 1 else x
        self.cumsum += x - self.mean - self.delta
        if self.cumsum < self.min_cumsum:
            self.min_cumsum = self.cumsum
        alarmed = (self.cumsum - self.min_cumsum) > self.lambda_
        if alarmed:
            self.cumsum = 0.0
            self.min_cumsum = 0.0
            self.last_reset_t = t
        return alarmed

    def snapshot(self, axis: str, t: float = 0.0) -> DriftSignal:
        return DriftSignal(axis=axis, cumsum=self.cumsum, alarmed=False, last_reset_at=self.last_reset_t)
