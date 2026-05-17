"""Tests for `AuditEnvelope` and result dataclasses. CPU-only, no model deps."""

from __future__ import annotations

import pytest

from aria_audit.core import (
    AttributionResult,
    AuditEnvelope,
    CalibrationResult,
    ConsistencyResult,
    EquityResult,
    FaithfulnessResult,
)


def test_envelope_roundtrip_json():
    env = AuditEnvelope(
        request_id="r1",
        model_name="qwen3:8b",
        prompt="hello",
        response="world",
    )
    s = env.to_json()
    assert '"schema_version"' in s
    assert '"prompt": "hello"' in s


def test_composite_score_empty():
    assert AuditEnvelope().composite_score == 0.0


def test_composite_score_weighted():
    env = AuditEnvelope(
        calibration=CalibrationResult(ece_overall=0.05, ece_per_group={}, confidence=0.8, confidence_bin="high"),
        faithfulness=FaithfulnessResult(hhem_score=0.9, claims_total=4, claims_supported=4, claims_unsupported=[], raga_faithfulness=1.0),
        consistency=ConsistencyResult(semantic_entropy=0.2, n_samples=3, n_meaning_clusters=1, majority_share=1.0),
        equity=EquityResult(disparate_impact=1.0, equalized_odds_gap=0.0, groups_tested=["a", "b"], response_property="sentiment", counterfactual_pairs_n=10),
        attribution=AttributionResult(jaccard_at_k=0.8, k=5, supporting_chunks_a=[], supporting_chunks_b=[]),
    )
    # All axes near-perfect: composite should be high (>80).
    assert env.composite_score > 80.0


def test_drift_alarm_fires_under_step_change():
    from aria_audit.drift import PageHinkley
    ph = PageHinkley(delta=0.005, lambda_=0.1)
    for _ in range(50):
        ph.update(0.0)
    fired = False
    for _ in range(50):
        if ph.update(1.0):
            fired = True
            break
    assert fired


def test_calibration_ece_perfect():
    import numpy as np
    from aria_audit.axes.calibration import expected_calibration_error
    confs = np.array([0.05, 0.95, 0.5])
    correct = np.array([0.0, 1.0, 1.0])  # 0.5 conf but correct
    ece = expected_calibration_error(confs, correct, n_bins=10)
    assert 0.0 <= ece <= 1.0


def test_jaccard_edge_cases():
    from aria_audit.axes.attribution import jaccard
    assert jaccard(set(), set()) == 1.0
    assert jaccard(set("a"), set()) == 0.0
    assert jaccard({"a", "b"}, {"a", "c"}) == pytest.approx(1 / 3)


def test_gpu_manager_anchor_reuse():
    from aria_audit.gpu_manager import GPUManager, QWEN3_8B
    mgr = GPUManager()
    counter = {"loads": 0}

    def fake_loader():
        counter["loads"] += 1
        return object()

    with mgr.acquire(QWEN3_8B, fake_loader, lambda _: None) as a:
        with mgr.acquire(QWEN3_8B, fake_loader, lambda _: None) as b:
            assert a is b
    assert counter["loads"] == 1
