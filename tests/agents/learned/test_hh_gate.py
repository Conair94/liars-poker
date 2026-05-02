"""AR-2 Phase 2: HH-gate truth table (design §7.3).

Verifies `should_declare_hh` fires iff `argmax(q) == standing_bid` AND
`q[standing_bid] >= hh_band * q.max()`, across 100 random (q, standing_bid)
pairs spanning a sweep of `hh_band` values.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.learned.hh_gate import should_declare_hh
from game.bids import NUM_BIDS


def _sample_distribution(rng: np.random.Generator) -> np.ndarray:
    """Random simplex point over NUM_BIDS, with occasional sparsity."""
    if rng.random() < 0.3:
        # Concentrated: a small support set with random weights.
        k = int(rng.integers(2, 8))
        idx = rng.choice(NUM_BIDS, size=k, replace=False)
        q = np.zeros(NUM_BIDS, dtype=np.float64)
        q[idx] = rng.dirichlet(np.ones(k))
    else:
        q = rng.dirichlet(np.ones(NUM_BIDS) * 0.5)
    return q


def _reference_should_fire(q: np.ndarray, standing_bid: int, hh_band: float) -> bool:
    """Direct re-statement of the design §4.4 spec."""
    if standing_bid is None:
        return False
    q_max = float(q.max())
    if q_max <= 0.0:
        return False
    return int(np.argmax(q)) == int(standing_bid) and float(q[int(standing_bid)]) >= hh_band * q_max


def test_truth_table_random_pairs() -> None:
    rng = np.random.default_rng(0xA12)
    for _ in range(100):
        q = _sample_distribution(rng)
        standing_bid = int(rng.integers(0, NUM_BIDS))
        hh_band = float(rng.uniform(0.5, 1.0))
        expected = _reference_should_fire(q, standing_bid, hh_band)
        actual = should_declare_hh(q, standing_bid, hh_band=hh_band)
        assert actual == expected, (
            f"mismatch: q[argmax={int(np.argmax(q))}]={q.max():.4f}, "
            f"standing_bid={standing_bid}, q[bi]={q[standing_bid]:.4f}, "
            f"hh_band={hh_band:.3f}, expected={expected}, actual={actual}"
        )


def test_standing_bid_none_never_fires() -> None:
    rng = np.random.default_rng(1)
    q = _sample_distribution(rng)
    assert not should_declare_hh(q, None)
    assert not should_declare_hh(q, None, hh_band=0.0)


def test_fires_when_standing_bid_is_argmax() -> None:
    q = np.zeros(NUM_BIDS, dtype=np.float64)
    q[42] = 0.6
    q[10] = 0.2
    q[15] = 0.2
    assert should_declare_hh(q, 42, hh_band=0.95)
    assert should_declare_hh(q, 42, hh_band=0.5)


def test_does_not_fire_when_argmax_differs() -> None:
    # Standing bid is the second-highest, within hh_band of peak.
    # Old ExactRulesConditional would have fired here; the design's strict
    # gate must NOT.
    q = np.zeros(NUM_BIDS, dtype=np.float64)
    q[42] = 0.40   # peak
    q[10] = 0.39   # close second — would pass the old `cur_p >= hh_band*peak_p` test
    q[5]  = 0.21
    assert not should_declare_hh(q, 10, hh_band=0.9)
    assert not should_declare_hh(q, 10, hh_band=0.5)
    assert should_declare_hh(q, 42, hh_band=0.9)


def test_hh_band_boundary() -> None:
    q = np.zeros(NUM_BIDS, dtype=np.float64)
    q[7] = 0.5
    q[12] = 0.5  # tied peak; argmax returns the first occurrence
    # argmax is 7 → only 7 can fire.
    assert should_declare_hh(q, 7, hh_band=1.0)
    assert not should_declare_hh(q, 12, hh_band=1.0)


def test_zero_distribution_never_fires() -> None:
    q = np.zeros(NUM_BIDS, dtype=np.float64)
    assert not should_declare_hh(q, 0)
    assert not should_declare_hh(q, 50)


@pytest.mark.parametrize("hh_band", [0.0, 0.5, 0.9, 0.95, 1.0])
def test_band_monotonic(hh_band: float) -> None:
    """If gate fires at hh_band=b, it also fires at any b' < b (more permissive)."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        q = _sample_distribution(rng)
        bi = int(np.argmax(q))
        if should_declare_hh(q, bi, hh_band=hh_band):
            for b_lower in [0.0, hh_band * 0.5]:
                assert should_declare_hh(q, bi, hh_band=b_lower)
