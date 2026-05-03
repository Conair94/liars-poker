"""AR-2 Phase 4 — inference-time entropy floor (design §5.3, §4.1)."""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agents.learned.bidpolicy.network import _entropy, apply_entropy_floor
from game.bids import NUM_BIDS

_FLOOR = {2: 0.6, 3: 0.5, 4: 0.4}


def _random_pi(rng: np.random.Generator, mask: np.ndarray, peakedness: float = 1.0) -> np.ndarray:
    """Random distribution supported on `mask`. `peakedness` scales the logits."""
    logits = rng.standard_normal(NUM_BIDS) * peakedness
    logits[~mask] = -np.inf
    m = np.exp(logits - np.max(logits[mask]))
    m[~mask] = 0.0
    s = m.sum()
    if s <= 0.0:
        out = np.zeros_like(m)
        out[mask] = 1.0 / mask.sum()
        return out.astype(np.float32)
    return (m / s).astype(np.float32)


def test_floor_noop_above_floor():
    rng = np.random.default_rng(0)
    mask = np.zeros(NUM_BIDS, dtype=np.bool_)
    mask[:20] = True
    # Use a uniform pi over the support — already maximizes entropy.
    pi = mask.astype(np.float32) / mask.sum()
    out = apply_entropy_floor(pi, mask, n=2, floor_frac=_FLOOR)
    np.testing.assert_array_equal(out, pi)


def test_floor_at_n_5_is_noop():
    rng = np.random.default_rng(0)
    mask = np.zeros(NUM_BIDS, dtype=np.bool_)
    mask[:10] = True
    pi = _random_pi(rng, mask, peakedness=5.0)  # peaked
    out = apply_entropy_floor(pi, mask, n=5, floor_frac=_FLOOR)
    # n=5 not in floor_frac → returns input unchanged.
    np.testing.assert_array_equal(out, pi)


def test_floor_raises_to_target():
    rng = np.random.default_rng(1)
    mask = np.zeros(NUM_BIDS, dtype=np.bool_)
    mask[:8] = True
    pi = _random_pi(rng, mask, peakedness=8.0)  # very peaked
    H_target = _FLOOR[2] * float(np.log(int(mask.sum())))

    out = apply_entropy_floor(pi, mask, n=2, floor_frac=_FLOOR)
    assert _entropy(out) >= H_target - 1e-3
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-5)
    # Floor should not put mass on infeasible actions.
    assert (out[~mask] == 0.0).all()


def test_floor_property_n2():
    """100 random peaked pi at n=2: all satisfy the floor."""
    rng = np.random.default_rng(42)
    fails = 0
    for trial in range(100):
        size = int(rng.integers(2, 30))
        mask = np.zeros(NUM_BIDS, dtype=np.bool_)
        idxs = rng.choice(NUM_BIDS, size=size, replace=False)
        mask[idxs] = True
        pi = _random_pi(rng, mask, peakedness=float(rng.uniform(0.5, 6.0)))
        out = apply_entropy_floor(pi, mask, n=2, floor_frac=_FLOOR)
        H_target = _FLOOR[2] * float(np.log(size))
        if _entropy(out) < H_target - 1e-3:
            fails += 1
    assert fails == 0, f"{fails}/100 trials failed the n=2 entropy floor"
