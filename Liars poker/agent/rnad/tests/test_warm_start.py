"""
Tests for agent.rnad.warm_start. Run via:

    cd "papers/Liars poker/"
    python -m agent.rnad.tests.test_warm_start
"""

import sys, os
_PAPER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PAPER_DIR not in sys.path:
    sys.path.insert(0, _PAPER_DIR)

import numpy as np
from agent.rnad.warm_start import WarmStartLookup, match_condition, _KNOWN_CONDITIONS
from agent.game.bids import NUM_BIDS

# Shared fixture — load once for the whole test module.
_LOOKUP: WarmStartLookup = None

def _get_lookup() -> WarmStartLookup:
    global _LOOKUP
    if _LOOKUP is None:
        _LOOKUP = WarmStartLookup()
    return _LOOKUP


# ---------------------------------------------------------------------------
# match_condition tests (no IO needed — uses _KNOWN_CONDITIONS populated by
# the first WarmStartLookup instantiation)
# ---------------------------------------------------------------------------

def test_match_trips():
    _get_lookup()
    # A♠=51, A♥=50, A♦=49
    assert match_condition([51, 50, 49]) == "trips_A"
    # 2♠=3, 2♥=2, 2♦=1
    assert match_condition([3, 2, 1]) == "trips_2"
    print("  test_match_trips: PASS")


def test_match_pair():
    _get_lookup()
    # 9♠=31, 9♥=30
    assert match_condition([31, 30]) == "pair_9"
    # K♠=47, K♦=45
    assert match_condition([47, 45]) == "pair_K"
    print("  test_match_pair: PASS")


def test_match_3suited():
    _get_lookup()
    # K♠=47, Q♠=43, 9♠=31
    assert match_condition([47, 43, 31]) == "3suited_high_K"
    print("  test_match_3suited: PASS")


def test_match_suited():
    _get_lookup()
    # A♠=51, J♠=39
    assert match_condition([51, 39]) == "suited_high_A"
    print("  test_match_suited: PASS")


def test_match_adjacent():
    _get_lookup()
    # Q♠=43 (rank 10, suit♠), K♦=45 (rank 11, suit♦) — adjacent, different suits
    assert match_condition([43, 45]) == "adjacent_low_Q"
    # 2♠=3 (rank 0, suit♠), 3♦=5 (rank 1, suit♦) — adjacent, different suits
    assert match_condition([3, 5]) == "adjacent_low_2"
    print("  test_match_adjacent: PASS")


def test_match_3range():
    _get_lookup()
    # 5♠=15 (rank 3), 6♣=16 (rank 4), 7♦=21 (rank 5)  — max-min = 2 ≤ 4
    assert match_condition([15, 16, 21]) == "3range_low_5"
    print("  test_match_3range: PASS")


def test_match_none():
    _get_lookup()
    # A♠=51 (rank 12), 3♦=5 (rank 1): gap of 11, not adjacent, not suited
    assert match_condition([51, 5]) is None
    # Single card: no condition
    assert match_condition([51]) is None
    print("  test_match_none: PASS")


def test_trips_beats_pair():
    """When a hand has both trips and pair(s), trips takes priority."""
    _get_lookup()
    # A♠=51, A♥=50, A♦=49, K♠=47, K♥=46  (trips Aces + pair Kings)
    result = match_condition([51, 50, 49, 47, 46])
    assert result == "trips_A", f"Expected trips_A, got {result}"
    print("  test_trips_beats_pair: PASS")


def test_pair_beats_suited():
    """
    When a hand has both a pair and 2 suited cards, pair takes priority.
    9♠=31, 9♦=29 (pair 9s, different suits) + K♠=47, 7♠=27 (suited spades)
    """
    _get_lookup()
    result = match_condition([31, 29, 47, 27])
    assert result == "pair_9", f"Expected pair_9, got {result}"
    print("  test_pair_beats_suited: PASS")


# ---------------------------------------------------------------------------
# WarmStartLookup.get_features tests
# ---------------------------------------------------------------------------

def test_valid_probability_distributions():
    """marginal_vec and conditional_vec must be non-negative and sum to 1."""
    lookup = _get_lookup()
    test_cases = [
        ([31, 30], 10),      # pair_9, n=10
        ([51, 50, 49], 8),   # trips_A, n=8
        ([47, 43, 31], 15),  # 3suited_high_K, n=15
        ([51, 39], 12),      # suited_high_A, n=12
        ([43, 47], 7),       # adjacent_low_Q, n=7
        ([15, 16, 21], 9),   # 3range_low_5, n=9
        ([51, 5], 10),       # no condition, n=10
    ]
    for hand, n in test_cases:
        m, c, k = lookup.get_features(hand, n)
        assert m.dtype == np.float32
        assert c.dtype == np.float32
        assert m.shape == (NUM_BIDS,)
        assert c.shape == (NUM_BIDS,)
        assert (m >= 0).all(),           f"negative marginal for hand={hand}, n={n}"
        assert (c >= 0).all(),           f"negative conditional for hand={hand}, n={n}"
        assert abs(m.sum() - 1.0) < 1e-4, f"marginal not normalized: {m.sum()}"
        assert abs(c.sum() - 1.0) < 1e-4, f"cond not normalized: {c.sum()}"
    print("  test_valid_probability_distributions: PASS")


def test_no_condition_returns_marginal():
    """When no condition matches, conditional_vec must equal marginal_vec."""
    lookup = _get_lookup()
    m, c, key = lookup.get_features([51, 5], n=10)  # A♠ 3♦ — no condition
    assert key is None
    assert np.allclose(m, c), "conditional should equal marginal when key is None"
    print("  test_no_condition_returns_marginal: PASS")


def test_condition_shifts_distribution():
    """A pair in hand should meaningfully shift P(pool has Pair or better)."""
    lookup = _get_lookup()
    # pair of 9s, n=10
    m, c, key = lookup.get_features([31, 30], n=10)
    assert key == "pair_9"
    assert not np.allclose(m, c), "conditional must differ from marginal"

    from agent.game.bids import all_bids, PAIR
    bids = all_bids()
    pair_and_above_indices = [i for i, b in enumerate(bids) if b.hand_type >= PAIR]

    marginal_pair_plus = m[pair_and_above_indices].sum()
    cond_pair_plus    = c[pair_and_above_indices].sum()
    # Holding a pair raises P(pool >= pair) vs the marginal
    assert cond_pair_plus > marginal_pair_plus, (
        f"Pair in hand should increase P(pool >= pair): "
        f"marginal={marginal_pair_plus:.4f}, cond={cond_pair_plus:.4f}"
    )
    print(f"  test_condition_shifts_distribution: PASS  "
          f"(P(pool>=pair): marginal={marginal_pair_plus:.3f}, cond={cond_pair_plus:.3f})")


def test_out_of_range_n():
    """Pool sizes outside 5..25 return a uniform distribution safely."""
    lookup = _get_lookup()
    for n in [1, 2, 3, 4, 26, 30]:
        m, c, key = lookup.get_features([31, 30], n=n)
        assert key is None
        assert abs(m.sum() - 1.0) < 1e-4
        assert np.allclose(m, c)
    print("  test_out_of_range_n: PASS")


def test_aux_target_matches_conditional():
    """get_aux_target must return the same vector as the conditional in get_features."""
    lookup = _get_lookup()
    m, c, key = lookup.get_features([31, 30], n=10)
    assert key == "pair_9"
    target = lookup.get_aux_target(key, 10)
    assert target is not None
    assert np.allclose(c, target)
    print("  test_aux_target_matches_conditional: PASS")


def test_aux_target_none_for_no_condition():
    """get_aux_target(None, n) returns None."""
    lookup = _get_lookup()
    assert lookup.get_aux_target(None, 10) is None
    print("  test_aux_target_none_for_no_condition: PASS")


def test_cache_load_is_fast():
    """Second instantiation (cache hit) should complete without running MC."""
    import time
    t0 = time.time()
    WarmStartLookup()
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"Cache load took {elapsed:.1f}s — expected < 5s"
    print(f"  test_cache_load_is_fast: PASS  ({elapsed:.2f}s)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running warm_start tests...\n")
    tests = [
        test_match_trips,
        test_match_pair,
        test_match_3suited,
        test_match_suited,
        test_match_adjacent,
        test_match_3range,
        test_match_none,
        test_trips_beats_pair,
        test_pair_beats_suited,
        test_valid_probability_distributions,
        test_no_condition_returns_marginal,
        test_condition_shifts_distribution,
        test_out_of_range_n,
        test_aux_target_matches_conditional,
        test_aux_target_none_for_no_condition,
        test_cache_load_is_fast,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  {t.__name__}: FAIL — {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed.")
    if failed:
        sys.exit(1)
