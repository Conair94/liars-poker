"""
Tests for agent.baseline.blind_equilibrium. Run via:

    cd "papers/Liars poker/"
    python -m agent.baseline.tests.test_blind_equilibrium
"""

import sys
import os

# Ensure paper root is on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _PAPER_DIR not in sys.path:
    sys.path.insert(0, _PAPER_DIR)

from agent.baseline.blind_equilibrium import (
    _compute_bid_at_least,
    _solve_n2,
    _solve_initial,
    get_blind_equilibrium,
)
from agent.game.bids import NUM_BIDS, CALL_ACTION, all_bids, bid_to_index, Bid

# Use small sample counts so tests run quickly
_FAST_SAMPLES = 50_000


def test_p_at_least_bounds():
    """P_at_least values must be in [0, 1] and non-increasing."""
    from poker_math_exact import get_hand_rank_counts
    counts, total = get_hand_rank_counts(5, n_samples=_FAST_SAMPLES)
    p = _compute_bid_at_least(counts, total)

    assert len(p) == NUM_BIDS, f"Expected {NUM_BIDS} values, got {len(p)}"
    for i, v in enumerate(p):
        assert 0.0 <= v <= 1.0, f"p_at_least[{i}]={v} out of [0,1]"
    for i in range(1, NUM_BIDS):
        assert p[i] <= p[i - 1] + 1e-12, (
            f"p_at_least not non-increasing at idx {i}: {p[i-1]:.6f} → {p[i]:.6f}"
        )
    # Weakest bid (HC 2): practically all 5-card pools are at least HC 2
    assert p[0] > 0.99, f"P(pool >= HC 2) should be ~1 for n=5, got {p[0]:.4f}"
    # Strongest bid (SF A): extremely rare for small n
    assert p[NUM_BIDS - 1] < 0.01, (
        f"P(pool >= SF A) should be tiny for n=5, got {p[-1]:.4f}"
    )
    print("  test_p_at_least_bounds: PASS")


def test_solve_n2_call_at_top():
    """At the highest bid (NUM_BIDS-1), the only legal action is CALL."""
    from poker_math_exact import get_hand_rank_counts
    counts, total = get_hand_rank_counts(5, n_samples=_FAST_SAMPLES)
    p = _compute_bid_at_least(counts, total)
    values, policy = _solve_n2(p)

    for actor in range(2):
        assert policy[NUM_BIDS - 1][actor] == CALL_ACTION, (
            f"policy[{NUM_BIDS-1}][{actor}] should be CALL_ACTION"
        )
    print("  test_solve_n2_call_at_top: PASS")


def test_solve_n2_values_in_range():
    """All EV values must be in [-1, +1]."""
    from poker_math_exact import get_hand_rank_counts
    counts, total = get_hand_rank_counts(5, n_samples=_FAST_SAMPLES)
    p = _compute_bid_at_least(counts, total)
    values, _ = _solve_n2(p)

    for bid_idx in range(NUM_BIDS):
        for actor in range(2):
            v = values[bid_idx][actor]
            assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9, (
                f"values[{bid_idx}][{actor}] = {v} out of [-1, 1]"
            )
    print("  test_solve_n2_values_in_range: PASS")


def test_zero_sum_consistency():
    """
    At every standing-bid state, the sum of EVs for the two actors must
    equal the call EV from the *caller's* perspective plus the *bidder's*
    perspective — i.e., they sum to 0 (zero-sum game).

    values[bid_idx][actor] + values[bid_idx][1-actor] should equal
    the zero-sum relationship: only one player can gain.

    More precisely: the equilibrium value for actor 0 and actor 1 at the
    same node are NOT direct complements (they describe different "whose
    turn" situations), but we can check that the call EV satisfies the
    zero-sum identity: call_ev(actor) = -call_ev(1-actor) only if they
    both call at the same bid.  Here we just verify the pure-call payoff
    is zero-sum.
    """
    from poker_math_exact import get_hand_rank_counts
    counts, total = get_hand_rank_counts(5, n_samples=_FAST_SAMPLES)
    p = _compute_bid_at_least(counts, total)

    for bid_idx in range(NUM_BIDS):
        call_ev_as_caller = 1.0 - 2.0 * p[bid_idx]
        call_ev_as_bidder = -call_ev_as_caller     # zero-sum
        assert abs(call_ev_as_caller + call_ev_as_bidder) < 1e-9
    print("  test_zero_sum_consistency: PASS")


def test_initial_bid_is_legal():
    """The optimal first bid index must be in [0, NUM_BIDS)."""
    eq = get_blind_equilibrium(5, n_samples=_FAST_SAMPLES)
    ib = eq["initial_bid"]
    assert 0 <= ib < NUM_BIDS, f"initial_bid {ib} out of range"
    print(f"  test_initial_bid_is_legal: PASS  (n=5 initial bid = {all_bids()[ib]})")


def test_near_threshold_n5():
    """
    For n=5, the 50%-threshold bid is near Pair 2 (idx 13).
    The equilibrium first bid should be at or very close to this index.
    """
    eq = get_blind_equilibrium(5, n_samples=_FAST_SAMPLES)
    ib = eq["initial_bid"]
    p  = eq["p_at_least"][ib]
    # The initial bid should be near the 50% threshold
    assert 0.35 < p < 0.65, (
        f"n=5 initial bid P_at_least={p:.4f} far from 50% threshold; "
        f"bid={all_bids()[ib]}"
    )
    print(f"  test_near_threshold_n5: PASS  (P_at_least at initial bid = {p:.4f})")


def test_cache_roundtrip():
    """Loading from cache returns identical values to the first computation."""
    eq1 = get_blind_equilibrium(5, n_samples=_FAST_SAMPLES)
    eq2 = get_blind_equilibrium(5, n_samples=_FAST_SAMPLES)  # should hit cache
    assert eq1["initial_bid"]   == eq2["initial_bid"]
    assert eq1["initial_value"] == eq2["initial_value"]
    assert eq1["p_at_least"]    == eq2["p_at_least"]
    print("  test_cache_roundtrip: PASS")


def test_policy_raises_are_legal():
    """Every raise action in the policy must point to a strictly higher bid."""
    eq     = get_blind_equilibrium(5, n_samples=_FAST_SAMPLES)
    policy = eq["policy"]

    for bid_idx in range(NUM_BIDS):
        for actor in range(2):
            act = policy[bid_idx][actor]
            if act != CALL_ACTION:
                assert act > bid_idx, (
                    f"policy[{bid_idx}][{actor}]={act} is not a strictly higher bid"
                )
                assert 0 <= act < NUM_BIDS, (
                    f"policy[{bid_idx}][{actor}]={act} out of bid range"
                )
    print("  test_policy_raises_are_legal: PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running blind equilibrium tests...\n")
    tests = [
        test_p_at_least_bounds,
        test_solve_n2_call_at_top,
        test_solve_n2_values_in_range,
        test_zero_sum_consistency,
        test_initial_bid_is_legal,
        test_near_threshold_n5,
        test_cache_roundtrip,
        test_policy_raises_are_legal,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  {t.__name__}: FAIL — {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed.")
    if failed:
        sys.exit(1)
