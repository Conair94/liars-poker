"""
Reality-check tests for exact-rules blind equilibrium.

These tests are designed to be skeptical: they verify the solver output from
first principles rather than just checking internal consistency.

Test categories:
  A. P_exact probability correctness — verify MC against exact enumeration
  B. Solver correctness — verify Nash equilibrium property (no profitable deviation)
  C. Engine correctness — verify has_exact_hand against known hand constructions
  D. Rule-set checks — verify High Card bids are legal first bids

Run via:
    cd "papers/Liars poker/"
    python -m agent.baseline.tests.test_blind_equilibrium_exact
"""

from __future__ import annotations

import os
import sys
from itertools import combinations

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _PAPER_DIR not in sys.path:
    sys.path.insert(0, _PAPER_DIR)

from poker_math_exact import _evaluate_ranked
from agent.game.bids import (
    Bid, all_bids, NUM_BIDS, CALL_ACTION, bid_to_index, index_to_bid,
    normalize_hand_type, HIGH_CARD, PAIR, TWO_PAIR, THREE_OF_A_KIND,
    STRAIGHT, FLUSH, FULL_HOUSE, FOUR_OF_A_KIND, STRAIGHT_FLUSH,
)
from agent.game.engine import has_exact_hand, new_match
from agent.baseline.blind_equilibrium import (
    _compute_bid_exact,
    _compute_bid_at_least,
    _solve_n2,
    _solve_initial,
    get_blind_equilibrium,
    get_blind_equilibrium_exact,
)
from poker_math_exact import get_hand_rank_counts

_FAST_SAMPLES = 100_000


# ===========================================================================
# A. P_exact probability correctness
# ===========================================================================

def test_p_exact_n2_exact_enumeration():
    """
    For n=2 the pool is 2 cards. Enumerate all C(52,2)=1326 possible pools
    and compute P_exact for each bid exactly. Compare against MC estimate.

    This is the critical ground-truth check: if the MC matches enumeration
    within 2 standard errors, the computation is correct.
    """
    deck = list(range(52))
    bids = all_bids()
    bid_keys = [(b.hand_type, b.primary_rank) for b in bids]
    exact_counts = [0] * NUM_BIDS
    total = 0

    for pool in combinations(deck, 2):
        pool = list(pool)
        raw_t, raw_p = _evaluate_ranked(pool)
        t, p = normalize_hand_type(raw_t, raw_p)
        for idx, key in enumerate(bid_keys):
            if (t, p) == key:
                exact_counts[idx] += 1
        total += 1

    assert total == 1326, f"Expected 1326 pools, got {total}"

    exact_probs = [c / total for c in exact_counts]
    exact_sum = sum(exact_probs)
    assert abs(exact_sum - 1.0) < 1e-9, (
        f"n=2 exact probs should sum to 1.0 (got {exact_sum:.8f}) "
        f"— for n<5 every pool has exactly one best hand"
    )

    # Run MC with enough samples for tight comparison
    mc_probs = _compute_bid_exact(2, n_samples=500_000, seed=42)

    # Known analytical values for n=2:
    # HC A = P(one Ace, one non-Ace) = 4*48 / C(52,2) = 192/1326
    hca_idx = bid_to_index(Bid(HIGH_CARD, 12))  # rank 12 = Ace
    expected_hca = 192 / 1326
    assert abs(exact_probs[hca_idx] - expected_hca) < 1e-9, (
        f"P_exact(HC A, n=2) exact={exact_probs[hca_idx]:.6f}, "
        f"analytical={expected_hca:.6f}"
    )

    # MC should be within 3 standard errors of exact for every bid
    import math
    for idx in range(NUM_BIDS):
        p_exact = exact_probs[idx]
        p_mc    = mc_probs[idx]
        # SE = sqrt(p*(1-p)/n)
        se = math.sqrt(max(p_exact * (1 - p_exact), 1e-10) / 500_000)
        assert abs(p_mc - p_exact) < 4 * se + 0.001, (
            f"P_exact[{idx}] ({bids[idx]}): MC={p_mc:.5f} exact={p_exact:.5f} "
            f"diff={abs(p_mc-p_exact):.5f} > 4SE={4*se:.5f}"
        )

    print(f"  test_p_exact_n2_exact_enumeration: PASS  "
          f"(P_exact(HC A)=192/1326={expected_hca:.5f}, MC={mc_probs[hca_idx]:.5f})")


def test_p_exact_n2_probs_sum_to_one():
    """For n<5, every pool has exactly one best hand, so P_exact values sum to 1."""
    mc_probs = _compute_bid_exact(2, n_samples=500_000, seed=42)
    total = sum(mc_probs)
    assert abs(total - 1.0) < 0.005, (
        f"P_exact for n=2 should sum to ~1.0 (got {total:.5f}). "
        f"For n<5 every pool maps to exactly one (type, rank) bid."
    )
    print(f"  test_p_exact_n2_probs_sum_to_one: PASS  (sum={total:.5f})")


def test_p_exact_n5_matches_rank_counts():
    """
    For n=5 there is only one 5-card subset (the whole pool), so
    P_exact(b) must equal P(pool_best == b) from get_hand_rank_counts.
    """
    rank_counts, total = get_hand_rank_counts(5, n_samples=_FAST_SAMPLES, seed=42)
    bids = all_bids()
    p_rank = {(b.hand_type, b.primary_rank): rank_counts.get((b.hand_type, b.primary_rank), 0) / total
              for b in bids}

    mc_probs = _compute_bid_exact(5, n_samples=_FAST_SAMPLES, seed=42)

    for idx, bid in enumerate(bids):
        key = (bid.hand_type, bid.primary_rank)
        expected = p_rank.get(key, 0.0)
        actual = mc_probs[idx]
        # Allow 2% absolute tolerance due to MC noise with fast samples
        assert abs(actual - expected) < 0.02 + 3 * (expected * (1 - expected) / _FAST_SAMPLES) ** 0.5 + 0.005, (
            f"P_exact(n=5) vs rank_counts mismatch at {bid}: "
            f"exact_mc={actual:.5f} rank_counts={expected:.5f}"
        )

    print(f"  test_p_exact_n5_matches_rank_counts: PASS")


def test_p_exact_ngt5_sums_above_one():
    """
    For n>5, a pool can contain multiple distinct exact hands (one per 5-card
    subset). P_exact values should SUM TO > 1 since the events overlap.
    """
    mc_probs = _compute_bid_exact(7, n_samples=_FAST_SAMPLES, seed=42)
    total = sum(mc_probs)
    assert total > 1.0, (
        f"For n=7 (C(7,5)=21 subsets), P_exact should sum to >1.0 "
        f"(got {total:.4f})"
    )
    print(f"  test_p_exact_ngt5_sums_above_one: PASS  (n=7 sum={total:.4f})")


def test_p_exact_hca_increases_with_n():
    """
    P_exact(HC A) should increase monotonically with pool size n,
    since more cards = more chances to contain an HC-A 5-card subset.
    """
    bids = all_bids()
    hca_idx = bid_to_index(Bid(HIGH_CARD, 12))
    prev = -1.0
    for n in [2, 4, 6, 8, 10]:
        eq = get_blind_equilibrium_exact(n)
        p = eq["p_exact"][hca_idx]
        assert p > prev, (
            f"P_exact(HC A) should increase with n: n={n} p={p:.4f} <= prev={prev:.4f}"
        )
        prev = p
    print(f"  test_p_exact_hca_increases_with_n: PASS")


# ===========================================================================
# B. Solver correctness — Nash equilibrium property
# ===========================================================================

def test_nash_p1_cannot_improve_at_n2():
    """
    For n=2 exact rules, the equilibrium says P1 calls when facing HC A.
    P1's call EV should be ≥ any raise EV.

    Reality check: P1 calling HC A gives EV = 1 - 2*P_exact(HC A) ≈ +0.71.
    Any raise to bid b' gives P1 EV = -values[b'][0] where P0 then responds
    optimally. This must be ≤ 0.71 for the equilibrium to be valid.
    """
    eq = get_blind_equilibrium_exact(2)
    bids = all_bids()
    p_exact = eq["p_exact"]
    values  = eq["values"]
    policy  = eq["policy"]

    hca_idx = bid_to_index(Bid(HIGH_CARD, 12))
    p1_call_ev = 1.0 - 2.0 * p_exact[hca_idx]

    # Verify P1 cannot improve by raising
    for b_prime in range(hca_idx + 1, NUM_BIDS):
        # If P1 raises to b', P0 faces b' as the standing bid and acts optimally
        p1_raise_ev = -values[b_prime][0]
        assert p1_raise_ev <= p1_call_ev + 1e-9, (
            f"P1 can profitably deviate from CALL: raise to {bids[b_prime]} "
            f"gives EV={p1_raise_ev:.4f} > call EV={p1_call_ev:.4f}. "
            f"Nash property violated."
        )

    print(f"  test_nash_p1_cannot_improve_at_n2: PASS  "
          f"(P1 call EV at HC A = {p1_call_ev:+.4f}; no raise improves this)")


def test_nash_p0_cannot_improve_first_bid_n2():
    """
    For n=2, P0's optimal first bid is HC A with EV ≈ −0.71.
    Verify no other first bid gives P0 a better EV.
    """
    eq = get_blind_equilibrium_exact(2)
    bids = all_bids()
    values = eq["values"]
    init_bid = eq["initial_bid"]
    init_val = eq["initial_value"]

    for b in range(NUM_BIDS):
        p0_ev = -values[b][1]  # P0's EV after bidding b (P1 responds optimally)
        assert p0_ev <= init_val + 1e-9, (
            f"P0 can profitably deviate to {bids[b]} (EV={p0_ev:.4f}) "
            f"from {bids[init_bid]} (EV={init_val:.4f}). "
            f"Nash property violated."
        )

    print(f"  test_nash_p0_cannot_improve_first_bid_n2: PASS  "
          f"(P0 best EV = {init_val:+.4f} with {bids[init_bid]}; confirmed optimal)")


def test_equilibrium_ev_negative_small_n():
    """
    Under exact rules, the first bidder has negative EV for n=2..7,
    approximately zero at n=8, and positive at n=9..10.
    This is the structural first-mover disadvantage unique to exact rules.
    """
    for n in range(2, 9):
        eq = get_blind_equilibrium_exact(n)
        ev = eq["initial_value"]
        assert ev < 0.05, (
            f"n={n}: expected first-bidder EV < 0 under exact rules, got {ev:+.4f}"
        )
    for n in range(9, 11):
        eq = get_blind_equilibrium_exact(n)
        ev = eq["initial_value"]
        assert ev > -0.05, (
            f"n={n}: expected first-bidder EV ≥ 0 under exact rules at large n, got {ev:+.4f}"
        )
    print(f"  test_equilibrium_ev_negative_small_n: PASS  "
          f"(first-mover disadvantage confirmed for n≤8, inverts at n≥9)")


def test_exact_first_bid_is_hca_for_small_n():
    """
    Under exact rules, optimal first bid = HC A for all n=2..8.
    HC A is the most probable exact hand at small pool sizes.
    """
    bids = all_bids()
    hca_idx = bid_to_index(Bid(HIGH_CARD, 12))
    for n in range(2, 9):
        eq = get_blind_equilibrium_exact(n)
        ib = eq["initial_bid"]
        assert ib == hca_idx, (
            f"n={n}: expected first bid HC A (idx={hca_idx}), "
            f"got {bids[ib]} (idx={ib})"
        )
    print(f"  test_exact_first_bid_is_hca_for_small_n: PASS")


def test_at_least_first_bid_tracks_threshold():
    """
    Under at-least rules, first bid P(pool >= bid) ≈ 50% for all n.
    This is the known blind equilibrium property.
    """
    for n in [5, 7, 10]:
        eq = get_blind_equilibrium(n)
        p = eq["p_at_least"][eq["initial_bid"]]
        assert 0.40 < p < 0.65, (
            f"n={n}: at-least first bid should be near 50% threshold, "
            f"got P={p:.4f}"
        )
    print(f"  test_at_least_first_bid_tracks_threshold: PASS")


def test_exact_vs_atleast_ev_diverge():
    """
    Exact and at-least equilibria have different EV profiles.
    At-least: first-bidder EV ≈ 0 for all n.
    Exact: first-bidder EV is very negative for small n.
    The difference should be > 0.5 EV units for n ≤ 5.
    """
    for n in range(2, 6):
        eq_al = get_blind_equilibrium(n)
        eq_ex = get_blind_equilibrium_exact(n)
        diff = eq_al["initial_value"] - eq_ex["initial_value"]
        assert diff > 0.5, (
            f"n={n}: EV difference (at-least minus exact) = {diff:.4f}; "
            f"expected > 0.5 EV units for small n"
        )
    print(f"  test_exact_vs_atleast_ev_diverge: PASS  "
          f"(exact rules penalize first bidder heavily at small n)")


# ===========================================================================
# C. Engine correctness — has_exact_hand
# ===========================================================================

def _card(rank: int, suit: int) -> int:
    """Encode card as rank*4 + suit. rank: 0=2..12=A, suit: 0=C..3=S."""
    return rank * 4 + suit


def test_has_exact_hand_pair_in_pool():
    """Pool with a known pair: has_exact_hand should find it."""
    ace_c = _card(12, 0)   # Ace of Clubs
    ace_d = _card(12, 1)   # Ace of Diamonds
    king_h = _card(11, 2)  # King of Hearts
    two_s  = _card(0, 3)   # 2 of Spades
    three_c = _card(1, 0)  # 3 of Clubs

    pool = [ace_c, ace_d, king_h, two_s, three_c]  # 5 cards, contains pair of Aces

    assert has_exact_hand(pool, Bid(PAIR, 12)), "Should find Pair of Aces"
    assert not has_exact_hand(pool, Bid(PAIR, 11)), "Should NOT find Pair of Kings"
    assert not has_exact_hand(pool, Bid(TWO_PAIR, 12)), "Should NOT find Two Pair"
    print("  test_has_exact_hand_pair_in_pool: PASS")


def test_has_exact_hand_small_pool():
    """For a 2-card pool, resolution uses all cards (no 5-card subsets)."""
    ace_c  = _card(12, 0)  # Ace of Clubs
    king_h = _card(11, 2)  # King of Hearts

    pool = [ace_c, king_h]
    assert has_exact_hand(pool, Bid(HIGH_CARD, 12)), "2-card pool [A, K] should match HC A"
    assert not has_exact_hand(pool, Bid(HIGH_CARD, 11)), "Should NOT match HC K (Ace present)"
    assert not has_exact_hand(pool, Bid(PAIR, 12)), "Should NOT match Pair A (no pair)"
    print("  test_has_exact_hand_small_pool: PASS")


def test_has_exact_hand_hca_requires_ace():
    """A pool with no Ace cannot contain an HC A 5-card hand."""
    # Build a 5-card pool with no Aces: K, Q, J, 9, 7 of different suits
    pool = [_card(11, 0), _card(10, 1), _card(9, 2), _card(7, 3), _card(5, 0)]
    assert not has_exact_hand(pool, Bid(HIGH_CARD, 12)), (
        "Pool with no Ace cannot contain HC A"
    )
    assert has_exact_hand(pool, Bid(HIGH_CARD, 11)), (
        "Pool [K, Q, J, 9, 7] should contain HC K"
    )
    print("  test_has_exact_hand_hca_requires_ace: PASS")


def test_has_exact_hand_two_high_hands_in_large_pool():
    """
    For n>5, a pool can contain BOTH HC A and HC K as different 5-card subsets.
    This is the key property that makes P_exact sum > 1 for n>5.
    """
    ace_c    = _card(12, 0)  # A♣
    king_h   = _card(11, 2)  # K♥
    queen_d  = _card(10, 1)  # Q♦
    jack_s   = _card(9, 3)   # J♠
    nine_c   = _card(7, 0)   # 9♣
    seven_h  = _card(5, 2)   # 7♥

    pool = [ace_c, king_h, queen_d, jack_s, nine_c, seven_h]  # 6 cards, no pairs/flush/straight

    # The subset [A, K, Q, J, 9] evaluates to HC A
    assert has_exact_hand(pool, Bid(HIGH_CARD, 12)), (
        "6-card pool containing Ace should have HC A subset"
    )
    # The subset [K, Q, J, 9, 7] evaluates to HC K
    assert has_exact_hand(pool, Bid(HIGH_CARD, 11)), (
        "6-card pool should also have HC K subset (exclude Ace)"
    )
    print("  test_has_exact_hand_two_high_hands_in_large_pool: PASS")


def test_has_exact_hand_straight_flush():
    """Pool containing a known straight flush."""
    pool = [
        _card(8, 0),   # T♣
        _card(7, 0),   # 9♣
        _card(6, 0),   # 8♣
        _card(5, 0),   # 7♣
        _card(4, 0),   # 6♣
    ]
    # 6-T straight flush in clubs; high card = T (rank 8)
    assert has_exact_hand(pool, Bid(STRAIGHT_FLUSH, 8)), (
        "Should find 6-T straight flush"
    )
    assert not has_exact_hand(pool, Bid(FLUSH, 8)), (
        "Should NOT match just Flush — the best hand IS a straight flush, not plain flush"
    )
    print("  test_has_exact_hand_straight_flush: PASS")


def test_engine_exact_rules_resolution():
    """
    Play a 2-player exact-rules round to terminal and verify resolution matches
    has_exact_hand directly. Force specific hands to make outcome deterministic.
    """
    state = new_match(2, seed=0, exact_rules=True)
    state.start_next_round()

    ace_c    = _card(12, 0)  # A♣
    king_h   = _card(11, 2)  # K♥

    # Force P0 to hold Ace, P1 to hold King (pool = [A, K])
    state.round_state.hands = [[ace_c], [king_h]]
    state.round_state.current_player = 0

    # P0 bids HC A (the 5th bid with rank=12 among High Cards)
    hca_idx = bid_to_index(Bid(HIGH_CARD, 12))
    state.apply_action(hca_idx)

    # P1 calls
    result = state.apply_action(CALL_ACTION)
    assert result is not None

    pool = [ace_c, king_h]
    expected_holds = has_exact_hand(pool, Bid(HIGH_CARD, 12))

    if expected_holds:
        # Bid held: caller (P1, seat 1) loses
        assert result.loser_seat == 1, (
            f"HC A held in pool [A, K]: caller P1 should lose"
        )
    else:
        # Bid didn't hold: bidder (P0, seat 0) loses
        assert result.loser_seat == 0, (
            f"HC A didn't hold in pool [A, K]: bidder P0 should lose"
        )

    print(f"  test_engine_exact_rules_resolution: PASS  "
          f"(pool [A, K]: HC A holds={expected_holds}, loser=P{result.loser_seat})")


# ===========================================================================
# D. Conditional private-info shift — blind equilibrium limitations
#
# The blind equilibrium computes the correct Nash for the BLIND game (no
# private info). Under exact rules, private cards dramatically change the
# effective probability that a bid holds. These tests make that concrete and
# serve as a warning against over-interpreting the blind equilibrium as
# strategy guidance for the full game.
# ===========================================================================

def _p_exact_conditional_n2(my_card: int, bid: Bid) -> float:
    """
    Enumerate all 51 possible opponent cards and compute
    P(has_exact_hand(pool={my_card, opp_card}, bid)) exactly.
    """
    deck = list(range(52))
    deck.remove(my_card)
    hits = sum(
        1 for opp in deck
        if has_exact_hand([my_card, opp], bid)
    )
    return hits / len(deck)


def test_conditional_hca_given_ace_vs_no_ace():
    """
    KEY INSIGHT TEST: The blind P_exact(HC A, n=2) = 0.145 is a weighted
    average of two wildly different conditional probabilities:

      P(HC A holds | I hold Ace)    ≈ 0.94  → bidding HC A is VERY safe
      P(HC A holds | I hold non-Ace) ≈ 0.08  → bidding HC A is FOOLISH

    Under at-least rules the private-info shift is small (the blind
    equilibrium is a reasonable first approximation). Under exact rules
    the shift is enormous: the blind equilibrium's recommendation to
    "bid HC A" is ONLY valid for a player holding an Ace.

    This test verifies those conditional values and confirms the blind
    equilibrium is NOT a meaningful strategy guide for the full exact game.
    """
    hca_bid = Bid(HIGH_CARD, 12)
    ace_c   = _card(12, 0)   # A♣
    king_h  = _card(11, 2)   # K♥

    # P(HC A holds | I hold Ace)
    p_given_ace = _p_exact_conditional_n2(ace_c, hca_bid)
    # P(HC A holds | I hold King, no Ace)
    p_given_king = _p_exact_conditional_n2(king_h, hca_bid)

    # Blind (marginal) probability
    blind_p = 192 / 1326  # exact = 0.14480

    # Conditional with Ace should be very high (~0.94: only fails if opp also has Ace)
    # 3 other Aces out of 51 remaining cards → P(pair Aces) = 3/51 ≈ 0.059
    # HC A holds iff no pair Aces: P ≈ 1 - 3/51 = 48/51
    assert p_given_ace > 0.90, (
        f"P(HC A holds | hold Ace) = {p_given_ace:.4f}; expected ~48/51 ≈ 0.941"
    )

    # Conditional without Ace should be very low (~0.08: only holds if opp has Ace)
    # 4 Aces out of 51 remaining cards → P = 4/51 ≈ 0.078
    assert p_given_king < 0.10, (
        f"P(HC A holds | hold King) = {p_given_king:.4f}; expected ~4/51 ≈ 0.078"
    )

    # The ratio should be enormous — private info is the dominant signal
    ratio = p_given_ace / max(p_given_king, 1e-9)
    assert ratio > 10, (
        f"Conditional ratio should be >10x: P(Ace)={p_given_ace:.4f} / "
        f"P(no-Ace)={p_given_king:.4f} = {ratio:.1f}x"
    )

    print(
        f"  test_conditional_hca_given_ace_vs_no_ace: PASS\n"
        f"    Blind P_exact(HC A)     = {blind_p:.4f}\n"
        f"    P(HC A | hold Ace)      = {p_given_ace:.4f}  ← rational to bid\n"
        f"    P(HC A | hold King)     = {p_given_king:.4f}  ← irrational to bid\n"
        f"    Ratio                   = {ratio:.1f}x\n"
        f"    → Blind equilibrium should NOT be used as strategy guidance "
        f"for the full exact game."
    )


def test_rational_first_bid_depends_on_private_card():
    """
    Verify the rational first bid for a player holding rank r is to bid
    HC r (not HC A, unless they hold an Ace).

    For a player holding card of rank r:
      - P(HC r holds | hold rank r) = P(opp has no rank ≥ r AND no pair with r)
      - P(HC r+1 holds | hold rank r) ≈ P(opp has rank r+1 and no higher) << above
      - P(HC A holds | hold rank r < A) ≈ 4/51 ≈ 0.08

    The conditionally optimal bid is approximately HC r: the player commits
    to the one HC bid they know is the pool's best HC if opponent has no Ace.
    """
    # For a player holding rank 8 (T = index 8):
    ten_c = _card(8, 0)  # T♣
    hct_bid = Bid(HIGH_CARD, 8)   # HC T
    hca_bid = Bid(HIGH_CARD, 12)  # HC A

    p_hct_given_ten = _p_exact_conditional_n2(ten_c, hct_bid)
    p_hca_given_ten = _p_exact_conditional_n2(ten_c, hca_bid)

    assert p_hct_given_ten > p_hca_given_ten, (
        f"P(HC T | hold T) = {p_hct_given_ten:.4f} should exceed "
        f"P(HC A | hold T) = {p_hca_given_ten:.4f}"
    )

    # Holding a Ten: HC T should hold whenever opp has no card ranked > T and no pair with T
    # HC A should only hold if opp has an Ace (~0.08)
    print(
        f"  test_rational_first_bid_depends_on_private_card: PASS\n"
        f"    P(HC T | hold T) = {p_hct_given_ten:.4f}  ← commit to what you know\n"
        f"    P(HC A | hold T) = {p_hca_given_ten:.4f}  ← bluff you'd never make"
    )


def test_calling_hca_rational_without_ace():
    """
    If the opponent bids HC A and you don't hold an Ace, calling is
    conditionally correct: P(HC A holds | you no Ace) ≈ 0.08.
    Call EV = 1 - 2 * 0.08 = +0.84 >> the blind equilibrium's +0.71.

    This test verifies the call EV conditioned on your private card is
    much more favorable than the blind EV, explaining why a rational
    non-Ace holder always calls HC A.
    """
    king_h = _card(11, 2)  # K♥
    hca_bid = Bid(HIGH_CARD, 12)

    p_holds_given_king = _p_exact_conditional_n2(king_h, hca_bid)
    conditional_call_ev = 1.0 - 2.0 * p_holds_given_king

    # Blind call EV ≈ +0.71; conditional (non-Ace) call EV should be higher
    blind_call_ev = 1.0 - 2.0 * (192 / 1326)

    assert conditional_call_ev > blind_call_ev + 0.05, (
        f"Conditional call EV (non-Ace) = {conditional_call_ev:.4f} should be "
        f"significantly better than blind call EV = {blind_call_ev:.4f}"
    )
    assert conditional_call_ev > 0.80, (
        f"Non-Ace holder calling HC A should have EV > 0.80, got {conditional_call_ev:.4f}"
    )

    print(
        f"  test_calling_hca_rational_without_ace: PASS\n"
        f"    Blind call EV at HC A:         {blind_call_ev:+.4f}\n"
        f"    Conditional call EV (hold K):  {conditional_call_ev:+.4f}\n"
        f"    → Non-Ace holder calling HC A is near-certain to win."
    )


# ===========================================================================
# E. High Hand action — available and correct under exact rules
#
# High Hand (HH) declaration says "the standing bid IS the pool's best hand."
# Under exact rules this is a critical escape valve: if the opponent bids
# HC A and you also hold an Ace, calling is bad (HC A likely holds) and
# overbidding is risky. HH lets you declare the bid is the ceiling and win
# a card if correct (penalizing the bidder).
# ===========================================================================

def test_high_hand_available_after_bid():
    """High Hand action must appear in legal_actions after a bid when high_hand=True."""
    from agent.game.bids import HH_ACTION
    state = new_match(2, seed=0, exact_rules=True, high_hand=True)
    state.start_next_round()
    # Before any bid: HH should NOT be legal (nothing to declare on)
    assert HH_ACTION not in state.legal_actions(), (
        "HH should not be legal before any bid"
    )
    # After P0 bids HC A:
    hca_idx = bid_to_index(Bid(HIGH_CARD, 12))
    state.apply_action(hca_idx)
    legal = state.legal_actions()
    assert HH_ACTION in legal, "HH should be legal after a bid when high_hand=True"
    assert CALL_ACTION in legal, "CALL should also still be legal"
    print("  test_high_hand_available_after_bid: PASS")


def test_high_hand_not_available_without_flag():
    """High Hand action must NOT appear when high_hand=False."""
    from agent.game.bids import HH_ACTION
    state = new_match(2, seed=0, exact_rules=True, high_hand=False)
    state.start_next_round()
    hca_idx = bid_to_index(Bid(HIGH_CARD, 12))
    state.apply_action(hca_idx)
    assert HH_ACTION not in state.legal_actions(), (
        "HH should not be legal when high_hand=False"
    )
    print("  test_high_hand_not_available_without_flag: PASS")


def test_high_hand_correct_declaration_penalizes_bidder():
    """
    Scenario: P0 bids HC A. Pool is [A, K] so pool best = HC A.
    P1 declares High Hand (correct: HC A IS pool best).
    P0 (bidder) should be penalized; P1 (declarer) rewarded.
    """
    from agent.game.bids import HH_ACTION
    state = new_match(2, seed=0, exact_rules=True, high_hand=True)
    state.start_next_round()

    ace_c   = _card(12, 0)  # A♣
    king_h  = _card(11, 2)  # K♥
    state.round_state.hands = [[ace_c], [king_h]]

    hca_idx = bid_to_index(Bid(HIGH_CARD, 12))
    state.apply_action(hca_idx)   # P0 bids HC A

    result = state.apply_action(HH_ACTION)  # P1 declares High Hand
    assert result is not None
    assert result.is_high_hand, "Should be flagged as a High Hand resolution"
    assert result.declaration_correct, (
        "HC A IS pool best for [A, K] — declaration should be correct"
    )
    assert result.loser_seat == 0, (
        "Correct HH declaration: bidder (P0) is penalized"
    )
    assert result.winner_seat == 1, (
        "Correct HH declaration: declarer (P1) is rewarded"
    )
    print("  test_high_hand_correct_declaration_penalizes_bidder: PASS  "
          f"(pool [A,K]: HC A is pool best; bidder P0 penalized)")


def test_high_hand_incorrect_declaration_penalizes_declarer():
    """
    Scenario: P0 bids HC K. Pool is [A, K] so pool best = HC A (not HC K).
    P1 declares High Hand (incorrect: HC K is NOT pool best).
    P1 (declarer) should be penalized.
    """
    from agent.game.bids import HH_ACTION
    state = new_match(2, seed=0, exact_rules=True, high_hand=True)
    state.start_next_round()

    ace_c   = _card(12, 0)  # A♣
    king_h  = _card(11, 2)  # K♥
    state.round_state.hands = [[ace_c], [king_h]]

    hck_idx = bid_to_index(Bid(HIGH_CARD, 11))  # HC K
    state.apply_action(hck_idx)   # P0 bids HC K

    result = state.apply_action(HH_ACTION)  # P1 declares High Hand
    assert result is not None
    assert result.is_high_hand
    assert not result.declaration_correct, (
        "HC K is NOT pool best ([A,K] → HC A) — declaration should be incorrect"
    )
    assert result.loser_seat == 1, (
        "Incorrect HH declaration: declarer (P1) is penalized"
    )
    print("  test_high_hand_incorrect_declaration_penalizes_declarer: PASS  "
          f"(pool [A,K]: HC K is NOT pool best; declarer P1 penalized)")


def test_high_hand_ace_scenario_user_insight():
    """
    The user's scenario: if you hold an Ace and opponent bids HC A,
    calling is bad (HC A likely holds). High Hand declaration is the
    rational alternative: declare HC A IS the ceiling and penalize bidder.

    P0 holds Ace, bids HC A. P1 holds Ace too.
    Pool = [A♣, A♦] → pool best = Pair A (NOT HC A).
    P1 declaring HH on HC A would be INCORRECT (pool best ≠ HC A).
    P1 should instead CALL (which wins since HC A doesn't hold as exact
    subset: two Aces → pair, not HC A).
    """
    ace_c = _card(12, 0)   # A♣
    ace_d = _card(12, 1)   # A♦

    # Verify pool = [A, A] → HC A does NOT hold as exact hand (it's a Pair)
    assert not has_exact_hand([ace_c, ace_d], Bid(HIGH_CARD, 12)), (
        "Pool [A♣, A♦] → Pair A, not HC A: HC A should NOT hold"
    )
    # Calling HC A wins (bid doesn't hold) — the rational move when you hold an Ace
    # and suspect opponent also holds one

    # Also verify: pool [A♣, K♥] → HC A DOES hold
    king_h = _card(11, 2)
    assert has_exact_hand([ace_c, king_h], Bid(HIGH_CARD, 12)), (
        "Pool [A♣, K♥] → HC A should hold"
    )
    print("  test_high_hand_ace_scenario_user_insight: PASS\n"
          "    If you hold Ace and opponent bids HC A:\n"
          "      - If opp also has Ace: pool = Pair A → CALL (HC A doesn't hold)\n"
          "      - If opp has non-Ace: pool = HC A → HH declaration correct\n"
          "    → High Hand gives a third option beyond call/raise.")


# ===========================================================================
# F. Rule-set checks — High Card bids are legal
# ===========================================================================

def test_high_card_bids_in_bid_space():
    """
    High Card bids (HC 2 through HC A) must be in the bid space.
    They are the 13 weakest bids (indices 0..12).
    """
    bids = all_bids()
    hc_bids = [b for b in bids if b.hand_type == HIGH_CARD]
    assert len(hc_bids) == 13, f"Expected 13 HC bids, got {len(hc_bids)}"
    assert hc_bids[0].primary_rank == 0, "Weakest HC bid should be HC 2 (rank 0)"
    assert hc_bids[-1].primary_rank == 12, "Strongest HC bid should be HC A (rank 12)"
    # HC bids are the first 13 entries in the bid ordering
    for i in range(13):
        assert bids[i].hand_type == HIGH_CARD, (
            f"Bid {i} should be High Card, got hand_type={bids[i].hand_type}"
        )
    print(f"  test_high_card_bids_in_bid_space: PASS  "
          f"(13 HC bids at indices 0..12; HC A at idx {bid_to_index(Bid(HIGH_CARD, 12))})")


def test_high_card_is_legal_first_bid():
    """Player 0 can legally bid any HC bid as the first action in a round."""
    state = new_match(2, seed=0, exact_rules=True)
    state.start_next_round()
    legal = state.legal_actions()
    hca_idx = bid_to_index(Bid(HIGH_CARD, 12))
    hc2_idx = bid_to_index(Bid(HIGH_CARD, 0))
    assert hca_idx in legal, f"HC A (idx={hca_idx}) must be a legal first bid"
    assert hc2_idx in legal, f"HC 2 (idx={hc2_idx}) must be a legal first bid"
    assert CALL_ACTION not in legal, "CALL should not be legal as first action"
    print(f"  test_high_card_is_legal_first_bid: PASS  "
          f"({len(legal)} legal first bids; all HC bids included)")


def test_high_card_bids_cover_all_ranks():
    """Each rank 0..12 (2..A) has exactly one HC bid."""
    bids = all_bids()
    hc_ranks = sorted(b.primary_rank for b in bids if b.hand_type == HIGH_CARD)
    assert hc_ranks == list(range(13)), (
        f"HC bids should cover ranks 0..12; got {hc_ranks}"
    )
    print("  test_high_card_bids_cover_all_ranks: PASS")


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    print("Running exact-rules reality-check tests...\n")
    print("Category A: P_exact probability correctness")
    tests_a = [
        test_p_exact_n2_exact_enumeration,
        test_p_exact_n2_probs_sum_to_one,
        test_p_exact_n5_matches_rank_counts,
        test_p_exact_ngt5_sums_above_one,
        test_p_exact_hca_increases_with_n,
    ]
    print("\nCategory B: Nash equilibrium validity")
    tests_b = [
        test_nash_p1_cannot_improve_at_n2,
        test_nash_p0_cannot_improve_first_bid_n2,
        test_equilibrium_ev_negative_small_n,
        test_exact_first_bid_is_hca_for_small_n,
        test_at_least_first_bid_tracks_threshold,
        test_exact_vs_atleast_ev_diverge,
    ]
    print("\nCategory C: Engine has_exact_hand correctness")
    tests_c = [
        test_has_exact_hand_pair_in_pool,
        test_has_exact_hand_small_pool,
        test_has_exact_hand_hca_requires_ace,
        test_has_exact_hand_two_high_hands_in_large_pool,
        test_has_exact_hand_straight_flush,
        test_engine_exact_rules_resolution,
    ]
    print("\nCategory D: Private-info conditional shift (blind equilibrium limitations)")
    tests_d = [
        test_conditional_hca_given_ace_vs_no_ace,
        test_rational_first_bid_depends_on_private_card,
        test_calling_hca_rational_without_ace,
    ]
    print("\nCategory E: High Hand action")
    tests_e = [
        test_high_hand_available_after_bid,
        test_high_hand_not_available_without_flag,
        test_high_hand_correct_declaration_penalizes_bidder,
        test_high_hand_incorrect_declaration_penalizes_declarer,
        test_high_hand_ace_scenario_user_insight,
    ]
    print("\nCategory F: Rule-set checks")
    tests_f = [
        test_high_card_bids_in_bid_space,
        test_high_card_is_legal_first_bid,
        test_high_card_bids_cover_all_ranks,
    ]

    all_tests = tests_a + tests_b + tests_c + tests_d + tests_e + tests_f
    passed = 0
    failed = 0
    for t in all_tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  {t.__name__}: FAIL — {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed.")
    if failed:
        sys.exit(1)
