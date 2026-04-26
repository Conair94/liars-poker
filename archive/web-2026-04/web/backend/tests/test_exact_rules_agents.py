"""
Unit tests for ExactRulesConditionalAgent's five game-theoretic fixes.

Tests each fix in isolation using hand-crafted MatchState objects:
    1. HH declaration when standing bid is near the peak of the adjusted dist
    2. Escalation-aware bidding — prefers smallest safe raise, not global peak
    3. Decision-theoretic call threshold — call when P(holds) < 0.5
    4. Opponent-bid Bayesian update — soft bump toward opp's primary rank
    5. Conditional-table adjustment (exact-rules table preferred; fallback OK)
"""

from __future__ import annotations

import os
import sys

_TESTS_DIR   = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.abspath(os.path.join(_TESTS_DIR, ".."))
_AGENT_DIR   = os.path.abspath(os.path.join(_BACKEND_DIR, "..", ".."))
_PAPER_DIR   = os.path.abspath(os.path.join(_AGENT_DIR, ".."))
for _p in (_PAPER_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest

from agent.game.engine import MatchState, RoundState
from agent.game.bids   import (
    Bid, CALL_ACTION, HH_ACTION, NUM_BIDS,
    bid_to_index, all_bids, HIGH_CARD, PAIR,
)
from agent.web.backend.agents import ExactRulesConditionalAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    hand_sizes,
    hands,
    current_bid=None,
    current_player=0,
    exact_rules=True,
    high_hand=True,
) -> MatchState:
    n_players = len(hand_sizes)
    rs = RoundState(
        hands=list(hands),
        history=[],
        current_player=current_player,
        last_bidder=(0 if current_bid is not None else -1),
        current_bid=current_bid,
    )
    state = MatchState(
        num_players=n_players,
        hand_sizes=list(hand_sizes),
        active=[True] * n_players,
        first_bidder_next=0,
        round_state=rs,
        round_history=[],
        terminal=False,
        winner=None,
        mode='countup',
        exact_rules=exact_rules,
        high_hand=high_hand,
    )
    return state


class FakeLookup:
    """Minimal stand-in to control the exact_prob / conditional vectors."""
    def __init__(self, exact_prob, features=None, exact_cond=None):
        self._exact_prob = np.asarray(exact_prob, dtype=np.float32)
        self._features   = features  # (marginal, cond, cond_key) or None
        self._exact_cond = exact_cond

    def get_exact_rules_exact(self, n):
        return self._exact_prob

    def get_features(self, own_hand, n):
        if self._features is None:
            uni = np.full(NUM_BIDS, 1.0 / NUM_BIDS, dtype=np.float32)
            return uni, uni, None
        return self._features

    def get_exact_rules_conditional(self, n, cond_key):
        if self._exact_cond is None:
            return None
        return self._exact_cond.get(cond_key)


# ---------------------------------------------------------------------------
# Fix 2.1 — HH declaration at the peak
# ---------------------------------------------------------------------------

def test_hh_declaration_when_standing_bid_equals_peak(monkeypatch):
    """Standing bid at the peak of adj_exact → agent declares HH."""
    import agent.web.backend.agents as agents_mod

    # Peak at HC A (bid index = 12 under the standard all_bids ordering).
    peak_bid = Bid(HIGH_CARD, 12)
    peak_idx = bid_to_index(peak_bid)
    exact_prob = np.full(NUM_BIDS, 0.05, dtype=np.float32)
    exact_prob[peak_idx] = 0.9  # dominant peak

    lookup = FakeLookup(exact_prob)
    monkeypatch.setattr(agents_mod, "_get_warm_start", lambda: lookup)

    agent = ExactRulesConditionalAgent()
    # n=10 pool, 5+5 hands, P0 responding to P1's HC A.
    state = _make_state([5, 5], [[0]*5, [0]*5], current_bid=peak_bid, current_player=0)
    action = agent.choose_action(state)
    assert action == HH_ACTION, f"expected HH_ACTION, got {action}"


def test_no_hh_when_standing_bid_far_below_peak(monkeypatch):
    """Standing bid well below peak → agent should not declare HH."""
    import agent.web.backend.agents as agents_mod

    peak_bid = Bid(HIGH_CARD, 12)           # HC A
    peak_idx = bid_to_index(peak_bid)
    low_bid  = Bid(HIGH_CARD, 0)            # HC 2
    exact_prob = np.full(NUM_BIDS, 0.05, dtype=np.float32)
    exact_prob[peak_idx] = 0.9
    exact_prob[bid_to_index(low_bid)] = 0.02

    lookup = FakeLookup(exact_prob)
    monkeypatch.setattr(agents_mod, "_get_warm_start", lambda: lookup)

    agent = ExactRulesConditionalAgent(opp_bid_alpha=0.0)
    state = _make_state([5, 5], [[0]*5, [0]*5], current_bid=low_bid, current_player=0)
    action = agent.choose_action(state)
    assert action != HH_ACTION, f"HH should not fire at low bid, got {action}"


# ---------------------------------------------------------------------------
# Fix 2.2 — Smallest safe raise rather than always bidding the peak
# ---------------------------------------------------------------------------

def test_opening_picks_smallest_safe_raise(monkeypatch):
    """When multiple bids clear safety_frac*peak, agent picks the smallest one
    (preserves bid space) instead of jumping to the global peak."""
    import agent.web.backend.agents as agents_mod

    exact_prob = np.full(NUM_BIDS, 0.01, dtype=np.float32)
    bid_low  = bid_to_index(Bid(HIGH_CARD, 2))   # HC 4
    bid_mid  = bid_to_index(Bid(HIGH_CARD, 8))   # HC 10
    bid_peak = bid_to_index(Bid(HIGH_CARD, 12))  # HC A
    exact_prob[bid_low]  = 0.7
    exact_prob[bid_mid]  = 0.8
    exact_prob[bid_peak] = 1.0

    lookup = FakeLookup(exact_prob)
    monkeypatch.setattr(agents_mod, "_get_warm_start", lambda: lookup)

    agent = ExactRulesConditionalAgent(safety_frac=0.5, opp_bid_alpha=0.0)
    state = _make_state([5, 5], [[0]*5, [0]*5], current_bid=None, current_player=0)
    action = agent.choose_action(state)
    # All three of bid_low, bid_mid, bid_peak clear 0.5 safety — smallest wins.
    assert action == bid_low, f"expected smallest safe raise {bid_low}, got {action}"


# ---------------------------------------------------------------------------
# Fix 2.3 — Decision-theoretic call threshold (P(holds) < 0.5 → call)
# ---------------------------------------------------------------------------

def test_calls_when_current_bid_probability_below_half(monkeypatch):
    """Standing bid with adj_exact < 0.5 should trigger CALL."""
    import agent.web.backend.agents as agents_mod

    cur_bid = Bid(HIGH_CARD, 6)
    cur_idx = bid_to_index(cur_bid)
    peak_idx = bid_to_index(Bid(HIGH_CARD, 0))  # keep peak far from cur_bid
    exact_prob = np.full(NUM_BIDS, 0.01, dtype=np.float32)
    exact_prob[peak_idx] = 0.6
    exact_prob[cur_idx]  = 0.2   # P(standing bid holds) = 20%

    lookup = FakeLookup(exact_prob)
    monkeypatch.setattr(agents_mod, "_get_warm_start", lambda: lookup)

    # Disable HH band and opp-bid bump so only the call threshold is exercised.
    agent = ExactRulesConditionalAgent(
        hh_band=2.0,              # never fires (requires >= 200% of peak)
        opp_bid_alpha=0.0,
        call_prob_threshold=0.5,
        floor_frac=0.0,
    )
    state = _make_state([5, 5], [[0]*5, [0]*5], current_bid=cur_bid, current_player=0)
    action = agent.choose_action(state)
    assert action == CALL_ACTION, f"expected CALL_ACTION, got {action}"


def test_no_call_when_current_bid_probability_is_high(monkeypatch):
    """Standing bid with adj_exact >= 0.5 and below peak should NOT call."""
    import agent.web.backend.agents as agents_mod

    cur_bid = Bid(HIGH_CARD, 4)
    cur_idx = bid_to_index(cur_bid)
    exact_prob = np.full(NUM_BIDS, 0.05, dtype=np.float32)
    exact_prob[cur_idx] = 0.7
    peak_idx = bid_to_index(Bid(HIGH_CARD, 12))
    exact_prob[peak_idx] = 0.8  # peak slightly above cur so HH won't fire

    lookup = FakeLookup(exact_prob)
    monkeypatch.setattr(agents_mod, "_get_warm_start", lambda: lookup)

    agent = ExactRulesConditionalAgent(hh_band=2.0, opp_bid_alpha=0.0)
    state = _make_state([5, 5], [[0]*5, [0]*5], current_bid=cur_bid, current_player=0)
    action = agent.choose_action(state)
    assert action != CALL_ACTION, f"should raise, not call; got {action}"


# ---------------------------------------------------------------------------
# Fix 2.4 — Opponent-bid Bayesian up-weighting
# ---------------------------------------------------------------------------

def test_opp_bid_belief_bumps_ranks_matching_opponent_bid():
    """_apply_opp_bid_belief multiplies entries sharing the opp's primary rank."""
    agent = ExactRulesConditionalAgent(
        opp_bid_alpha=1.0,        # full application
        opp_bid_up_mult=2.0,
        opp_bid_down_mult=1.0,
    )
    adj = np.full(NUM_BIDS, 1.0, dtype=np.float32)
    cur_bid = Bid(HIGH_CARD, 12)  # primary_rank = 12 (A)
    out = agent._apply_opp_bid_belief(adj, cur_bid)

    # Every bid with primary_rank == 12 should be doubled; others unchanged or
    # (for far-from-A ranks) slightly damped (down_mult=1.0 here, so unchanged).
    for i, bid in enumerate(all_bids()):
        if bid.primary_rank == 12:
            assert out[i] == pytest.approx(2.0), f"bid {i} ({bid}) should be 2× after bump"
        else:
            assert out[i] == pytest.approx(1.0)


def test_opp_bid_belief_noop_when_alpha_zero():
    """opp_bid_alpha=0 → adj_exact passes through unchanged."""
    agent = ExactRulesConditionalAgent(opp_bid_alpha=0.0)
    adj = np.linspace(0.0, 1.0, NUM_BIDS, dtype=np.float32)
    out = agent._apply_opp_bid_belief(adj, Bid(HIGH_CARD, 5))
    np.testing.assert_array_equal(out, adj)


# ---------------------------------------------------------------------------
# Fix 2.5 — Conditional-table adjustment prefers exact-rules table
# ---------------------------------------------------------------------------

def test_adjust_for_own_hand_uses_exact_rules_conditional_when_available(monkeypatch):
    """If get_exact_rules_conditional returns a vector, it is used directly."""
    import agent.web.backend.agents as agents_mod

    exact_prob = np.full(NUM_BIDS, 0.3, dtype=np.float32)
    # Exact-rules conditional: spike on HC A
    ec = np.full(NUM_BIDS, 0.0, dtype=np.float32)
    ec[bid_to_index(Bid(HIGH_CARD, 12))] = 0.95

    features = (
        np.full(NUM_BIDS, 0.5, dtype=np.float32),  # marginal
        np.full(NUM_BIDS, 0.5, dtype=np.float32),  # cond  (unused when exact_cond hits)
        "pair_A",
    )
    lookup = FakeLookup(exact_prob, features=features, exact_cond={"pair_A": ec})
    monkeypatch.setattr(agents_mod, "_get_warm_start", lambda: lookup)

    agent = ExactRulesConditionalAgent()
    # Own hand of two Aces → condition key "pair_A"
    # (FakeLookup returns features regardless of hand content)
    adj = agent._adjust_for_own_hand(exact_prob, lookup, own_hand=[48, 49], n=10)
    # Should equal the exact conditional table exactly.
    np.testing.assert_allclose(adj, ec, rtol=1e-6)


def test_adjust_for_own_hand_falls_back_when_exact_cond_absent(monkeypatch):
    """When the exact-rules conditional cache is missing, fall back to the
    likelihood-ratio correction using marginal/conditional at-least tables."""
    import agent.web.backend.agents as agents_mod

    exact_prob = np.full(NUM_BIDS, 0.3, dtype=np.float32)
    # Marginal flat; cond spikes on HC A → ratio up-weights HC A entry.
    marg = np.full(NUM_BIDS, 0.05, dtype=np.float32)
    cond = np.full(NUM_BIDS, 0.05, dtype=np.float32)
    a_idx = bid_to_index(Bid(HIGH_CARD, 12))
    cond[a_idx] = 0.5

    features = (marg, cond, "pair_A")
    lookup   = FakeLookup(exact_prob, features=features, exact_cond=None)
    monkeypatch.setattr(agents_mod, "_get_warm_start", lambda: lookup)

    agent = ExactRulesConditionalAgent()
    adj = agent._adjust_for_own_hand(exact_prob, lookup, own_hand=[48, 49], n=10)
    # At the spiked rank, adj should be strictly greater than at a flat-ratio rank.
    flat_idx = bid_to_index(Bid(HIGH_CARD, 0))
    assert adj[a_idx] > adj[flat_idx], (
        f"likelihood ratio should up-weight bid at spiked rank: "
        f"adj[A]={adj[a_idx]}, adj[2]={adj[flat_idx]}"
    )
