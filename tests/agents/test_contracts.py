"""AR-0a tests for the shared types module."""

from __future__ import annotations

import numpy as np
import pytest

from agents.contracts import (
    AgentDecision,
    BidDistribution,
    CallDecision,
    HandBelief,
    HandModel,
    IdentityHandModel,
    Infostate,
)
from agents.registry import build_agent
from game.bids import CALL_ACTION, HH_ACTION, NUM_ACTIONS, NUM_BIDS
from game.engine import new_match
from game.feasibility import feasible_action_mask, feasible_bid_mask


# ---------------------------------------------------------------------------
# Infostate
# ---------------------------------------------------------------------------

def _drive_one_turn(seed: int = 0, *, exact: bool = True, hh: bool = True):
    """Step a fresh match to its first decision and return the state."""
    state = new_match(num_players=2, seed=seed,
                      exact_rules=exact, high_hand=hh)
    state.start_next_round()
    return state


def test_infostate_round_trip_via_json():
    """1000 random decision points round-trip through JSON identically."""
    for seed in range(1000):
        state = _drive_one_turn(seed=seed)
        info  = Infostate.from_match_state(state)
        rt    = Infostate.from_json(info.to_json())
        assert info == rt


def test_infostate_feasible_mask_matches_legal_intersect_feasible():
    state = _drive_one_turn(seed=42)
    info  = Infostate.from_match_state(state)
    legal_set = set(state.legal_actions())
    fa_mask   = feasible_action_mask(info.pool_size)
    for a in range(NUM_ACTIONS):
        expect = (a in legal_set) and bool(fa_mask[a])
        assert bool(info.feasible_mask[a]) == expect, (a, expect)


def test_infostate_first_decision_no_standing_bid():
    state = _drive_one_turn(seed=7)
    info  = Infostate.from_match_state(state)
    assert info.standing_bid is None
    assert info.bid_history == ()
    assert CALL_ACTION not in info.legal_actions
    assert HH_ACTION not in info.legal_actions


# ---------------------------------------------------------------------------
# HandBelief invariants
# ---------------------------------------------------------------------------

def _uniform_belief(n: int = 10) -> HandBelief:
    mask = feasible_bid_mask(n)
    nf   = int(mask.sum())
    q    = np.zeros(NUM_BIDS, dtype=np.float32)
    q[mask] = 1.0 / nf
    logits = np.where(mask, 0.0, -np.inf).astype(np.float32)
    return HandBelief(q=q, q_logits=logits, feasible_mask=mask, n=n)


def test_handbelief_uniform_constructs_cleanly():
    b = _uniform_belief(10)
    assert abs(float(b.q.sum()) - 1.0) < 1e-5
    assert b.top_k(3) and len(b.top_k(3)) == 3


def test_handbelief_rejects_mass_on_infeasible():
    n = 2  # only HC + Pair feasible (≤25)
    mask = feasible_bid_mask(n)
    q = np.zeros(NUM_BIDS, dtype=np.float32)
    q[NUM_BIDS - 1] = 1.0  # Straight Flush A — infeasible at n=2
    logits = np.zeros(NUM_BIDS, dtype=np.float32)
    with pytest.raises(ValueError, match="infeasible"):
        HandBelief(q=q, q_logits=logits, feasible_mask=mask, n=n)


def test_handbelief_rejects_unnormalised_q():
    n = 10
    mask = feasible_bid_mask(n)
    q = np.zeros(NUM_BIDS, dtype=np.float32)
    q[0] = 0.5  # sums to 0.5
    logits = np.zeros(NUM_BIDS, dtype=np.float32)
    with pytest.raises(ValueError, match="sum to 1"):
        HandBelief(q=q, q_logits=logits, feasible_mask=mask, n=n)


def test_handbelief_rejects_wrong_dtype():
    n = 10
    mask = feasible_bid_mask(n)
    q = np.zeros(NUM_BIDS, dtype=np.float64)
    q[mask] = 1.0 / mask.sum()
    logits = np.zeros(NUM_BIDS, dtype=np.float32)
    with pytest.raises(TypeError, match="float32"):
        HandBelief(q=q, q_logits=logits, feasible_mask=mask, n=n)


# ---------------------------------------------------------------------------
# BidDistribution
# ---------------------------------------------------------------------------

def test_bid_distribution_empty_construction():
    legal_mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    feas_mask  = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    bd = BidDistribution.empty(legal_mask=legal_mask, feasible_mask=feas_mask)
    assert bd.support_size == 0
    assert bd.entropy == 0.0
    assert float(bd.pi.sum()) == 0.0


def test_bid_distribution_rejects_mass_outside_legal():
    pi    = np.zeros(NUM_ACTIONS, dtype=np.float32)
    pi[5] = 1.0
    logits = np.zeros(NUM_ACTIONS, dtype=np.float32)
    legal_mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    legal_mask[10] = True   # 5 is NOT legal, so pi has illegal mass
    feas_mask  = np.ones(NUM_ACTIONS, dtype=np.bool_)
    with pytest.raises(ValueError, match="mass outside"):
        BidDistribution(pi=pi, pi_logits=logits, legal_mask=legal_mask,
                        feasible_mask=feas_mask, entropy=0.0, support_size=1)


# ---------------------------------------------------------------------------
# CallDecision
# ---------------------------------------------------------------------------

def test_call_decision_clamps():
    CallDecision(p_call=0.0)
    CallDecision(p_call=1.0)
    with pytest.raises(ValueError):
        CallDecision(p_call=-0.1)
    with pytest.raises(ValueError):
        CallDecision(p_call=1.5)


# ---------------------------------------------------------------------------
# AgentDecision and trace round-trip
# ---------------------------------------------------------------------------

def test_agent_decision_invariant():
    AgentDecision(action_probs={5: 0.4, 7: 0.6})
    with pytest.raises(ValueError):
        AgentDecision(action_probs={5: 0.4, 7: 0.4})


def test_agent_decision_trace_json_minimal():
    ad = AgentDecision(action_probs={CALL_ACTION: 1.0})
    trace = ad.trace_json()
    assert trace == {"hh_fired": False}


def test_agent_decision_trace_with_belief():
    b = _uniform_belief(10)
    ad = AgentDecision(action_probs={5: 1.0}, belief=b)
    trace = ad.trace_json()
    assert "belief" in trace
    assert trace["belief"]["n"] == 10
    assert isinstance(trace["belief"]["q_top5"], list)


# ---------------------------------------------------------------------------
# IdentityHandModel and Protocol runtime check
# ---------------------------------------------------------------------------

def test_identity_handmodel_satisfies_protocol():
    m = IdentityHandModel()
    assert isinstance(m, HandModel)


def test_identity_handmodel_belief_invariants():
    state = _drive_one_turn(seed=99)
    info  = Infostate.from_match_state(state)
    m     = IdentityHandModel()
    b     = m.belief(info)
    assert b.q.shape == (NUM_BIDS,)
    assert abs(float(b.q.sum()) - 1.0) < 1e-5
    assert float(b.q[~b.feasible_mask].sum()) < 1e-6


def test_identity_handmodel_belief_batch_matches_loop():
    states = [_drive_one_turn(seed=s) for s in range(8)]
    infos  = [Infostate.from_match_state(s) for s in states]
    m      = IdentityHandModel()
    serial = [m.belief(i).q.tolist() for i in infos]
    batch  = [b.q.tolist() for b in m.belief_batch(infos)]
    assert serial == batch


# ---------------------------------------------------------------------------
# End-to-end through agents.policy.decision()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("agent_key", ["random", "exactconditional", "cfr_nash_mb3"])
def test_policy_decision_wraps_legacy_agents(agent_key):
    """Legacy agents (no .decision()) get wrapped into a minimal AgentDecision."""
    from agents.policy import decision

    agent = build_agent(agent_key)
    state = _drive_one_turn(seed=11, exact=True, hh=True)
    ad = decision(agent, state)
    assert isinstance(ad, AgentDecision)
    assert sum(ad.action_probs.values()) == pytest.approx(1.0, abs=1e-5)
    # Legacy agents leave trace fields empty.
    assert ad.belief is None
    assert ad.call is None
    assert ad.bid is None
    assert ad.hh_fired is False
