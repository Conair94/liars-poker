"""AR-2 Phase 5: smoke tests for ModularNashAgent wiring.

Heads are *untrained* — these tests pin the wiring (HH gate → CallPolicy →
BidPolicy with `hh_fired=False`), not the policy's quality.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.contracts import (
    BidDistribution,
    CallDecision,
    HandBelief,
    Infostate,
)
from agents.learned.bidpolicy import BidPolicyConfig, BidPolicyNet, DistilledBidPolicy
from agents.learned.callpolicy import CallPolicyConfig, CallPolicyNet, DistilledCallPolicy
from agents.learned.handmodel.config import HandModelConfig
from agents.learned.handmodel.network import LearnedHandModel, LearnedHandModelNet
from agents.learned.modular_nash_agent import ModularNashAgent
from game.bids import CALL_ACTION, HH_ACTION, NUM_ACTIONS, NUM_BIDS, bid_to_index
from game.engine import new_match
from game.feasibility import feasible_action_mask, feasible_bid_mask


# ---------------------------------------------------------------------------
# Stub components — wiring tests don't need a trained net.
# ---------------------------------------------------------------------------

class _PeakedHandModel:
    """Deterministic HandModel with all mass on a chosen bid index."""

    def __init__(self, peak_bid: int) -> None:
        self.peak_bid = peak_bid

    def belief(self, info: Infostate) -> HandBelief:
        mask = feasible_bid_mask(info.pool_size)
        q = np.zeros(NUM_BIDS, dtype=np.float32)
        peak = self.peak_bid if mask[self.peak_bid] else int(np.argmax(mask))
        q[peak] = 1.0
        logits = np.where(mask, 0.0, -np.inf).astype(np.float32)
        return HandBelief(q=q, q_logits=logits, feasible_mask=mask, n=info.pool_size)

    def belief_batch(self, infos):
        return [self.belief(i) for i in infos]


class _ConstCallPolicy:
    def __init__(self, p: float) -> None:
        self.p = p

    def call_prob(self, info, q) -> CallDecision:
        return CallDecision(p_call=self.p, inputs={})


class _UniformBidPolicy:
    """Uniform over feasible bids; respects `hh_fired` short-circuit."""

    def bid_dist(self, info: Infostate, q, *, hh_fired: bool) -> BidDistribution:
        legal_mask    = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        legal_mask[list(info.legal_actions)] = True
        feasible_mask = np.array(info.feasible_mask, dtype=np.bool_)
        if hh_fired:
            return BidDistribution.empty(legal_mask=legal_mask, feasible_mask=feasible_mask)

        joint = legal_mask & feasible_mask
        bid_support = joint.copy()
        bid_support[NUM_BIDS:] = False  # bids only

        pi     = np.zeros(NUM_ACTIONS, dtype=np.float32)
        logits = np.full(NUM_ACTIONS, -np.inf, dtype=np.float32)
        k = int(bid_support.sum())
        if k == 0:
            return BidDistribution.empty(legal_mask=legal_mask, feasible_mask=feasible_mask)
        pi[bid_support] = 1.0 / k
        logits[bid_support] = 0.0
        entropy = float(np.log(k)) if k > 1 else 0.0
        return BidDistribution(
            pi=pi, pi_logits=logits,
            legal_mask=legal_mask, feasible_mask=feasible_mask,
            entropy=entropy, support_size=k,
        )


def _start_state(seed: int = 0, *, n_cards: int = 5):
    """An exact-rules + HH match at start of round 1, both seats hold `n_cards`."""
    state = new_match(2, seed=seed, exact_rules=True, high_hand=True)
    state.hand_sizes = [n_cards, n_cards]
    state.start_next_round()
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_action_probs_sums_to_one_on_opener() -> None:
    """At the opener, CALL is illegal → all mass on bids; sums to 1."""
    agent = ModularNashAgent(
        hand_model=_PeakedHandModel(peak_bid=10),
        call_policy=_ConstCallPolicy(p=0.7),  # ignored: CALL not legal
        bid_policy=_UniformBidPolicy(),
    )
    state = _start_state(seed=1)
    probs = agent.action_probs(state)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert CALL_ACTION not in probs
    assert HH_ACTION not in probs


def test_action_probs_mixes_call_and_bids_with_standing_bid() -> None:
    """With a standing bid, mass splits as `p_call : (1 - p_call) * pi`."""
    state = _start_state(seed=2)
    state.apply_action(0)  # P0 opens with bid index 0 (lowest)
    agent = ModularNashAgent(
        hand_model=_PeakedHandModel(peak_bid=NUM_BIDS - 1),  # peak far from standing bid 0
        call_policy=_ConstCallPolicy(p=0.4),
        bid_policy=_UniformBidPolicy(),
    )
    probs = agent.action_probs(state)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert probs.get(CALL_ACTION, 0.0) == pytest.approx(0.4, abs=1e-5)
    # HH must NOT fire (peak != standing bid).
    assert HH_ACTION not in probs
    # Remaining 0.6 spread across legal feasible bids.
    bid_mass = sum(p for a, p in probs.items() if a < NUM_BIDS)
    assert bid_mass == pytest.approx(0.6, abs=1e-5)


def test_hh_gate_fires_when_peak_matches_standing_bid() -> None:
    """If HandModel.q peaks on the standing bid, agent returns {HH: 1.0}."""
    state = _start_state(seed=3)
    standing = 5  # arbitrary low bid
    state.apply_action(standing)  # P0 opens
    agent = ModularNashAgent(
        hand_model=_PeakedHandModel(peak_bid=standing),
        call_policy=_ConstCallPolicy(p=0.4),
        bid_policy=_UniformBidPolicy(),
    )
    probs = agent.action_probs(state)
    assert probs == {HH_ACTION: 1.0}
    dec = agent.decision(state)
    assert dec.hh_fired is True


def test_choose_action_returns_legal_action() -> None:
    state = _start_state(seed=4)
    state.apply_action(0)
    agent = ModularNashAgent(
        hand_model=_PeakedHandModel(peak_bid=NUM_BIDS - 1),
        call_policy=_ConstCallPolicy(p=0.5),
        bid_policy=_UniformBidPolicy(),
    )
    a = agent.choose_action(state)
    assert a in state.legal_actions()


def test_choose_action_uses_match_rng_reproducibly() -> None:
    """Two states with identical seeds + actions produce identical samples."""
    agent = ModularNashAgent(
        hand_model=_PeakedHandModel(peak_bid=NUM_BIDS - 1),
        call_policy=_ConstCallPolicy(p=0.5),
        bid_policy=_UniformBidPolicy(),
    )

    def _sample(seed):
        s = _start_state(seed=seed)
        s.apply_action(0)
        return agent.choose_action(s)

    assert _sample(123) == _sample(123)
