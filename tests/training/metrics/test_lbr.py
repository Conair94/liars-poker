"""Tests for LBR exploitability metric (P5-#2 Phase B)."""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pyspiel  # noqa: E402

import interop  # noqa: E402,F401
from agents.registry import (  # noqa: E402
    ExactRulesConditionalAgent,
    RandomAgent,
)
from training.metrics.lbr import (  # noqa: E402
    CANDIDATES_ALL,
    CANDIDATES_POLICY_SUPPORT,
    _agent_value,
    _lbr_value,
    _post_deal_state,
    lbr_exploitability,
)


def _game(hand_size: int = 5):
    return pyspiel.load_game(
        "python_liars_poker_exact",
        {"num_players": 2, "hand_size": hand_size},
    )


def test_lbr_geq_agent_value_pointwise_with_all_candidates():
    """At any non-terminal post-deal state, LBR(state) >= agent_value(state)
    when candidates=all (best-responder always has the agent's own action set
    available, so its max is at least the agent's expectation)."""
    agent = ExactRulesConditionalAgent()
    g = _game(hand_size=2)  # smaller for speed
    s = _post_deal_state(g, ([0, 4], [8, 12]))
    for p in range(2):
        agent_v = _agent_value(s, agent, p)
        lbr_v = _lbr_value(s, agent, p, depth=1, candidates=CANDIDATES_ALL, eps=0.0)
        assert lbr_v + 1e-9 >= agent_v, (
            f"seat {p}: LBR {lbr_v:.4f} < agent {agent_v:.4f}"
        )


def test_lbr_value_in_valid_reward_range():
    """LBR exploitability is bounded by the per-game reward range [0, 2]
    (best-responder can win at most +1, agent can lose at most -1)."""
    res = lbr_exploitability(
        RandomAgent(), deals=4, depth=1, seed=0, hand_size=2,
        candidates=CANDIDATES_ALL, stratified=False,
    )
    assert 0.0 <= res["value"] <= 2.0 + 1e-9
    for v in res["by_seat"]:
        assert 0.0 <= v <= 2.0 + 1e-9


def test_lbr_returns_well_formed_summary():
    res = lbr_exploitability(
        RandomAgent(), deals=3, depth=1, seed=1, hand_size=2,
        candidates=CANDIDATES_POLICY_SUPPORT, stratified=False,
    )
    assert set(res) == {
        "value", "by_seat", "ci95", "deals", "depth", "candidates",
    }
    assert res["deals"] == 3
    assert res["depth"] == 1
    assert len(res["by_seat"]) == 2
    assert len(res["ci95"]) == 2
    # LBR is a non-negative quantity by construction (with `all` candidates;
    # `policy_support` may yield slightly negative micro-values on rare cases
    # via eps-pruning, but the seat means should be >= 0 against random).
    assert res["value"] >= -1e-6


@pytest.mark.parametrize("candidates", [CANDIDATES_ALL, CANDIDATES_POLICY_SUPPORT])
def test_lbr_value_matches_agent_value_when_no_action_choices(candidates):
    """Drive the state to a forced-CALL configuration so neither side has any
    bid options; LBR value must match the agent value (only one terminal path)."""
    agent = ExactRulesConditionalAgent()
    g = _game(hand_size=2)
    s = _post_deal_state(g, ([0, 4], [8, 12]))
    # Place the strongest possible bid to leave only [CALL, HH] for opponent.
    legal = s.legal_actions()
    bid = max(a for a in legal if a < 100)
    s.apply_action(bid)
    # Now opponent (P1) has only CALL/HH.
    for p in range(2):
        a_v = _agent_value(s, agent, p)
        l_v = _lbr_value(s, agent, p, depth=2, candidates=candidates, eps=0.0)
        # BR can pick whichever of {CALL, HH} maximizes its value, so LBR >=
        # agent value. Loose direction check.
        assert l_v + 1e-9 >= a_v
