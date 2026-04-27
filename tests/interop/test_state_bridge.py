"""Tests for adapter↔engine state bridge (P5-#2 Phase A, S2)."""

from __future__ import annotations

import os
import random
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pyspiel  # noqa: E402

import interop  # noqa: E402,F401
from game.bids import HH_ACTION  # noqa: E402
from interop.state_bridge import adapter_state_to_match_state  # noqa: E402


def _new_dealt_state(seed: int):
    g = pyspiel.load_game("python_liars_poker_exact", {"num_players": 2, "hand_size": 5})
    s = g.new_initial_state()
    rng = random.Random(seed)
    while s.is_chance_node():
        outs = [o for o, _ in s.chance_outcomes()]
        s.apply_action(rng.choice(outs))
    return s, rng


def test_bridge_rejects_pre_deal():
    g = pyspiel.load_game("python_liars_poker_exact", {"num_players": 2, "hand_size": 5})
    s = g.new_initial_state()
    with pytest.raises(ValueError):
        adapter_state_to_match_state(s)


def test_bridge_post_deal_no_bids():
    s, _ = _new_dealt_state(seed=1)
    m = adapter_state_to_match_state(s)
    assert m.exact_rules is True
    assert m.high_hand is True
    assert m.num_players == 2
    assert m.hand_sizes == [5, 5]
    assert m.round_state is not None
    assert m.round_state.current_bid is None
    assert m.round_state.current_player == 0
    # Hands recovered.
    for p in range(2):
        assert sorted(m.round_state.hands[p]) == sorted(s._hand(p))
    assert sorted(m.legal_actions()) == sorted(s.legal_actions())


def test_bridge_after_some_bids():
    s, rng = _new_dealt_state(seed=2)
    # Place a few bids without terminating.
    for _ in range(3):
        legal = [a for a in s.legal_actions() if a < 100]  # bid actions only
        if not legal:
            break
        s.apply_action(rng.choice(legal))
    m = adapter_state_to_match_state(s)
    # Legal-action sets agree (adapter always exposes HH; engine has hh=True).
    assert sorted(m.legal_actions()) == sorted(s.legal_actions())
    assert m.round_state.current_player == s._next_player
    # History length matches.
    assert len(m.round_state.history) == len(s._bets)


def test_bridge_legal_action_parity_random_walks():
    """Random-walk a few games; at every non-terminal step bridged legal
    actions match adapter legal actions exactly."""
    for seed in range(5):
        s, rng = _new_dealt_state(seed=seed + 100)
        while not s.is_terminal():
            m = adapter_state_to_match_state(s)
            assert sorted(m.legal_actions()) == sorted(s.legal_actions())
            assert m.current_player() == s.current_player()
            a = rng.choice(s.legal_actions())
            s.apply_action(a)
