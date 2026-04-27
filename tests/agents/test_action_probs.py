"""Tests for the P5-#2 Phase A action_probs policy contract."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agents.policy import action_probs  # noqa: E402
from agents.registry import (  # noqa: E402
    ConditionalAgent,
    ExactRulesConditionalAgent,
    ExactRulesMixedAgent,
    RandomAgent,
)
from game.engine import new_match  # noqa: E402


def _make_state(seed: int = 0, exact: bool = True, hh: bool = True):
    m = new_match(num_players=2, seed=seed, exact_rules=exact, high_hand=hh)
    m.hand_sizes = [5, 5]
    m.start_next_round()
    return m


def _is_distribution(d, legal):
    assert all(a in legal for a in d), f"unknown action(s): {set(d) - set(legal)}"
    assert all(p >= 0.0 for p in d.values())
    total = sum(d.values())
    assert abs(total - 1.0) < 1e-6, f"probs sum to {total}, not 1"


def test_default_action_probs_is_one_hot():
    """Agents without an action_probs method get a one-hot fallback."""
    state = _make_state(seed=42, exact=False, hh=False)
    agent = RandomAgent()
    d = action_probs(agent, state)
    _is_distribution(d, state.legal_actions())
    # Random agent has no action_probs override → one-hot
    assert len(d) == 1
    assert next(iter(d.values())) == 1.0


def test_conditional_agent_action_probs():
    """ExactRulesConditionalAgent's distribution is one-hot (no internal mixing)."""
    state = _make_state(seed=7)
    agent = ExactRulesConditionalAgent()
    d = action_probs(agent, state)
    _is_distribution(d, state.legal_actions())
    assert len(d) == 1
    chosen, p = next(iter(d.items()))
    assert p == 1.0
    # The single one-hot point must equal what choose_action would do.
    assert chosen == agent.choose_action(state)


def test_mixed_agent_distribution_sums_to_one():
    """ExactRulesMixedAgent exposes its 4-way internal mix on opening bids."""
    state = _make_state(seed=11)
    agent = ExactRulesMixedAgent()
    d = action_probs(agent, state)
    _is_distribution(d, state.legal_actions())
    # Opening-bid path uses _bid_distribution → up to 4 distinct bids.
    assert 1 <= len(d) <= 4


def test_mixed_agent_distribution_matches_sampling_frequency():
    """Empirical sampling frequency converges to the declared distribution."""
    state = _make_state(seed=23)
    agent = ExactRulesMixedAgent()
    declared = action_probs(agent, state)

    rng = np.random.RandomState(0)
    counts: dict[int, int] = {}
    n_trials = 4000
    for _ in range(n_trials):
        # Force np.random to use our seeded stream so the test is deterministic.
        np.random.seed(int(rng.randint(0, 2**31 - 1)))
        a = agent.choose_action(state)
        counts[a] = counts.get(a, 0) + 1

    # All sampled actions must be in the declared distribution's support.
    for a in counts:
        assert a in declared, f"sampled action {a} not in declared dist {declared}"

    for a, expected in declared.items():
        observed = counts.get(a, 0) / n_trials
        # ~3σ of 4000 Bernoulli with p=0.25 is ~0.02; allow 0.05 slack.
        assert abs(observed - expected) < 0.05, (
            f"action {a}: declared {expected:.3f}, observed {observed:.3f}"
        )


def test_action_probs_call_path():
    """Drive a bid history that should trigger the CALL deterministic path."""
    state = _make_state(seed=3)
    agent = ExactRulesConditionalAgent()
    # Place an extreme bid (top hand-rank) — opponent's response should
    # collapse to something deterministic (CALL or HH).
    legal = state.legal_actions()
    top_bid = max(a for a in legal if a < 100)  # any high index
    state.apply_action(top_bid)
    d = action_probs(agent, state)
    _is_distribution(d, state.legal_actions())
    # Either deterministic CALL/HH or a single-bid response — but always
    # a valid distribution.


def test_conditional_agent_default_fallback():
    """ConditionalAgent has no action_probs override; helper returns one-hot."""
    state = _make_state(seed=5, exact=False, hh=False)
    agent = ConditionalAgent()
    d = action_probs(agent, state)
    _is_distribution(d, state.legal_actions())
    assert len(d) == 1
