"""Stable policy contract for the agent zoo (P5-#2 Phase A, S1).

Every metric in P5-#2 (LBR, sampled subgame exploitability) only ever asks
agents one question: "what is your action distribution at this infoset?"
That is the entire surface area we need; anything richer is YAGNI.

Usage
-----
    from agents.policy import action_probs

    probs = action_probs(agent, state)  # dict[action_int -> probability]

The dispatcher prefers `agent.action_probs(state)` when the agent class
implements it directly. When it does not, the default is a one-hot on
`agent.choose_action(state)` (correct for deterministic agents and a
stochastic one-sample fallback for the rest).

Invariants
----------
- Returned dict's keys are always a subset of `state.legal_actions()`.
- Values are non-negative and sum to 1.0 within `1e-6`.
- Implementations may omit zero-probability actions.
"""

from __future__ import annotations

from typing import Protocol

from game.engine import MatchState


class Policy(Protocol):
    def action_probs(self, state: MatchState) -> dict[int, float]: ...


def action_probs(agent, state: MatchState) -> dict[int, float]:
    """Return the agent's action distribution at `state`.

    Falls back to one-hot on `choose_action(state)` when the agent does not
    implement `action_probs` directly.
    """
    fn = getattr(agent, "action_probs", None)
    if callable(fn):
        return fn(state)
    return {agent.choose_action(state): 1.0}


def decision(agent, state: MatchState):
    """Return the agent's full `AgentDecision` at `state`.

    Falls back to wrapping `action_probs(agent, state)` when the agent
    does not implement `.decision()`; in that case the trace fields
    (belief / call / bid) are None.

    Importing from agents.contracts is done lazily to keep this module
    cheap to import (some test paths poke at `action_probs` without
    needing torch / numpy).
    """
    fn = getattr(agent, "decision", None)
    if callable(fn):
        return fn(state)

    from agents.contracts import AgentDecision
    probs = action_probs(agent, state)
    return AgentDecision(action_probs=probs)
