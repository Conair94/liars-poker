"""Local Best Response (LBR) exploitability (P5-#2 Phase B).

DeepStack-style lower bound on exploitability. For each sampled deal we
freeze the agent's policy and search a best response with bounded
lookahead: at the best-responder's turns we maximize over a candidate set;
at the opponent's turns within the lookahead horizon we take expectation
under the agent's own policy; below the horizon we replace the value with
the agent-vs-agent expected continuation. The LBR value minus the agent's
own expected value is a lower bound on true exploitability.

Operates on the registered single-round 52-card adapter
(``python_liars_poker_exact``). Per-seat then averaged for symmetric reporting.

Design pin: ``docs-internal/design/p5_2_exploitability.md`` §2a.

Public API
----------
    lbr_exploitability(agent, deals, depth, seed, candidates, eps,
                       hand_size, num_players, stratified) -> dict
"""

from __future__ import annotations

import math
import os
import sys

import pyspiel

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import interop  # noqa: F401,E402  (registers games)
from agents.policy import action_probs  # noqa: E402
from interop.openspiel_adapter import _FULL_CALL, _FULL_HH  # noqa: E402
from interop.state_bridge import adapter_state_to_match_state  # noqa: E402
from training.metrics.deal_sampler import sample_deals  # noqa: E402

CANDIDATES_ALL = "all"
CANDIDATES_POLICY_SUPPORT = "policy_support"


def _legal_probs(agent, state) -> dict[int, float]:
    """Agent's distribution at `state`, restricted to legal actions and renormalized."""
    ms = adapter_state_to_match_state(state)
    probs = action_probs(agent, ms)
    legal = set(state.legal_actions())
    filtered = {a: p for a, p in probs.items() if a in legal and p > 0.0}
    total = sum(filtered.values())
    if total <= 0.0:
        n = len(legal)
        return {a: 1.0 / n for a in legal}
    return {a: p / total for a, p in filtered.items()}


def _agent_value(state, agent, best_responder: int) -> float:
    """E[reward to `best_responder`] under agent-vs-agent rollout from `state`."""
    if state.is_terminal():
        return state.returns()[best_responder]
    probs = _legal_probs(agent, state)
    return sum(
        p * _agent_value(state.child(a), agent, best_responder)
        for a, p in probs.items()
    )


def _lbr_candidates(state, agent, candidates: str, eps: float) -> list[int]:
    legal = state.legal_actions()
    if candidates == CANDIDATES_ALL:
        return legal
    if candidates != CANDIDATES_POLICY_SUPPORT:
        raise ValueError(f"unknown candidates mode: {candidates!r}")
    ms = adapter_state_to_match_state(state)
    probs = action_probs(agent, ms)
    supp = [a for a in legal if probs.get(a, 0.0) > eps]
    for term in (_FULL_CALL, _FULL_HH):
        if term in legal and term not in supp:
            supp.append(term)
    if not supp:
        supp = list(legal)
    return supp


def _lbr_value(
    state,
    agent,
    best_responder: int,
    depth: int,
    candidates: str,
    eps: float,
) -> float:
    """LBR's expected value to `best_responder` from `state` under depth-bounded BR."""
    if state.is_terminal():
        return state.returns()[best_responder]
    if state.current_player() == best_responder:
        cands = _lbr_candidates(state, agent, candidates, eps)
        return max(
            _lbr_value(state.child(a), agent, best_responder, depth, candidates, eps)
            for a in cands
        )
    # Opponent's turn — within lookahead use agent policy expectation; below
    # the horizon, replace with the agent-vs-agent continuation value.
    if depth <= 0:
        return _agent_value(state, agent, best_responder)
    probs = _legal_probs(agent, state)
    return sum(
        p * _lbr_value(state.child(a), agent, best_responder,
                       depth - 1, candidates, eps)
        for a, p in probs.items()
    )


def _post_deal_state(game, hands: tuple[list[int], ...]):
    """Build a post-deal adapter state by force-applying chance actions in seat order."""
    state = game.new_initial_state()
    for hand in hands:
        for c in hand:
            state.apply_action(c)
    return state


def lbr_exploitability(
    agent,
    deals: int = 200,
    depth: int = 3,
    seed: int = 0,
    candidates: str = CANDIDATES_POLICY_SUPPORT,
    eps: float = 0.01,
    hand_size: int = 5,
    num_players: int = 2,
    stratified: bool = True,
) -> dict:
    """Estimate LBR exploitability (lower bound) for `agent` on the full adapter.

    Returns:
        dict with keys
            ``value``       — mean over seats of (LBR_v - agent_v)
            ``by_seat``     — per-seat list of mean per-deal exploitability
            ``ci95``        — symmetric 95% CI half-width per seat (gaussian)
            ``deals``       — sample count
            ``depth``       — lookahead depth used
            ``candidates``  — pruning mode used
    """
    if num_players != 2:
        raise NotImplementedError("LBR currently supports 2-player only")
    game = pyspiel.load_game(
        "python_liars_poker_exact",
        {"num_players": num_players, "hand_size": hand_size},
    )
    per_deal: list[list[float]] = [[] for _ in range(num_players)]
    for hands in sample_deals(
        deals, seed,
        hand_size=hand_size,
        num_players=num_players,
        stratified=stratified,
    ):
        for p in range(num_players):
            state = _post_deal_state(game, hands)
            agent_v = _agent_value(state, agent, p)
            lbr_v = _lbr_value(state, agent, p, depth, candidates, eps)
            per_deal[p].append(lbr_v - agent_v)

    by_seat = [sum(s) / len(s) for s in per_deal]
    # Symmetric 95% gaussian CI half-width per seat (n typically large enough).
    ci95: list[float] = []
    for s, mean in zip(per_deal, by_seat):
        n = len(s)
        if n < 2:
            ci95.append(float("nan"))
            continue
        var = sum((x - mean) ** 2 for x in s) / (n - 1)
        ci95.append(1.96 * math.sqrt(var / n))

    return {
        "value": sum(by_seat) / num_players,
        "by_seat": by_seat,
        "ci95": ci95,
        "deals": deals,
        "depth": depth,
        "candidates": candidates,
    }
