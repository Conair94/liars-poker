"""CFR+ solver on the perfect-information bidding subgame (AR-2 §1, §2).

Given a sampled deal `D = (h0, h1)`, the bidding subgame is a small,
perfect-information, ±1 zero-sum game over the bid tree. State is keyed
on `(cur_bid_idx, current_player)`; terminal values come from the engine's
call / HH resolution rules.

This module owns the **shared primitives** the exploitability metric and
the distillation pipeline both need (legal action enumeration, terminal
return resolution, canonical `MatchState` construction). The new
`CFRPlusSubgameSolver` is the AR-2 piece — `subgame_exploitability`
re-exports the primitives so its existing callers and tests continue to
work unchanged.

HH gating during solving
------------------------
Per AR-2 design §5.1, HH is treated as a *deterministic, forced* action:
at any node where the standing bid equals the pool's exact best 5-card
hand bid, the next-to-act player is forced to declare HH. At every other
node HH is excluded from the action set. This eliminates the ε-tail on a
deterministic rule and shrinks the regret-matching state space.

The legacy `legal_subgame_actions` (used by the exploitability metric)
keeps the original "HH always available with a standing bid" semantics —
that path predates AR-2 and we preserve it for byte-equivalence.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_PROBS = os.path.join(_SRC, "training", "probs")
if _PROBS not in sys.path:
    sys.path.insert(0, _PROBS)

import random as _random

from game.bids import (  # noqa: E402
    Bid,
    CALL_ACTION,
    HH_ACTION,
    NUM_BIDS,
    bid_to_index,
    index_to_bid,
    normalize_hand_type,
)
from game.engine import MatchState, RoundState  # noqa: E402
from interop.openspiel_adapter import _has_exact_hand  # noqa: E402
from poker_math_exact import _evaluate_ranked  # noqa: E402


# ---------------------------------------------------------------------------
# Shared bidding-tree primitives
# ---------------------------------------------------------------------------

def legal_subgame_actions(cur_bid_idx: int | None) -> list[int]:
    """Action set used by the **exploitability metric** (legacy).

    HH is included whenever there is a standing bid. The CFR+ solver uses
    a different gating rule (`_actions_at` below) per AR-2 §5.1.
    """
    if cur_bid_idx is None:
        return list(range(NUM_BIDS))
    return list(range(cur_bid_idx + 1, NUM_BIDS)) + [CALL_ACTION, HH_ACTION]


def resolve_call_returns(
    hands: tuple[list[int], ...],
    cur_bid_idx: int,
    caller: int,
) -> list[float]:
    bid = index_to_bid(cur_bid_idx)
    bidder = 1 - caller
    pool = [c for h in hands for c in h]
    holds = _has_exact_hand(pool, bid)
    winner = bidder if holds else caller
    return [1.0 if i == winner else -1.0 for i in range(2)]


def resolve_hh_returns(
    hands: tuple[list[int], ...],
    cur_bid_idx: int,
    declarer: int,
) -> list[float]:
    bid = index_to_bid(cur_bid_idx)
    bidder = 1 - declarer
    pool = [c for h in hands for c in h]
    raw_t, raw_p = _evaluate_ranked(pool)
    pool_t, pool_p = normalize_hand_type(raw_t, raw_p)
    correct = (pool_t == bid.hand_type and pool_p == bid.primary_rank)
    winner = declarer if correct else bidder
    return [1.0 if i == winner else -1.0 for i in range(2)]


def build_canonical_match_state(
    hands: tuple[list[int], ...],
    cur_bid_idx: int | None,
    current_player: int,
) -> MatchState:
    """Minimum-history canonical MatchState arriving at `(cur_bid_idx, current_player)`.

    Identical semantics to the pre-AR-2 helper of the same name in
    `subgame_exploitability` — preserved verbatim for byte-equivalence.
    """
    history: list[tuple[int, int]] = []
    last_bidder = -1
    current_bid = None

    if cur_bid_idx is not None:
        if current_player == 1:
            history.append((0, cur_bid_idx))
            last_bidder = 0
        else:
            opener = 0
            if cur_bid_idx == 0:
                history.append((1, cur_bid_idx))
                last_bidder = 1
            else:
                history.append((0, opener))
                history.append((1, cur_bid_idx))
                last_bidder = 1
        current_bid = index_to_bid(cur_bid_idx)

    rs = RoundState(
        hands=[list(h) for h in hands],
        history=list(history),
        current_player=current_player,
        last_bidder=last_bidder,
        current_bid=current_bid,
    )
    hand_size = len(hands[0])
    return MatchState(
        num_players=2,
        hand_sizes=[hand_size, hand_size],
        active=[True, True],
        first_bidder_next=0,
        round_state=rs,
        round_history=[],
        terminal=False,
        winner=None,
        rng=_random.Random(),
        mode='countup',
        exact_rules=True,
        high_hand=True,
        five_kings=False,
    )


def pool_best_bid_idx(hands: tuple[list[int], ...]) -> int:
    """Index of the (normalized) best 5-card bid the pool can satisfy exactly."""
    pool = [c for h in hands for c in h]
    raw_t, raw_p = _evaluate_ranked(pool)
    pool_t, pool_p = normalize_hand_type(raw_t, raw_p)
    return bid_to_index(Bid(pool_t, pool_p))


# ---------------------------------------------------------------------------
# CFR+ solver
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubgameSolution:
    """Average strategy at convergence, keyed by (cur_bid_idx | None, current_player).

    Conventions:
    - `avg_bid_dist[key]` is a `(NUM_BIDS,)` float32 vector — bid index → prob.
    - `avg_call_prob[key]` and `avg_hh_prob[key]` are scalars in [0, 1].
    - The three quantities sum to 1 at every key (within solver tolerance).
    - HH-forced nodes do **not** appear as keys: callers walking the strategy
      detect them by re-running the gate (`pool_best_bid_idx == cur_bid_idx`).
    """
    avg_call_prob: dict[tuple[int | None, int], float]
    avg_hh_prob:   dict[tuple[int | None, int], float]
    avg_bid_dist:  dict[tuple[int | None, int], np.ndarray]
    iters_used:    int
    final_eps:     float
    visited_keys:  tuple[tuple[int | None, int], ...] = field(default_factory=tuple)


class CFRPlusSubgameSolver:
    """CFR+ on the perfect-information bidding subgame for a fixed deal.

    Algorithm: regret-matching+ with linear (t-weighted) averaging. Each
    iteration is a single full-tree traversal; the bidding tree is small
    (≲ a few thousand reachable nodes at n=10) so 500–1000 iters is
    typically sufficient to ε < 1e-3.

    Parameters
    ----------
    max_iters : int
        Cap on CFR+ iterations. Default 500 (empirically converges
        well under ε=1e-3 at n=10).
    eps : float
        Target average exploitability. If `compute_eps` is True, the
        solver runs a final BR pass and reports the achieved ε; iteration
        cap is still binding.
    seed : int
        Reserved for future stochastic variants. Vanilla CFR+ on full
        traversal is deterministic in `hands`.
    compute_eps : bool
        Compute final ε via a single BR pass after the iteration cap.
    """

    def __init__(
        self,
        *,
        max_iters: int = 500,
        eps: float = 1e-3,
        seed: int = 0,
        compute_eps: bool = False,
    ) -> None:
        self.max_iters = int(max_iters)
        self.eps = float(eps)
        self.seed = int(seed)
        self.compute_eps = bool(compute_eps)

    def solve(self, hands: tuple[list[int], ...]) -> SubgameSolution:
        """Solve via forward/backward DP over the (cur_bid_idx, current_player) DAG.

        The bidding subgame's value at any state depends only on
        `(cur_bid_idx, current_player)`, but multiple bid sequences reach
        the same state. A naive recursive CFR traversal explodes
        exponentially over those paths; the correct approach for this
        DAG is two passes per iteration:

        1. **Forward pass** (reach probabilities): top-down by cur_bid_idx,
           accumulating per-state per-player reach.
        2. **Backward pass** (values + regret/strat updates): bottom-up,
           with each state's value computed from its children's values
           weighted by the current strategy.

        Per-iteration cost is O(|states| × NUM_BIDS) ≈ 220 × 110 ≈ 24k ops.
        """
        hands_lst = tuple(list(h) for h in hands)
        best_idx = pool_best_bid_idx(hands_lst)

        # ---- Precompute terminal values per (cur_bid_idx, actor) ----
        # call_v[a][cp] = value to *each* player when player cp calls at standing bid a.
        call_v_table = np.zeros((NUM_BIDS, 2, 2), dtype=np.float64)
        hh_v_table   = np.zeros((NUM_BIDS, 2, 2), dtype=np.float64)
        for a in range(NUM_BIDS):
            for cp in (0, 1):
                cv = resolve_call_returns(hands_lst, a, cp)
                hv = resolve_hh_returns(hands_lst, a, cp)
                call_v_table[a, cp, 0] = cv[0]; call_v_table[a, cp, 1] = cv[1]
                hh_v_table[a, cp, 0] = hv[0]; hh_v_table[a, cp, 1] = hv[1]

        def hh_gate_at(cur_bid_idx: int) -> bool:
            return cur_bid_idx == best_idx

        # ---- Enumerate reachable decision states ----
        # Root: (None, 0). Then (b, 1) for b ∈ [0, NUM_BIDS) reachable via P0
        # opening with b. Then (b, 0) for b ∈ [1, NUM_BIDS) reachable via
        # P1 raising to b after some P0 bid < b. HH-gated states are NOT
        # decision states (HH is forced terminal there).
        ROOT = (None, 0)
        decision_states: list[tuple[int | None, int]] = [ROOT]
        action_sets: dict[tuple[int | None, int], list[int]] = {}
        action_sets[ROOT] = list(range(NUM_BIDS))
        for b in range(NUM_BIDS):
            if hh_gate_at(b):
                continue
            decision_states.append((b, 1))
            action_sets[(b, 1)] = list(range(b + 1, NUM_BIDS)) + [CALL_ACTION]
        for b in range(1, NUM_BIDS):
            if hh_gate_at(b):
                continue
            decision_states.append((b, 0))
            action_sets[(b, 0)] = list(range(b + 1, NUM_BIDS)) + [CALL_ACTION]

        # Topological order for forward pass: cur_bid_idx ascending.
        # `None` first, then 0, 1, ..., NUM_BIDS-1.
        def state_order_key(s):
            cur = s[0]
            return (-1 if cur is None else cur, s[1])
        forward_order = sorted(decision_states, key=state_order_key)
        backward_order = list(reversed(forward_order))

        # ---- Per-state arrays (regret_sum, strat_sum) sized to action set ----
        regret: dict[tuple[int | None, int], np.ndarray] = {
            s: np.zeros(len(action_sets[s]), dtype=np.float64) for s in decision_states
        }
        strat:  dict[tuple[int | None, int], np.ndarray] = {
            s: np.zeros(len(action_sets[s]), dtype=np.float64) for s in decision_states
        }

        def regret_matching(s) -> np.ndarray:
            r = regret[s]
            pos = np.maximum(r, 0.0)
            tot = pos.sum()
            if tot > 0:
                return pos / tot
            n = r.shape[0]
            return np.full(n, 1.0 / n, dtype=np.float64)

        for t in range(1, self.max_iters + 1):
            # Current strategies — reused in forward + backward pass.
            strategy: dict[tuple[int | None, int], np.ndarray] = {
                s: regret_matching(s) for s in decision_states
            }

            # ---- Forward pass: reach probabilities ----
            reach_p0 = {s: 0.0 for s in decision_states}
            reach_p1 = {s: 0.0 for s in decision_states}
            reach_p0[ROOT] = 1.0
            reach_p1[ROOT] = 1.0
            for s in forward_order:
                cur, cp = s
                actions = action_sets[s]
                strat_s = strategy[s]
                rp0, rp1 = reach_p0[s], reach_p1[s]
                if rp0 == 0.0 and rp1 == 0.0:
                    continue
                for ai, a in enumerate(actions):
                    if a == CALL_ACTION:
                        continue  # terminal — no successor state
                    # successor: bid `a` made by current player → next state (a, 1-cp)
                    if hh_gate_at(a):
                        continue  # forced HH terminal at the successor
                    succ = (a, 1 - cp)
                    sp = strat_s[ai]
                    if cp == 0:
                        reach_p0[succ] += rp0 * sp
                        reach_p1[succ] += rp1
                    else:
                        reach_p0[succ] += rp0
                        reach_p1[succ] += rp1 * sp

            # ---- Backward pass: values + regret/strat updates ----
            value_p0: dict[tuple[int | None, int], float] = {}
            value_p1: dict[tuple[int | None, int], float] = {}
            for s in backward_order:
                cur, cp = s
                actions = action_sets[s]
                strat_s = strategy[s]
                action_v0 = np.empty(len(actions), dtype=np.float64)
                action_v1 = np.empty(len(actions), dtype=np.float64)
                for ai, a in enumerate(actions):
                    if a == CALL_ACTION:
                        action_v0[ai] = call_v_table[cur, cp, 0]
                        action_v1[ai] = call_v_table[cur, cp, 1]
                    elif hh_gate_at(a):
                        # Successor (a, 1-cp) is forced HH; declarer is 1-cp.
                        action_v0[ai] = hh_v_table[a, 1 - cp, 0]
                        action_v1[ai] = hh_v_table[a, 1 - cp, 1]
                    else:
                        succ = (a, 1 - cp)
                        action_v0[ai] = value_p0[succ]
                        action_v1[ai] = value_p1[succ]
                v0 = float((strat_s * action_v0).sum())
                v1 = float((strat_s * action_v1).sum())
                value_p0[s] = v0
                value_p1[s] = v1

                # CFR+ regret update for current player.
                op_reach  = reach_p1[s] if cp == 0 else reach_p0[s]
                own_reach = reach_p0[s] if cp == 0 else reach_p1[s]
                if op_reach > 0.0:
                    own_v   = v0 if cp == 0 else v1
                    av_cp   = action_v0 if cp == 0 else action_v1
                    np.maximum(regret[s] + op_reach * (av_cp - own_v), 0.0,
                               out=regret[s])
                if own_reach > 0.0:
                    strat[s] += (t * own_reach) * strat_s

        # ---- Build SubgameSolution from strat_sum ----
        avg_call: dict[tuple[int | None, int], float] = {}
        avg_hh:   dict[tuple[int | None, int], float] = {}
        avg_bid:  dict[tuple[int | None, int], np.ndarray] = {}
        visited: list[tuple[int | None, int]] = []
        for s in decision_states:
            ss = strat[s]
            tot = float(ss.sum())
            if tot <= 0.0:
                # Never visited under any iteration's reach — skip.
                continue
            avg = ss / tot
            p_call = 0.0
            bid_vec = np.zeros(NUM_BIDS, dtype=np.float32)
            for ai, a in enumerate(action_sets[s]):
                if a == CALL_ACTION:
                    p_call = float(avg[ai])
                elif 0 <= a < NUM_BIDS:
                    bid_vec[a] = float(avg[ai])
            avg_call[s] = p_call
            avg_hh[s]   = 0.0
            avg_bid[s]  = bid_vec
            visited.append(s)

        final_eps = 0.0
        if self.compute_eps:
            final_eps = self._exploitability_dag(
                avg_call, avg_bid, action_sets, decision_states,
                forward_order, hh_gate_at, call_v_table, hh_v_table,
            )

        return SubgameSolution(
            avg_call_prob=avg_call,
            avg_hh_prob=avg_hh,
            avg_bid_dist=avg_bid,
            iters_used=self.max_iters,
            final_eps=final_eps,
            visited_keys=tuple(visited),
        )

    @staticmethod
    def _exploitability_dag(
        avg_call: dict,
        avg_bid: dict,
        action_sets: dict,
        decision_states: list,
        forward_order: list,
        hh_gate_at,
        call_v_table: np.ndarray,
        hh_v_table:   np.ndarray,
    ) -> float:
        """One-shot best-response value vs. average strategy. Returns mean ε."""
        backward_order = list(reversed(forward_order))
        ROOT = (None, 0)

        # Average-strategy value at each state (for both players).
        avg_v0: dict = {}
        avg_v1: dict = {}
        for s in backward_order:
            cur, cp = s
            actions = action_sets[s]
            # Reconstruct strategy from avg_call + avg_bid.
            probs = np.zeros(len(actions), dtype=np.float64)
            for ai, a in enumerate(actions):
                if a == CALL_ACTION:
                    probs[ai] = avg_call.get(s, 0.0)
                else:
                    bv = avg_bid.get(s)
                    probs[ai] = float(bv[a]) if bv is not None else 0.0
            tot = probs.sum()
            if tot <= 0.0:
                # Fallback: uniform.
                probs[:] = 1.0 / len(actions)
            else:
                probs /= tot
            v0 = v1 = 0.0
            for ai, a in enumerate(actions):
                if a == CALL_ACTION:
                    av0 = call_v_table[cur, cp, 0]; av1 = call_v_table[cur, cp, 1]
                elif hh_gate_at(a):
                    av0 = hh_v_table[a, 1 - cp, 0]; av1 = hh_v_table[a, 1 - cp, 1]
                else:
                    succ = (a, 1 - cp)
                    av0 = avg_v0[succ]; av1 = avg_v1[succ]
                v0 += probs[ai] * av0
                v1 += probs[ai] * av1
            avg_v0[s] = v0
            avg_v1[s] = v1

        # BR value for each player.
        gaps = []
        for br in (0, 1):
            br_v: dict = {}
            for s in backward_order:
                cur, cp = s
                actions = action_sets[s]
                child_v = np.empty(len(actions), dtype=np.float64)
                for ai, a in enumerate(actions):
                    if a == CALL_ACTION:
                        child_v[ai] = call_v_table[cur, cp, br]
                    elif hh_gate_at(a):
                        child_v[ai] = hh_v_table[a, 1 - cp, br]
                    else:
                        child_v[ai] = br_v[(a, 1 - cp)]
                if cp == br:
                    br_v[s] = float(child_v.max())
                else:
                    # Opponent plays the average strategy.
                    probs = np.zeros(len(actions), dtype=np.float64)
                    for ai, a in enumerate(actions):
                        if a == CALL_ACTION:
                            probs[ai] = avg_call.get(s, 0.0)
                        else:
                            bv = avg_bid.get(s)
                            probs[ai] = float(bv[a]) if bv is not None else 0.0
                    tot = probs.sum()
                    if tot <= 0.0:
                        probs[:] = 1.0 / len(actions)
                    else:
                        probs /= tot
                    br_v[s] = float((probs * child_v).sum())
            agent_v = avg_v0[ROOT] if br == 0 else avg_v1[ROOT]
            gaps.append(br_v[ROOT] - agent_v)
        return sum(gaps) / len(gaps)

__all__ = [
    "CFRPlusSubgameSolver",
    "SubgameSolution",
    "build_canonical_match_state",
    "legal_subgame_actions",
    "pool_best_bid_idx",
    "resolve_call_returns",
    "resolve_hh_returns",
]
