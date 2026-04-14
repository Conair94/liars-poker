"""
Blind baseline equilibrium solver for card-based Liar's Poker (N=2).

The "blind" variant removes private cards. Both players share the same
prior distribution over pool outcomes and see only the public bid history.
Nash equilibrium is computed by backward induction on the finite game tree.

Game tree structure (N=2):
  - Chance: nature draws the pool hand h ~ P(pool of size n) before the round.
            h is not revealed until someone calls.
  - Player 0 must bid first (no call before any bid exists).
  - Players alternate: each may raise to any strictly higher bid, or call.
  - On a call at standing bid b:
      caller wins  (+1)  if pool < b  (bid was a bluff)
      caller loses (-1)  if pool >= b (bid was truthful)
  - The game tree is finite (bids are strictly increasing; at most NUM_BIDS
    raises before the only legal action is a call).

Equilibrium values and policies are cached to agent/data/blind_equilibrium.json.

Usage (from the papers/Liars poker/ root):
    python -m agent.baseline.blind_equilibrium
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Path setup: make poker_math_exact importable from the paper root
# ---------------------------------------------------------------------------
_BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR    = os.path.abspath(os.path.join(_BASELINE_DIR, ".."))
_PAPER_DIR    = os.path.abspath(os.path.join(_AGENT_DIR,    ".."))

for _p in (_PAPER_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from poker_math_exact import get_hand_rank_counts  # noqa: E402
from agent.game.bids import (                       # noqa: E402
    Bid, all_bids, NUM_BIDS, CALL_ACTION,
    bid_to_index, index_to_bid,
)

_DATA_DIR   = os.path.join(_AGENT_DIR, "data")
_CACHE_FILE = os.path.join(_DATA_DIR, "blind_equilibrium.json")


# ---------------------------------------------------------------------------
# Per-bid P(pool_best >= bid) for pool size n
# ---------------------------------------------------------------------------

def _compute_bid_at_least(rank_counts: Dict[Tuple[int, int], int], total: int) -> List[float]:
    """
    From rank-level MC counts, compute P(pool_best >= bid) for every bid in
    the canonical bid ordering (index 0 = weakest, NUM_BIDS-1 = strongest).

    The result is a list of length NUM_BIDS.  Index i gives
    P(pool_best >= all_bids()[i]).
    """
    bids = all_bids()

    # Precompute cumulative "at-least" count from the top of the bid order.
    # We iterate bids from strongest to weakest, accumulating.
    p_at_least: List[float] = [0.0] * NUM_BIDS

    cumulative = 0
    for idx in range(NUM_BIDS - 1, -1, -1):
        b = bids[idx]
        key = (b.hand_type, b.primary_rank)
        cumulative += rank_counts.get(key, 0)
        p_at_least[idx] = cumulative / total

    return p_at_least


# ---------------------------------------------------------------------------
# Backward-induction solver for N=2 blind game
# ---------------------------------------------------------------------------

def _solve_n2(p_at_least: List[float]) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Backward induction for the N=2 blind game.

    Returns:
        values  — values[bid_idx][actor]  = EV for the current actor when the
                  standing bid has index bid_idx.  EV is in [-1, +1] (per-round).
        policy  — policy[bid_idx][actor]  = optimal action for the current
                  actor.  CALL_ACTION means call; otherwise it's a bid index
                  (the index of the bid to raise to).

    Caller EV at bid_idx:
        1 - 2 * p_at_least[bid_idx]
        Reasoning: caller wins iff pool < bid  → P(pool < bid) = 1 - p_at_least
                   caller loses iff pool >= bid → P(pool >= bid) = p_at_least
                   EV = (1 - p_at_least) * (+1) + p_at_least * (-1)

    Raise-to-b' EV (zero-sum):
        - values[b'][1 - actor]
        Because the actor receives the negative of whatever value the
        opponent achieves at the next state.

    Tie-breaking: call preferred over raise when EVs are equal (conservative).
    """
    # values[bid_idx][actor], policy[bid_idx][actor]
    values: List[List[float]] = [[0.0, 0.0] for _ in range(NUM_BIDS)]
    policy: List[List[int]]   = [[CALL_ACTION, CALL_ACTION] for _ in range(NUM_BIDS)]

    for bid_idx in range(NUM_BIDS - 1, -1, -1):
        for actor in range(2):
            call_ev = 1.0 - 2.0 * p_at_least[bid_idx]

            best_ev     = call_ev
            best_action = CALL_ACTION

            for b_prime in range(bid_idx + 1, NUM_BIDS):
                raise_ev = -values[b_prime][1 - actor]
                if raise_ev > best_ev:
                    best_ev     = raise_ev
                    best_action = b_prime

            values[bid_idx][actor] = best_ev
            policy[bid_idx][actor] = best_action

    return values, policy


def _solve_initial(values: List[List[float]]) -> Tuple[int, float]:
    """
    Optimal first bid for player 0 (who must bid; calling is illegal before any bid).

    Returns (initial_bid_idx, initial_value_for_player0).
    """
    best_ev     = float("-inf")
    best_bid    = 0

    for b in range(NUM_BIDS):
        # After player 0 bids b, player 1 faces state (b, actor=1).
        ev = -values[b][1]
        if ev > best_ev:
            best_ev  = ev
            best_bid = b

    return best_bid, best_ev


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_blind_equilibrium(n: int, n_samples: int = 3_000_000, seed: int = 42) -> dict:
    """
    Return the N=2 blind equilibrium for pool size n.

    Loads from JSON cache if available; otherwise runs MC + backward
    induction and saves.

    Returned dict keys:
        n                 — pool size
        p_at_least        — list[float], length NUM_BIDS
        values            — list[list[float]], shape [NUM_BIDS][2]
        policy            — list[list[int]],  shape [NUM_BIDS][2]
        initial_bid       — int, optimal first bid index for player 0
        initial_value     — float, EV for player 0 at game start
    """
    cache = _load_cache()
    key   = str(n)

    if key in cache:
        return cache[key]

    print(f"  [blind_eq] n={n}: running Monte Carlo ({n_samples:,} samples)...",
          end=" ", flush=True)
    rank_counts, total = get_hand_rank_counts(n, n_samples=n_samples, seed=seed)
    print("done")

    p_at_least             = _compute_bid_at_least(rank_counts, total)
    values, policy         = _solve_n2(p_at_least)
    initial_bid, init_val  = _solve_initial(values)

    entry = {
        "n":             n,
        "p_at_least":    p_at_least,
        "values":        values,
        "policy":        policy,
        "initial_bid":   initial_bid,
        "initial_value": init_val,
    }

    cache[key] = entry
    _save_cache(cache)
    return entry


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    if os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


# ---------------------------------------------------------------------------
# Human-readable summary helpers
# ---------------------------------------------------------------------------

def _describe_action(action: int) -> str:
    if action == CALL_ACTION:
        return "CALL"
    return f"raise → {index_to_bid(action)}"


def print_equilibrium_summary(eq: dict) -> None:
    """Print a compact summary of the equilibrium for one pool size."""
    n           = eq["n"]
    p_at_least  = eq["p_at_least"]
    values      = eq["values"]
    policy      = eq["policy"]
    init_bid    = eq["initial_bid"]
    init_val    = eq["initial_value"]

    bids = all_bids()

    print(f"\n=== N=2 Blind Equilibrium  (pool n={n}) ===")
    print(f"  Player 0 first bid: {bids[init_bid]}  (EV={init_val:+.4f})")

    # Show the range of bids where calling is optimal (EV >= raise EV)
    # and where raising is still preferred.
    print(f"\n  Standing bid threshold analysis:")
    print(f"  {'Bid':<26} {'P(pool>=bid)':>13}  {'call EV':>8}  "
          f"{'P0 action':<22}  {'P1 action':<22}")
    print("  " + "-" * 97)

    bids_shown = 0
    for idx, bid in enumerate(bids):
        p = p_at_least[idx]
        call_ev = 1.0 - 2.0 * p
        act0 = _describe_action(policy[idx][0])
        act1 = _describe_action(policy[idx][1])
        # Only print bids near decision boundaries to keep output compact.
        prev_call = (idx == 0) or (policy[idx - 1][0] != policy[idx][0]
                                   or policy[idx - 1][1] != policy[idx][1])
        if prev_call or bids_shown < 3 or idx >= NUM_BIDS - 3:
            print(f"  [{idx:3d}] {str(bid):<22}  {p:>13.4f}  {call_ev:>+8.4f}  "
                  f"{act0:<22}  {act1:<22}")
            bids_shown += 1


# ---------------------------------------------------------------------------
# CLI: compute and cache all n=5..25, print summaries
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Computing N=2 blind equilibria for n=5..25...")
    print("(Results cached to agent/data/blind_equilibrium.json)")

    for n in range(5, 26):
        eq = get_blind_equilibrium(n)
        print_equilibrium_summary(eq)
