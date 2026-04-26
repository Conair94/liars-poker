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
      at-least rules:
          caller wins  (+1)  if pool_best < b  (bid was a bluff)
          caller loses (-1)  if pool_best >= b (bid was truthful)
      exact rules:
          caller wins  (+1)  if no 5-card subset of pool evaluates to exactly b
          caller loses (-1)  if some 5-card subset evaluates to exactly b
  - The game tree is finite (bids are strictly increasing; at most NUM_BIDS
    raises before the only legal action is a call).

Equilibrium values and policies are cached to data/probs/blind_equilibrium.json.

Usage (from the papers/Liars poker/ root):
    python -m agent.baseline.blind_equilibrium [--exact] [--compare]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from itertools import combinations

# ---------------------------------------------------------------------------
# Path setup: src/training/probs/ for poker_math_exact, src/ for game.bids
# ---------------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.abspath(os.path.join(_HERE, "..", ".."))
_PROBS_DIR = os.path.join(_SRC_DIR, "training", "probs")
_REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))

for _p in (_PROBS_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from poker_math_exact import _evaluate_ranked, get_hand_rank_counts  # noqa: E402

from game.bids import (  # noqa: E402
    CALL_ACTION,
    NUM_BIDS,
    all_bids,
    index_to_bid,
    normalize_hand_type,
)

_DATA_DIR   = os.path.join(_REPO_ROOT, "data", "probs")
_CACHE_FILE = os.path.join(_DATA_DIR, "blind_equilibrium.json")


# ---------------------------------------------------------------------------
# Per-bid P(pool_best >= bid) for pool size n  [at-least rules]
# ---------------------------------------------------------------------------

def _compute_bid_at_least(rank_counts: dict[tuple[int, int], int], total: int) -> list[float]:
    """
    From rank-level MC counts, compute P(pool_best >= bid) for every bid in
    the canonical bid ordering (index 0 = weakest, NUM_BIDS-1 = strongest).

    The result is a list of length NUM_BIDS.  Index i gives
    P(pool_best >= all_bids()[i]).
    """
    bids = all_bids()

    # Precompute cumulative "at-least" count from the top of the bid order.
    # We iterate bids from strongest to weakest, accumulating.
    p_at_least: list[float] = [0.0] * NUM_BIDS

    cumulative = 0
    for idx in range(NUM_BIDS - 1, -1, -1):
        b = bids[idx]
        key = (b.hand_type, b.primary_rank)
        cumulative += rank_counts.get(key, 0)
        p_at_least[idx] = cumulative / total

    return p_at_least


# ---------------------------------------------------------------------------
# Per-bid P(pool contains exact 5-card subset for bid)  [exact rules]
# ---------------------------------------------------------------------------

def _compute_bid_exact(n: int, n_samples: int = 500_000, seed: int = 42) -> list[float]:
    """
    Monte Carlo estimate of P(pool of n cards contains a 5-card subset whose
    best hand evaluates to exactly bid b) for every bid b.

    For n < 5: evaluates all pool cards together (matches has_exact_hand
    engine behavior for small pools — no 5-card subsets possible).
    For n >= 5: iterates all C(n, 5) subsets per sample and collects the
    set of distinct (hand_type, primary_rank) values present.

    Returns a list of length NUM_BIDS.
    """
    rng = random.Random(seed)
    deck = list(range(52))
    bids = all_bids()
    bid_keys = [(b.hand_type, b.primary_rank) for b in bids]

    counts = [0] * NUM_BIDS

    use_combos = n >= 5

    for _ in range(n_samples):
        rng.shuffle(deck)
        pool = deck[:n]

        seen: set = set()
        if use_combos:
            for combo in combinations(pool, 5):
                raw_t, raw_p = _evaluate_ranked(list(combo))
                t, p = normalize_hand_type(raw_t, raw_p)
                seen.add((t, p))
        else:
            raw_t, raw_p = _evaluate_ranked(pool)
            t, p = normalize_hand_type(raw_t, raw_p)
            seen.add((t, p))

        for idx, key in enumerate(bid_keys):
            if key in seen:
                counts[idx] += 1

    return [c / n_samples for c in counts]


# ---------------------------------------------------------------------------
# Backward-induction solver for N=2 blind game
# ---------------------------------------------------------------------------

def _solve_n2(bid_holds_prob: list[float]) -> tuple[list[list[float]], list[list[int]]]:
    """
    Backward induction for the N=2 blind game.

    Works for both at-least and exact rules — the only difference is the
    meaning of bid_holds_prob[i]:
      at-least: P(pool_best >= bid_i)
      exact:    P(pool contains a 5-card subset evaluating to exactly bid_i)

    Returns:
        values  — values[bid_idx][actor]  = EV for the current actor when the
                  standing bid has index bid_idx.  EV is in [-1, +1] (per-round).
        policy  — policy[bid_idx][actor]  = optimal action for the current
                  actor.  CALL_ACTION means call; otherwise it's a bid index
                  (the index of the bid to raise to).

    Caller EV at bid_idx:
        1 - 2 * bid_holds_prob[bid_idx]
        Reasoning: caller wins iff bid does NOT hold → EV = (1-p)*(+1) + p*(-1)

    Raise-to-b' EV (zero-sum):
        - values[b'][1 - actor]

    Tie-breaking: call preferred over raise when EVs are equal (conservative).
    """
    values: list[list[float]] = [[0.0, 0.0] for _ in range(NUM_BIDS)]
    policy: list[list[int]]   = [[CALL_ACTION, CALL_ACTION] for _ in range(NUM_BIDS)]

    for bid_idx in range(NUM_BIDS - 1, -1, -1):
        for actor in range(2):
            call_ev = 1.0 - 2.0 * bid_holds_prob[bid_idx]

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


def _solve_initial(values: list[list[float]]) -> tuple[int, float]:
    """
    Optimal first bid for player 0 (who must bid; calling is illegal before any bid).

    Returns (initial_bid_idx, initial_value_for_player0).
    """
    best_ev     = float("-inf")
    best_bid    = 0

    for b in range(NUM_BIDS):
        ev = -values[b][1]
        if ev > best_ev:
            best_ev  = ev
            best_bid = b

    return best_bid, best_ev


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def get_blind_equilibrium(n: int, n_samples: int = 3_000_000, seed: int = 42) -> dict:
    """
    Return the N=2 blind equilibrium for pool size n under at-least rules.

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

    print(f"  [blind_eq at-least] n={n}: running Monte Carlo ({n_samples:,} samples)...",
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


def get_blind_equilibrium_exact(
    n: int,
    n_samples: int = 500_000,
    seed: int = 42,
) -> dict:
    """
    Return the N=2 blind equilibrium for pool size n under exact rules.

    Cache key: "exact_{n}".  Stored in the same JSON file as at-least results.

    Returned dict keys:
        n                 — pool size
        p_exact           — list[float], length NUM_BIDS
                            P(pool contains a 5-card subset evaluating to exactly bid)
        values            — list[list[float]], shape [NUM_BIDS][2]
        policy            — list[list[int]],  shape [NUM_BIDS][2]
        initial_bid       — int, optimal first bid index for player 0
        initial_value     — float, EV for player 0 at game start
    """
    cache = _load_cache()
    key   = f"exact_{n}"

    if key in cache:
        return cache[key]

    print(f"  [blind_eq exact]    n={n}: running Monte Carlo ({n_samples:,} samples, "
          f"C({n},5)={len(list(combinations(range(n), 5))) if n >= 5 else 1} combos/sample)...",
          end=" ", flush=True)
    p_exact               = _compute_bid_exact(n, n_samples=n_samples, seed=seed)
    print("done")

    values, policy         = _solve_n2(p_exact)
    initial_bid, init_val  = _solve_initial(values)

    entry = {
        "n":             n,
        "p_exact":       p_exact,
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


def print_equilibrium_summary(eq: dict, rule_label: str = "at-least") -> None:
    """Print a compact summary of the equilibrium for one pool size."""
    n           = eq["n"]
    prob_key    = "p_exact" if "p_exact" in eq else "p_at_least"
    probs       = eq[prob_key]
    policy      = eq["policy"]
    init_bid    = eq["initial_bid"]
    init_val    = eq["initial_value"]

    bids = all_bids()

    print(f"\n=== N=2 Blind Equilibrium [{rule_label}]  (pool n={n}) ===")
    print(f"  Player 0 first bid: {bids[init_bid]}  (EV={init_val:+.4f})")

    print("\n  Standing bid threshold analysis:")
    print(f"  {'Bid':<26} {'P(bid holds)':>13}  {'call EV':>8}  "
          f"{'P0 action':<22}  {'P1 action':<22}")
    print("  " + "-" * 97)

    bids_shown = 0
    for idx, bid in enumerate(bids):
        p = probs[idx]
        call_ev = 1.0 - 2.0 * p
        act0 = _describe_action(policy[idx][0])
        act1 = _describe_action(policy[idx][1])
        prev_call = (idx == 0) or (policy[idx - 1][0] != policy[idx][0]
                                   or policy[idx - 1][1] != policy[idx][1])
        if prev_call or bids_shown < 3 or idx >= NUM_BIDS - 3:
            print(f"  [{idx:3d}] {str(bid):<22}  {p:>13.4f}  {call_ev:>+8.4f}  "
                  f"{act0:<22}  {act1:<22}")
            bids_shown += 1


def print_comparison(n: int, eq_al: dict, eq_ex: dict) -> None:
    """Side-by-side comparison of at-least vs exact equilibria for pool size n."""
    bids = all_bids()
    al_bid  = eq_al["initial_bid"]
    ex_bid  = eq_ex["initial_bid"]
    al_val  = eq_al["initial_value"]
    ex_val  = eq_ex["initial_value"]
    al_p    = eq_al["p_at_least"][al_bid]
    ex_p    = eq_ex["p_exact"][ex_bid]

    changed = "  <-- CHANGED" if al_bid != ex_bid else ""
    print(f"  n={n:2d}:  at-least first bid = {str(bids[al_bid]):<22} "
          f"(p={al_p:.3f}, EV={al_val:+.4f})  |  "
          f"exact first bid = {str(bids[ex_bid]):<22} "
          f"(p={ex_p:.3f}, EV={ex_val:+.4f}){changed}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute N=2 blind equilibria.")
    parser.add_argument("--exact",   action="store_true",
                        help="Compute exact-rules equilibria (in addition to at-least).")
    parser.add_argument("--compare", action="store_true",
                        help="Print side-by-side comparison table (implies --exact).")
    parser.add_argument("--n-min",   type=int, default=2,  help="Minimum pool size (default 2).")
    parser.add_argument("--n-max",   type=int, default=10, help="Maximum pool size (default 10).")
    parser.add_argument("--samples-exact", type=int, default=500_000,
                        help="MC samples for exact computation (default 500,000).")
    args = parser.parse_args()

    run_exact = args.exact or args.compare
    n_range = range(args.n_min, args.n_max + 1)

    print(f"Computing N=2 blind equilibria for n={args.n_min}..{args.n_max}...")
    print("(Results cached to data/probs/blind_equilibrium.json)\n")

    for n in n_range:
        eq_al = get_blind_equilibrium(n)
        if not run_exact:
            print_equilibrium_summary(eq_al, rule_label="at-least")

    if run_exact:
        print()
        for n in n_range:
            eq_ex = get_blind_equilibrium_exact(n, n_samples=args.samples_exact)
            if not args.compare:
                print_equilibrium_summary(eq_ex, rule_label="exact")

    if args.compare:
        print("\n" + "=" * 110)
        print("COMPARISON: at-least vs exact-rules first bid")
        print("=" * 110)
        for n in n_range:
            eq_al = get_blind_equilibrium(n)
            eq_ex = get_blind_equilibrium_exact(n)
            print_comparison(n, eq_al, eq_ex)
