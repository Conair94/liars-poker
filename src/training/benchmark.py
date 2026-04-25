"""
Agent benchmark: plays every same-rules agent pair against each other for N matches
and saves the head-to-head win-rate matrix to agent/data/benchmark_results.json.

Usage:
    cd "Liars poker/"
    python -m agent.benchmark [--games 100] [--seed 0] [--groups standard exact]

Groups:
    standard        — standard rules, countup from n=2
    exact           — exact rules + high hand, countup from n=2
    exact_ladder    — ladder agents only, exact rules, countdown from n=5/player
                      (tests Bayesian/adaptive features that require n>=5)

Cross-ruleset matchups are always skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from itertools import combinations

# ---------------------------------------------------------------------------
# Path setup
_HERE      = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.abspath(os.path.join(_HERE, ".."))
_PROBS_DIR = os.path.join(_SRC_DIR, "training", "probs")
_REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
for _p in (_PROBS_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from game.engine import new_match
from agents.registry import AGENT_REGISTRY, build_agent

# ---------------------------------------------------------------------------
# Agent lists
STANDARD_AGENTS = ["biased30", "biased40", "random", "biased60", "biased70", "blind", "conditional"]
EXACT_AGENTS    = [
    # Sophistication ladder (rung 1→4)
    "exactconditional",
    "exact_mixed",
    "exact_opp_model",
    "exact_adaptive",
    # Reference agents
    "cfr_nash_mb3",
    "exact_biased70", "exact_biased60", "exact_biased40", "exact_biased30", "exact_random",
]
LADDER_AGENTS   = ["exactconditional", "exact_mixed", "exact_opp_model", "exact_adaptive"]

# ---------------------------------------------------------------------------
# Benchmark groups
#
# match_kwargs are forwarded directly to new_match().  The extra "mode" key
# controls countup vs countdown; "hand_size" sets the starting hand size (only
# meaningful for countdown — countdown always starts at MAX_HAND_SIZE=5).
GROUPS = {
    "standard": {
        "keys":        STANDARD_AGENTS,
        "match_kwargs": {"exact_rules": False, "high_hand": False, "five_kings": False},
    },
    "exact": {
        "keys":        EXACT_AGENTS,
        "match_kwargs": {"exact_rules": True, "high_hand": True, "five_kings": False},
    },
    # Ladder-only group in countdown mode: each player starts with 5 cards (n=10).
    # Conditional tables (n>=5) are active for the entire game, giving the
    # Bayesian opponent model and adaptive call threshold enough data to act.
    "exact_ladder": {
        "keys":        LADDER_AGENTS,
        "match_kwargs": {"exact_rules": True, "high_hand": True, "five_kings": False,
                         "mode": "countdown"},
    },
}

ALL_GROUP_NAMES = list(GROUPS.keys())


def run_match(key_a: str, key_b: str, match_kwargs: dict, seed: int) -> int:
    """Run a single 2-player match; return the winning seat (0=a, 1=b)."""
    agent_a = build_agent(key_a)
    agent_b = build_agent(key_b)
    agents = [agent_a, agent_b]

    st = new_match(num_players=2, seed=seed, **match_kwargs)
    st.start_next_round()

    while not st.terminal:
        rs = st.round_state
        if rs is None:
            st.start_next_round()
            continue
        cp = rs.current_player
        action = agents[cp].choose_action(st)
        result = st.apply_action(action)
        if result is not None and not st.terminal:
            st.start_next_round()

    return st.winner


def benchmark_pair(key_a: str, key_b: str, match_kwargs: dict,
                   n_games: int, base_seed: int) -> dict:
    wins_a = 0
    wins_b = 0
    for g in range(n_games):
        # Alternate who opens to remove first-bidder advantage.
        if g % 2 == 0:
            winner_seat = run_match(key_a, key_b, match_kwargs, seed=base_seed + g)
            if winner_seat == 0:
                wins_a += 1
            else:
                wins_b += 1
        else:
            winner_seat = run_match(key_b, key_a, match_kwargs, seed=base_seed + g)
            if winner_seat == 0:
                wins_b += 1
            else:
                wins_a += 1
    return {
        "agent_a": key_a,
        "agent_b": key_b,
        "games": n_games,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "win_rate_a": round(wins_a / n_games, 4),
        "win_rate_b": round(wins_b / n_games, 4),
    }


def run_benchmark(n_games: int = 100, base_seed: int = 0,
                  group_names: list | None = None) -> dict:
    active_groups = {
        k: v for k, v in GROUPS.items()
        if group_names is None or k in group_names
    }

    total_pairs = sum(
        len(list(combinations(g["keys"], 2))) for g in active_groups.values()
    )
    results: dict = {}
    done = 0

    for group_name, group in active_groups.items():
        keys = [k for k in group["keys"] if k in AGENT_REGISTRY]
        match_kwargs = group["match_kwargs"]

        for key_a, key_b in combinations(keys, 2):
            pair_key = f"{key_a}_vs_{key_b}"
            done += 1
            print(f"[{done}/{total_pairs}] {key_a} vs {key_b} ({n_games} games)...", flush=True)
            t0 = time.time()
            results[pair_key] = benchmark_pair(key_a, key_b, match_kwargs, n_games, base_seed)
            results[pair_key]["group"] = group_name
            elapsed = time.time() - t0
            r = results[pair_key]
            print(f"  → {key_a}: {r['wins_a']}/{n_games}  {key_b}: {r['wins_b']}/{n_games}  ({elapsed:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run agent benchmarks.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--groups", nargs="+", choices=ALL_GROUP_NAMES, default=None,
        metavar="GROUP",
        help=f"Which groups to run (default: all). Choices: {ALL_GROUP_NAMES}",
    )
    args = parser.parse_args()

    active = args.groups or ALL_GROUP_NAMES
    print(f"Running benchmark: {args.games} games/pair, seed={args.seed}, groups={active}")
    results = run_benchmark(n_games=args.games, base_seed=args.seed, group_names=active)

    output = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "games_per_pair": args.games,
        "seed": args.seed,
        "groups": active,
        "results": results,
    }

    out_path = os.path.join(_REPO_ROOT, "data", "runs", "benchmark", "benchmark_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
