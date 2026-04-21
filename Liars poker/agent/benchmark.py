"""
Agent benchmark: plays every same-rules agent pair against each other for N matches
and saves the head-to-head win-rate matrix to agent/data/benchmark_results.json.

Usage:
    cd "Liars poker/"
    python -m agent.benchmark [--games 100] [--seed 0]

Standard-rules agents are played on standard rules.
Exact-rules agents are played on exact-rules + high-hand.
Cross-ruleset matchups are skipped.
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
# Path setup (mirrors agents.py)
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PAPER_DIR = os.path.abspath(os.path.join(_AGENT_DIR, ".."))
for _p in (_PAPER_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.game.engine import new_match
from agent.web.backend.agents import AGENT_REGISTRY, build_agent

# ---------------------------------------------------------------------------
# Ruleset groups — only agents within the same group are benchmarked together.
STANDARD_AGENTS = ["biased30", "biased40", "random", "biased60", "biased70", "blind", "conditional"]
EXACT_AGENTS    = ["exactconditional", "cfr_nash_mb3"]

GROUPS = {
    "standard": {
        "keys":        STANDARD_AGENTS,
        "match_kwargs": {"exact_rules": False, "high_hand": False, "five_kings": False},
    },
    "exact": {
        "keys":        EXACT_AGENTS,
        "match_kwargs": {"exact_rules": True, "high_hand": True, "five_kings": False},
    },
}


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
        # Alternate who bids first by alternating seat assignment each game
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


def run_benchmark(n_games: int = 100, base_seed: int = 0) -> dict:
    results: dict = {}
    total_pairs = sum(
        len(list(combinations(g["keys"], 2))) for g in GROUPS.values()
    )
    done = 0

    for group_name, group in GROUPS.items():
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
    args = parser.parse_args()

    print(f"Running benchmark: {args.games} games/pair, seed={args.seed}")
    results = run_benchmark(n_games=args.games, base_seed=args.seed)

    output = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "games_per_pair": args.games,
        "seed": args.seed,
        "results": results,
    }

    out_path = os.path.join(_AGENT_DIR, "data", "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
