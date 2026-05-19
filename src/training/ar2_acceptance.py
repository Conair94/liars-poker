"""AR-2 Phase 8 acceptance gate.

Runs the three checks from the checklist:
    1. Win-rate gate: modular_nash vs exactconditional, 200 games 1v1 5-card
       (exact_rules + high_hand). 95% CI must exclude 50%, and win-rate must
       be >= 55% (i.e. >= 5 pp over the opponent).
    2. Exploitability gate: subgame_exploitability at hand_size=5, num_players=2
       (pool n=10) on 200 deals. modular_nash must be strictly below
       exactconditional.
    3. Distillation-budget gate: was satisfied at Phase 7 (elbow at N=10k).
       Re-asserted here from the sweep elbow.json for completeness.

Outputs JSON to data/runs/ar2_acceptance/<timestamp>/results.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import UTC, datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_SRC, ".."))
_PROBS = os.path.join(_SRC, "training", "probs")
for _p in (_PROBS, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.registry import build_agent  # noqa: E402
from training.benchmark import benchmark_pair  # noqa: E402
from training.metrics.subgame_exploitability import subgame_exploitability  # noqa: E402


def _win_rate_ci(wins: int, games: int) -> tuple[float, float, float]:
    p = wins / games
    se = math.sqrt(p * (1 - p) / games)
    return p, p - 1.96 * se, p + 1.96 * se


def run_acceptance(
    games: int,
    deals: int,
    seed: int,
    elbow_path: str | None,
) -> dict:
    match_kwargs = {"exact_rules": True, "high_hand": True, "five_kings": False}

    print(f"[gate 1] win-rate: modular_nash vs exactconditional, {games} games...", flush=True)
    t0 = time.time()
    bench = benchmark_pair(
        "modular_nash", "exactconditional",
        match_kwargs=match_kwargs,
        n_games=games, base_seed=seed,
    )
    wr_elapsed = time.time() - t0
    p, lo, hi = _win_rate_ci(bench["wins_a"], games)
    win_rate_pass = (p >= 0.55) and (lo > 0.5)
    print(f"  modular_nash {bench['wins_a']}/{games} = {p:.3f} (95% CI [{lo:.3f}, {hi:.3f}])  {wr_elapsed:.1f}s")
    print(f"  win-rate gate: {'PASS' if win_rate_pass else 'FAIL'}")

    print(f"[gate 2] exploitability at hand_size=5 (n=10), {deals} deals...", flush=True)
    t0 = time.time()
    a_mn = build_agent("modular_nash")
    expl_mn = subgame_exploitability(a_mn, deals=deals, seed=seed, hand_size=5, num_players=2)
    print(f"  modular_nash      value={expl_mn['value']:.4f}  ({time.time()-t0:.1f}s)")
    t0 = time.time()
    a_ec = build_agent("exactconditional")
    expl_ec = subgame_exploitability(a_ec, deals=deals, seed=seed, hand_size=5, num_players=2)
    print(f"  exactconditional  value={expl_ec['value']:.4f}  ({time.time()-t0:.1f}s)")
    expl_pass = expl_mn["value"] < expl_ec["value"]
    print(f"  exploitability gate: {'PASS' if expl_pass else 'FAIL'}")

    elbow_block: dict = {}
    elbow_pass = None
    if elbow_path and os.path.exists(elbow_path):
        with open(elbow_path) as f:
            elbow_block = json.load(f)
        elbow_pass = int(elbow_block.get("chosen_N", -1)) == 10000
        print(f"[gate 3] elbow: chosen_N={elbow_block.get('chosen_N')} (gate {'PASS' if elbow_pass else 'FAIL'})")
    else:
        print(f"[gate 3] elbow: skipped (no elbow.json at {elbow_path})")

    return {
        "win_rate": {
            **bench,
            "p": p, "ci95_lo": lo, "ci95_hi": hi,
            "pass": win_rate_pass,
        },
        "exploitability": {
            "modular_nash": expl_mn,
            "exactconditional": expl_ec,
            "diff": expl_mn["value"] - expl_ec["value"],
            "pass": expl_pass,
        },
        "elbow": {
            **elbow_block,
            "pass": elbow_pass,
        },
        "all_pass": bool(win_rate_pass and expl_pass and (elbow_pass or elbow_pass is None)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--deals", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--elbow", default=str(os.path.join(
        _REPO, "data", "sweeps",
        "ar2-20260517T200142Z-ar2_distillation_count-72a29c18",
        "elbow.json",
    )))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or os.path.join(_REPO, "data", "runs", "ar2_acceptance", stamp)
    os.makedirs(out_dir, exist_ok=True)

    results = run_acceptance(args.games, args.deals, args.seed, args.elbow)
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "timestamp": stamp,
            "results": results,
        }, f, indent=2)
    print(f"\nResults: {out_path}")
    print(f"OVERALL: {'PASS' if results['all_pass'] else 'FAIL'}")
    return 0 if results["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
