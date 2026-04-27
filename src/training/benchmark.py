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
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
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

from agents.registry import AGENT_REGISTRY, build_agent
from game.engine import new_match
from training.decision_capture import LoggingAgentWrapper
from training.logging import DecisionLogger

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


def run_match(key_a: str, key_b: str, match_kwargs: dict, seed: int,
              log_ctx: dict | None = None) -> int:
    """Run a single 2-player match; return the winning seat (0=a, 1=b).

    If ``log_ctx`` is provided, agent decisions are recorded to the
    supplied DecisionLogger. Required keys: ``logger`` (DecisionLogger),
    ``run_id`` (str), ``game_id`` (int).
    """
    agent_a = build_agent(key_a)
    agent_b = build_agent(key_b)
    agents = [agent_a, agent_b]

    if log_ctx is not None:
        ruleset = {k: match_kwargs.get(k, False) for k in ("exact_rules", "high_hand", "five_kings")}
        ruleset["mode"] = match_kwargs.get("mode", "countup")
        game_id_ref = [log_ctx["game_id"]]
        turn_ref = [0]
        agents = [
            LoggingAgentWrapper(
                agent_a, logger=log_ctx["logger"], run_id=log_ctx["run_id"],
                agent_name=key_a, agent_seat=0, opponent_name=key_b,
                ruleset=ruleset, game_id_ref=game_id_ref, turn_counter_ref=turn_ref,
            ),
            LoggingAgentWrapper(
                agent_b, logger=log_ctx["logger"], run_id=log_ctx["run_id"],
                agent_name=key_b, agent_seat=1, opponent_name=key_a,
                ruleset=ruleset, game_id_ref=game_id_ref, turn_counter_ref=turn_ref,
            ),
        ]

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
                   n_games: int, base_seed: int,
                   logger: DecisionLogger | None = None,
                   run_id: str | None = None,
                   game_id_start: int = 0) -> dict:
    wins_a = 0
    wins_b = 0

    def _ctx(gid: int) -> dict | None:
        if logger is None:
            return None
        return {"logger": logger, "run_id": run_id or "", "game_id": gid}

    for g in range(n_games):
        gid = game_id_start + g
        # Alternate who opens to remove first-bidder advantage.
        if g % 2 == 0:
            winner_seat = run_match(key_a, key_b, match_kwargs,
                                    seed=base_seed + g, log_ctx=_ctx(gid))
            if winner_seat == 0:
                wins_a += 1
            else:
                wins_b += 1
        else:
            winner_seat = run_match(key_b, key_a, match_kwargs,
                                    seed=base_seed + g, log_ctx=_ctx(gid))
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
                  group_names: list | None = None,
                  logger: DecisionLogger | None = None,
                  run_id: str | None = None) -> dict:
    active_groups = {
        k: v for k, v in GROUPS.items()
        if group_names is None or k in group_names
    }

    total_pairs = sum(
        len(list(combinations(g["keys"], 2))) for g in active_groups.values()
    )
    results: dict = {}
    done = 0
    game_id_cursor = 0

    for group_name, group in active_groups.items():
        keys = [k for k in group["keys"] if k in AGENT_REGISTRY]
        match_kwargs = group["match_kwargs"]

        for key_a, key_b in combinations(keys, 2):
            pair_key = f"{key_a}_vs_{key_b}"
            done += 1
            print(f"[{done}/{total_pairs}] {key_a} vs {key_b} ({n_games} games)...", flush=True)
            t0 = time.time()
            results[pair_key] = benchmark_pair(
                key_a, key_b, match_kwargs, n_games, base_seed,
                logger=logger, run_id=run_id, game_id_start=game_id_cursor,
            )
            results[pair_key]["group"] = group_name
            game_id_cursor += n_games
            elapsed = time.time() - t0
            r = results[pair_key]
            print(f"  → {key_a}: {r['wins_a']}/{n_games}  {key_b}: {r['wins_b']}/{n_games}  ({elapsed:.1f}s)")

    return results


def _compute_per_agent_exploitability(
    active_groups: list[str],
    do_lbr: bool,
    do_subgame: bool,
    deals: int,
    lbr_depth: int,
    seed: int,
) -> dict:
    """Run LBR and/or subgame metrics on every agent in the active groups.

    Only exact-rules+HH groups are eligible — the metrics' adapter is
    `python_liars_poker_exact` with HH wired (P5-1). Standard-rules agents
    are skipped with a note in the output.
    """
    from training.metrics.lbr import lbr_exploitability
    from training.metrics.subgame_exploitability import subgame_exploitability

    eligible_keys: list[str] = []
    for gname in active_groups:
        g = GROUPS.get(gname)
        if g is None:
            continue
        if not g["match_kwargs"].get("exact_rules", False):
            continue
        for k in g["keys"]:
            if k in AGENT_REGISTRY and k not in eligible_keys:
                eligible_keys.append(k)

    out: dict = {}
    for key in eligible_keys:
        print(f"Per-agent exploitability: {key} ...", flush=True)
        agent = build_agent(key)
        agent_block: dict = {}
        if do_lbr:
            t0 = time.time()
            agent_block["lbr"] = lbr_exploitability(
                agent, deals=deals, depth=lbr_depth, seed=seed,
            )
            print(f"  lbr value={agent_block['lbr']['value']:.4f} "
                  f"({time.time() - t0:.1f}s)")
        if do_subgame:
            t0 = time.time()
            agent_block["subgame"] = subgame_exploitability(
                agent, deals=deals, seed=seed,
            )
            print(f"  subgame value={agent_block['subgame']['value']:.4f} "
                  f"({time.time() - t0:.1f}s)")
        out[key] = agent_block
    return out


def _make_run_id(name: str, config: dict) -> str:
    """`YYYYMMDD-<name>-<short_hash>` per design §5."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    short = hashlib.sha1(blob).hexdigest()[:6]
    return f"{datetime.now(UTC).strftime('%Y%m%d')}-{name}-{short}"


def main():
    parser = argparse.ArgumentParser(description="Run agent benchmarks.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--groups", nargs="+", choices=ALL_GROUP_NAMES, default=None,
        metavar="GROUP",
        help=f"Which groups to run (default: all). Choices: {ALL_GROUP_NAMES}",
    )
    parser.add_argument(
        "--log-decisions", action="store_true",
        help="Write per-turn decision records to data/runs/<run_id>/decisions.jsonl.",
    )
    parser.add_argument(
        "--run-name", default="benchmark",
        help="Short name embedded in run_id (default: benchmark).",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Log run config + win-rate matrix + flaw counts to W&B.",
    )
    parser.add_argument(
        "--wandb-entity", default="conair92-university-of-maryland",
        help="W&B entity (default: conair92-university-of-maryland).",
    )
    parser.add_argument(
        "--wandb-project", default="liars-poker",
        help="W&B project (default: liars-poker).",
    )
    parser.add_argument(
        "--exploitability", action="store_true",
        help="Compute Kuhn-oracle exploitability (CFR on the small-game variant) "
             "and emit it under output['oracle_exploitability']. Per-agent "
             "exploitability is deferred to P5 (modular agent interfaces).",
    )
    parser.add_argument(
        "--exploitability-iters", type=int, default=2000,
        help="CFR iterations on the Kuhn-sized oracle (default 2000).",
    )
    parser.add_argument(
        "--lbr", action="store_true",
        help="Compute per-agent LBR exploitability (P5-#2 §2a). Slow on the "
             "full agent zoo; pair with --exploitability-deals to bound cost.",
    )
    parser.add_argument(
        "--subgame", action="store_true",
        help="Compute per-agent sampled-subgame exploitability (P5-#2 §2b).",
    )
    parser.add_argument(
        "--exploitability-deals", type=int, default=50,
        help="Sampled deals per agent for LBR / subgame metrics (default 50; "
             "set 200+ for paper-grade runs).",
    )
    parser.add_argument(
        "--lbr-depth", type=int, default=2,
        help="LBR lookahead depth (default 2 — keeps zoo runs tractable).",
    )
    args = parser.parse_args()

    active = args.groups or ALL_GROUP_NAMES
    config = {"games": args.games, "seed": args.seed, "groups": active}
    run_id = _make_run_id(args.run_name, config)

    print(f"Running benchmark: {args.games} games/pair, seed={args.seed}, groups={active}")
    if args.log_decisions:
        print(f"run_id = {run_id}  (decision log enabled)")

    logger: DecisionLogger | None = None
    decisions_path: str | None = None
    if args.log_decisions:
        decisions_path = os.path.join(
            _REPO_ROOT, "data", "runs", run_id, "decisions.jsonl"
        )
        logger = DecisionLogger(decisions_path)

    try:
        results = run_benchmark(
            n_games=args.games, base_seed=args.seed, group_names=active,
            logger=logger, run_id=run_id,
        )
    finally:
        if logger is not None:
            logger.close()

    # Per-agent exploitability slot — populated in P5 once agents implement the
    # modular policy interface. For now, emit `null` so the metrics.json schema
    # is stable and downstream readers can handle absence uniformly.
    for r in results.values():
        r.setdefault("exploitability_a", None)
        r.setdefault("exploitability_b", None)

    oracle_expl: float | None = None
    if args.exploitability:
        from training.metrics.exploitability import (
            compute_exploitability,
            kuhn_cfr_plus_solve,
        )
        print(f"Computing Kuhn-oracle exploitability ({args.exploitability_iters} CFR iters)...")
        g, pol = kuhn_cfr_plus_solve(iterations=args.exploitability_iters)
        oracle_expl = compute_exploitability(g, pol)
        print(f"  oracle_exploitability = {oracle_expl:.6f}")

    agent_exploitability: dict = {}
    if args.lbr or args.subgame:
        agent_exploitability = _compute_per_agent_exploitability(
            active_groups=[g for g in active if g in GROUPS],
            do_lbr=args.lbr,
            do_subgame=args.subgame,
            deals=args.exploitability_deals,
            lbr_depth=args.lbr_depth,
            seed=args.seed,
        )

    output = {
        "generated": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "games_per_pair": args.games,
        "seed": args.seed,
        "groups": active,
        "results": results,
        "oracle_exploitability": oracle_expl,
        "oracle_exploitability_iters": args.exploitability_iters if args.exploitability else None,
        "agent_exploitability": agent_exploitability,
    }

    if args.log_decisions:
        out_path = os.path.join(_REPO_ROOT, "data", "runs", run_id, "metrics.json")
    else:
        out_path = os.path.join(_REPO_ROOT, "data", "runs", "benchmark", "benchmark_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    if decisions_path:
        print(f"Decisions: {decisions_path}")

    if args.wandb:
        _push_to_wandb(args, run_id, config, output, decisions_path)


def _push_to_wandb(args, run_id: str, config: dict, output: dict,
                   decisions_path: str | None) -> None:
    """Optional: log run summary to W&B. Imports lazily so wandb is not a
    hard dependency for non-tracked runs."""
    import subprocess

    import wandb

    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT
        ).decode().strip()
    except Exception:
        pass

    config_hash = hashlib.sha1(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_id,
        id=run_id,
        config={**config, "git_sha": git_sha, "config_hash": config_hash},
        tags=[f"git:{git_sha}", f"cfg:{config_hash}", "benchmark"],
        reinit=True,
    )

    win_rate_rows = []
    for pair_key, r in output["results"].items():
        win_rate_rows.append([
            r["agent_a"], r["agent_b"], r.get("group", ""),
            r["games"], r["wins_a"], r["wins_b"],
            r["win_rate_a"], r["win_rate_b"],
        ])
    table = wandb.Table(
        columns=["agent_a", "agent_b", "group", "games", "wins_a", "wins_b",
                 "win_rate_a", "win_rate_b"],
        data=win_rate_rows,
    )
    run.log({"win_rate_matrix": table})

    if decisions_path and os.path.exists(decisions_path):
        from training.reflect import (
            _bucket_totals,
            _iter_records,
            rule_infeasible_bid,
            rule_stale_bid_repetition,
        )
        records = list(_iter_records(decisions_path))
        totals = _bucket_totals(records)
        flaws = {
            "infeasible_bid": sum(rule_infeasible_bid(records).values()),
            "stale_bid_repetition": sum(rule_stale_bid_repetition(records).values()),
            "total_decisions": len(records),
            "total_buckets": len(totals),
        }
        run.log(flaws)
        run.summary.update(flaws)

    run.finish()
    print(f"Logged to W&B: {args.wandb_entity}/{args.wandb_project}/{run_id}")


if __name__ == "__main__":
    main()
