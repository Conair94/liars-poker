"""
Overnight driver for the 1v1 n=2 exact-rules CFR solver.

Runs CFRSolver.run() in small batches, checkpointing solver state and per-
batch metrics to disk. Resumes from the last checkpoint on restart. Designed
to be safe to Ctrl-C or crash at any time without losing progress.

Output files (under agent/data/cfr_1v1_run/<run_name>/):
    checkpoint.json   solver state (regret_sum, strategy_sum, iterations, knobs)
    metrics.jsonl     one JSON object per batch: iter, exp, game_value, t_s
    summary.json      most recent summary (opening_mix, response_mix, etc.)
    config.json       run configuration (max_bids, bid_space, include_hh, ...)

Usage
-----
    # Start / resume a run named "mb4_hh" with max_bids=4 and HH enabled:
    python -m agent.baseline.cfr_1v1_overnight \
        --name mb4_hh --max-bids 4 --batch 50 --total-iters 5000

    # Short smoke test:
    python -m agent.baseline.cfr_1v1_overnight \
        --name smoke --max-bids 3 --batch 10 --total-iters 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

_BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR    = os.path.abspath(os.path.join(_BASELINE_DIR, ".."))
_PAPER_DIR    = os.path.abspath(os.path.join(_AGENT_DIR,    ".."))

for _p in (_PAPER_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.baseline.cfr_1v1 import (   # noqa: E402
    CFRSolver, HC_PAIR_BIDS, _all_bid_indices, _solver_summary, _cache_key,
)
from agent.baseline.cfr_1v1_fast import CFRSolverFast  # noqa: E402
from agent.game.bids import CALL_ACTION, HH_ACTION, bid_to_index, Bid, HIGH_CARD  # noqa: E402


def _make_solver(kind: str, **kwargs):
    if kind == "fast":
        return CFRSolverFast(**kwargs)
    return CFRSolver(**kwargs)


def _fast_solver_summary(solver: CFRSolverFast) -> dict:
    """Replicates _solver_summary for CFRSolverFast (different internal API)."""
    mix        = solver.opening_mix_by_rank()
    game_value = solver.game_value()
    exp        = solver.exploitability()
    return {
        "iterations":     solver._iterations,
        "max_bids":       solver.max_bids,
        "bid_space_size": len(solver.bid_space),
        "include_hh":     solver.include_hh,
        "exploitability": exp,
        "game_value":     game_value,
        "opening_mix":    {str(r): mix[r] for r in range(13)},
    }


def _run_dir(run_name: str) -> str:
    return os.path.join(_AGENT_DIR, "data", "cfr_1v1_run", run_name)


def _paths(run_name: str) -> Dict[str, str]:
    d = _run_dir(run_name)
    return {
        "dir":        d,
        "checkpoint": os.path.join(d, "checkpoint.json"),
        "checkpoint_tmp": os.path.join(d, "checkpoint.json.tmp"),
        "metrics":    os.path.join(d, "metrics.jsonl"),
        "summary":    os.path.join(d, "summary.json"),
        "config":     os.path.join(d, "config.json"),
    }


def _save_checkpoint(solver: CFRSolver, paths: Dict[str, str]) -> None:
    """Atomic-write the solver state via tmp+rename so a crash can't corrupt it."""
    os.makedirs(paths["dir"], exist_ok=True)
    with open(paths["checkpoint_tmp"], "w") as f:
        json.dump(solver.to_dict(), f)
    os.replace(paths["checkpoint_tmp"], paths["checkpoint"])


def _append_metrics(paths: Dict[str, str], row: dict) -> None:
    with open(paths["metrics"], "a") as f:
        f.write(json.dumps(row) + "\n")


def _save_summary(paths: Dict[str, str], summary: dict, solver) -> None:
    # Include response mixes for HC openings at each rank (most informative).
    response_mixes = {}
    for r in range(13):
        bid_idx = bid_to_index(Bid(HIGH_CARD, r))
        response_mixes[f"after_HC_{r}"] = solver.response_mix_by_rank(bid_idx)
    payload = {
        **summary,
        "response_mixes": response_mixes,
    }
    with open(paths["summary"], "w") as f:
        json.dump(payload, f, indent=2)


def _load_or_create_solver(args, paths: Dict[str, str]):
    solver_cls = CFRSolverFast if args.solver == "fast" else CFRSolver
    if os.path.exists(paths["checkpoint"]):
        with open(paths["checkpoint"]) as f:
            state = json.load(f)
        solver = solver_cls.from_dict(state)
        print(f"[resume] loaded checkpoint @ iter {solver._iterations:,}  "
              f"solver={args.solver}  max_bids={solver.max_bids}  "
              f"|bids|={len(solver.bid_space)}  hh={solver.include_hh}")
        return solver

    bid_space = _all_bid_indices() if args.full_bids else HC_PAIR_BIDS
    solver = solver_cls(
        max_bids   = args.max_bids,
        bid_space  = bid_space,
        include_hh = not args.no_hh,
    )
    os.makedirs(paths["dir"], exist_ok=True)
    with open(paths["config"], "w") as f:
        json.dump({
            "run_name":    args.name,
            "solver":      args.solver,
            "max_bids":    args.max_bids,
            "bid_space":   list(solver.bid_space),
            "bid_space_size": len(solver.bid_space),
            "include_hh":  solver.include_hh,
            "full_bids":   args.full_bids,
            "batch":       args.batch,
            "total_iters": args.total_iters,
        }, f, indent=2)
    print(f"[new]    starting run '{args.name}'  max_bids={solver.max_bids}  "
          f"|bids|={len(solver.bid_space)}  hh={solver.include_hh}")
    return solver


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight CFR runner (1v1 n=2 exact).")
    parser.add_argument("--name",        type=str, required=True,
                        help="Run name — used for the output directory.")
    parser.add_argument("--max-bids",    type=int, default=4)
    parser.add_argument("--batch",       type=int, default=50,
                        help="Iterations per batch between checkpoints.")
    parser.add_argument("--total-iters", type=int, default=5000,
                        help="Target total iterations (will resume if already past).")
    parser.add_argument("--no-hh",       action="store_true")
    parser.add_argument("--full-bids",   action="store_true",
                        help="Use all 110 bids (NOT recommended — tree explodes).")
    parser.add_argument("--eval-every",  type=int, default=10,
                        help="Compute exploitability every N batches (default 10).")
    parser.add_argument("--solver",      type=str, default="fast",
                        choices=["fast", "slow"],
                        help="Which solver backend to use (default fast=vectorized).")
    args = parser.parse_args()

    paths = _paths(args.name)
    solver = _load_or_create_solver(args, paths)

    start_iter = solver._iterations
    target     = args.total_iters
    if start_iter >= target:
        print(f"[done]   target {target:,} already reached (iter={start_iter:,}).")
        return

    batch_idx = 0
    while solver._iterations < target:
        remaining = target - solver._iterations
        n = min(args.batch, remaining)
        t0 = time.time()
        for _ in range(n):
            solver.iterate()
        dt = time.time() - t0

        _save_checkpoint(solver, paths)

        do_eval = ((batch_idx % max(1, args.eval_every)) == 0
                   or solver._iterations >= target)
        if do_eval:
            exp = solver.exploitability()
            gv  = solver.game_value()
        else:
            exp = None
            gv  = None

        row = {
            "iter":       solver._iterations,
            "batch":      batch_idx,
            "batch_n":    n,
            "t_batch_s":  round(dt, 3),
            "t_per_iter_s": round(dt / max(1, n), 3),
            "exploitability": exp,
            "game_value": gv,
        }
        _append_metrics(paths, row)

        if do_eval:
            summary = (_fast_solver_summary(solver)
                       if args.solver == "fast" else _solver_summary(solver))
            _save_summary(paths, summary, solver)
            print(f"  iter {solver._iterations:6d}/{target}  "
                  f"batch={n} in {dt:5.1f}s  exp={exp:.4f}  gv={gv:+.4f}")
        else:
            print(f"  iter {solver._iterations:6d}/{target}  "
                  f"batch={n} in {dt:5.1f}s")
        batch_idx += 1

    print(f"[done]   reached {solver._iterations:,} iterations.")


if __name__ == "__main__":
    main()
