"""
compute_exact_rules_probs.py — MC probability tables for Exact Hand Rules mode.

In Exact Hand Rules mode a bid holds only if the pool contains a 5-card subset
whose best hand is *exactly* the bid (not better, not worse).  The standard
MARGINAL_PAL encodes P(pool_best >= bid), which over-estimates validity for
high bids — an agent using it in exact mode will be overconfident.

This script computes two per-n tables:

  exact_prob[n][bid_i]   = P(pool contains 5-card subset with best hand == bid_i)
  exact_atleast[n][bid_i]= P(pool contains 5-card subset with best hand >= bid_i)
                         = sum(exact_prob[n][j] for j >= bid_i)

Output: agent/data/exact_rules_probs.json
  {
    "n_samples": 500000,
    "5": {"exact": [float×110], "at_least": [float×110]},
    ...
    "25": {...}
  }

Usage:
  cd "papers/Liars poker/"
  python -m agent.data.compute_exact_rules_probs
  python -m agent.data.compute_exact_rules_probs --samples 1000000 --workers 8

Cache-aware: if the file already exists, existing n-values are skipped.
Set --force to recompute everything.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from itertools import combinations
from typing import List, Tuple

HERE      = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
if PAPER_DIR not in sys.path:
    sys.path.insert(0, PAPER_DIR)

from poker_math_exact import _evaluate_ranked, ROYAL_FLUSH, STRAIGHT_FLUSH  # noqa: E402

# Agent data dir
DATA_DIR   = HERE
OUTPUT_FILE = os.path.join(DATA_DIR, "exact_rules_probs.json")

N_VALUES = list(range(5, 26))
DEFAULT_SAMPLES = 500_000
BASE_SEED = 99


# ---------------------------------------------------------------------------
# Bid list (must match ALL_BIDS order in bids.py)
# ---------------------------------------------------------------------------

_PRIMARY_RANGES = {
    0: (0, 12), 1: (0, 12), 2: (1, 12), 3: (0, 12),
    4: (3, 12), 5: (0, 12), 6: (0, 12), 7: (0, 12), 8: (3, 12),
}
_ALL_BIDS: List[Tuple[int, int]] = []
for _ht in range(9):
    lo, hi = _PRIMARY_RANGES[_ht]
    for _pr in range(lo, hi + 1):
        _ALL_BIDS.append((_ht, _pr))
NUM_BIDS = len(_ALL_BIDS)


def _evaluate_exact(pool: List[int], ht: int, pr: int) -> bool:
    """True if any 5-card subset of pool has best hand == (ht, pr)."""
    n = len(pool)
    if n < 5:
        raw_t, raw_p = _evaluate_ranked(pool)
        if raw_t == ROYAL_FLUSH:
            raw_t, raw_p = STRAIGHT_FLUSH, 12
        return raw_t == ht and raw_p == pr
    for combo in combinations(pool, 5):
        raw_t, raw_p = _evaluate_ranked(list(combo))
        if raw_t == ROYAL_FLUSH:
            raw_t, raw_p = STRAIGHT_FLUSH, 12
        if raw_t == ht and raw_p == pr:
            return True
    return False


def _simulate_n(args: Tuple) -> dict:
    """Worker: simulate exact-hand probabilities for one pool size n."""
    n, n_samples, seed = args
    rng = random.Random(seed)
    deck = list(range(52))

    # counts[bid_i] = number of samples where pool contains exact 5-card hand == bid_i
    counts = [0] * NUM_BIDS

    t0 = time.time()
    for _ in range(n_samples):
        rng.shuffle(deck)
        pool = deck[:n]
        for i, (ht, pr) in enumerate(_ALL_BIDS):
            if _evaluate_exact(pool, ht, pr):
                counts[i] += 1

    elapsed = time.time() - t0
    exact_prob = [c / n_samples for c in counts]

    # at_least[i] = sum(exact_prob[j] for j >= i)
    at_least = [0.0] * NUM_BIDS
    running = 0.0
    for i in range(NUM_BIDS - 1, -1, -1):
        running += exact_prob[i]
        at_least[i] = running

    print(f"  n={n}: {n_samples} samples in {elapsed:.1f}s", flush=True)
    return {"exact": exact_prob, "at_least": at_least}


def run(n_samples: int = DEFAULT_SAMPLES, workers: int = 1, force: bool = False) -> None:
    # Load existing cache
    existing: dict = {}
    if os.path.exists(OUTPUT_FILE) and not force:
        with open(OUTPUT_FILE) as f:
            existing = json.load(f)
        print(f"[exact_rules_probs] Loaded cache from {OUTPUT_FILE}")

    todo = [n for n in N_VALUES if str(n) not in existing or force]
    if not todo:
        print("[exact_rules_probs] All n-values cached. Use --force to recompute.")
        return

    print(f"[exact_rules_probs] Computing n={todo}, {n_samples} samples each, {workers} worker(s)...")

    args_list = [(n, n_samples, BASE_SEED + n * 1000) for n in todo]

    if workers > 1:
        with mp.Pool(workers) as pool:
            results = pool.map(_simulate_n, args_list)
    else:
        results = [_simulate_n(a) for a in args_list]

    for n, result in zip(todo, results):
        existing[str(n)] = result

    existing["n_samples"] = n_samples
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[exact_rules_probs] Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute exact-rules hand probabilities")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--force", action="store_true", help="Recompute all n-values")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    args = parser.parse_args()
    if args.dry_run:
        print(f"Would compute n={N_VALUES} × {args.samples} samples × {args.workers} workers")
    else:
        run(args.samples, args.workers, args.force)
