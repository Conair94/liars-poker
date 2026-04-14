"""
compute_exact_rules_probs.py — MC probability tables for Exact Hand Rules mode.

In Exact Hand Rules mode a bid holds only if the pool contains a 5-card subset
whose best hand is *exactly* the bid (not better, not worse).  The standard
MARGINAL_PAL encodes P(pool_best >= bid), which over-estimates validity for
high bids — an agent using it in exact mode will be overconfident.

This script computes two per-n tables:

  exact_prob[n][bid_i]    = P(pool contains 5-card subset with best hand == bid_i)
  exact_atleast[n][bid_i] = P(pool contains 5-card subset with best hand >= bid_i)
                          = sum(exact_prob[n][j] for j >= bid_i)

Algorithm: for each sampled pool, enumerate ALL C(n,5) subsets once, evaluate
each, and collect the set of (ht, pr) hands present.  This is O(C(n,5)) per
sample — 110× faster than checking each bid independently.

Output: agent/data/exact_rules_probs.json
  {
    "n_samples": 10000,
    "5": {"exact": [float×110], "at_least": [float×110]},
    ...
    "25": {...}
  }

Runtime estimates (single core, ~500k hand-evals/sec):
  n=5  : C(5,5)=1       → 10k × 1 = 10k evals  ≈ 0.02s
  n=10 : C(10,5)=252     → 10k × 252 = 2.5M evals ≈ 5s
  n=15 : C(15,5)=3003    → 10k × 3k = 30M evals  ≈ 60s
  n=20 : C(20,5)=15504   → 10k × 15k = 155M evals ≈ 310s
  n=25 : C(25,5)=53130   → 10k × 53k = 531M evals ≈ 1062s
  Total single-core (all 21 n): ~2500s ≈ 42 min
  With --workers 8: ~5–6 minutes

Cache-aware: if the file already exists, existing n-values are skipped.
Set --force to recompute everything.

Usage:
  cd "papers/Liars poker/"
  python -m agent.data.compute_exact_rules_probs
  python -m agent.data.compute_exact_rules_probs --samples 50000 --workers 8
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
from typing import List, Set, Tuple

HERE      = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
if PAPER_DIR not in sys.path:
    sys.path.insert(0, PAPER_DIR)

from poker_math_exact import _evaluate_ranked, ROYAL_FLUSH, STRAIGHT_FLUSH  # noqa: E402

DATA_DIR    = HERE
OUTPUT_FILE = os.path.join(DATA_DIR, "exact_rules_probs.json")

N_VALUES = list(range(5, 26))
DEFAULT_SAMPLES = 10_000   # quick default; use --samples 100000 for production
BASE_SEED = 99


# ---------------------------------------------------------------------------
# Bid list (must match ALL_BIDS order in bids.py / index.html)
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
NUM_BIDS = len(_ALL_BIDS)  # 110

_BID_TO_INDEX = {b: i for i, b in enumerate(_ALL_BIDS)}


def _exact_bids_in_pool(pool: List[int]) -> Set[Tuple[int, int]]:
    """
    Return the set of (ht, pr) bids that are EXACTLY present in the pool
    (i.e. there exists a 5-card subset whose best hand is exactly (ht, pr)).

    Uses a single O(C(n,5)) pass — much faster than checking each bid
    independently (which would be O(NUM_BIDS × C(n,5))).
    """
    n = len(pool)
    present: Set[Tuple[int, int]] = set()
    if n < 5:
        raw_t, raw_p = _evaluate_ranked(pool)
        if raw_t == ROYAL_FLUSH:
            raw_t, raw_p = STRAIGHT_FLUSH, 12
        present.add((raw_t, raw_p))
        return present
    for combo in combinations(pool, 5):
        raw_t, raw_p = _evaluate_ranked(list(combo))
        if raw_t == ROYAL_FLUSH:
            raw_t, raw_p = STRAIGHT_FLUSH, 12
        present.add((raw_t, raw_p))
    return present


def _simulate_n(args: Tuple) -> dict:
    """Worker: simulate exact-hand probabilities for one pool size n."""
    n, n_samples, seed = args
    rng = random.Random(seed)
    deck = list(range(52))

    # counts[bid_i] = number of samples where pool contains exact bid_i
    counts = [0] * NUM_BIDS

    t0 = time.time()
    for _ in range(n_samples):
        rng.shuffle(deck)
        pool = deck[:n]
        present = _exact_bids_in_pool(pool)
        for (ht, pr) in present:
            idx = _BID_TO_INDEX.get((ht, pr))
            if idx is not None:
                counts[idx] += 1

    elapsed = time.time() - t0
    exact_prob = [c / n_samples for c in counts]

    # at_least[i] = P(pool contains any exact subset >= bid_i)
    # = cumsum from top; note this is NOT the same as standard "at_least"
    # since multiple bids can simultaneously be present in the same pool.
    at_least = [0.0] * NUM_BIDS
    running = 0.0
    for i in range(NUM_BIDS - 1, -1, -1):
        running += exact_prob[i]
        at_least[i] = min(running, 1.0)  # cap at 1 (multiple bids can coexist)

    print(f"  n={n}: {n_samples} samples in {elapsed:.1f}s", flush=True)
    return {"exact": exact_prob, "at_least": at_least}


def run(n_samples: int = DEFAULT_SAMPLES, workers: int = 1, force: bool = False) -> None:
    # Load existing cache
    existing: dict = {}
    if os.path.exists(OUTPUT_FILE) and not force:
        with open(OUTPUT_FILE) as f:
            existing = json.load(f)
        print(f"[exact_rules_probs] Loaded cache with {len([k for k in existing if k.isdigit()])} n-values")

    todo = [n for n in N_VALUES if str(n) not in existing or force]
    if not todo:
        print("[exact_rules_probs] All n-values cached. Use --force to recompute.")
        return

    print(f"[exact_rules_probs] Computing n={todo}, {n_samples} samples each, {workers} worker(s)...")
    t_total = time.time()

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
    print(f"[exact_rules_probs] Done in {time.time()-t_total:.1f}s → {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute exact-rules hand probabilities")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"MC samples per n (default {DEFAULT_SAMPLES}; use 100000+ for production)")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--force", action="store_true", help="Recompute all n-values")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    args = parser.parse_args()
    if args.dry_run:
        print(f"Plan: n={N_VALUES}, {args.samples} samples, {args.workers} workers")
        from math import comb
        total = sum(comb(n, 5) * args.samples for n in N_VALUES)
        print(f"Estimated hand evals: {total/1e6:.1f}M  (~{total/500000:.0f}s single-core)")
    else:
        run(args.samples, args.workers, args.force)
