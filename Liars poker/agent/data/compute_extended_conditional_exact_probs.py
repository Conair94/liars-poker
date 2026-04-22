"""
compute_extended_conditional_exact_probs.py — exact-rules conditional tables.

Generates per-bid EXACT probabilities conditioned on the private-hand key:

    P(pool contains a 5-card subset with best hand == bid_i | n, condition)

These tables parallel `extended_conditional_probs_ranked.json` (at-least
semantics) but for exact-hand-rules mode, where validity requires a 5-card
subset that equals the bid exactly.

Algorithm per sample:
  1. Draw private cards from the condition's sampler.
  2. Draw opponents (size = MAX_N - n_private) from the remaining deck.
  3. For each n in 5..25, slice pool = private + opponents[:n-n_private]
     and run `_exact_bids_in_pool(pool)` — a set of (ht, pr) that appear.
  4. Increment counts[bid_i] for every present bid.

One opponents draw is shared across all 21 pool sizes (same amortization as
the at-least script).

Output:
  agent/data/extended_conditional_exact_probs.json
  {
    "n_samples": 1000000,
    "conditions": {
      "pair_A": {
        "5":  [float×110],   # per-bid exact prob
        ...
        "25": [float×110],
      },
      ...
    }
  }

Runtime: each sample requires enumerating C(n,5) subsets per n (same as
compute_exact_rules_probs.py). Per condition at N_SAMPLES=10000:
  sum over n of 10k × C(n,5) ≈ 820M evals ≈ 28 min single-core.
  74 conditions × 28 min / 8 cores ≈ 4.3 h. Use --samples 5000 for smoke.

Cache-aware: conditions already present are skipped unless --force.

Usage:
  cd "papers/Liars poker/"
  python -m agent.data.compute_extended_conditional_exact_probs --samples 10000 --workers 8
  python -m agent.data.compute_extended_conditional_exact_probs --dry-run
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
from typing import Callable, List, Set, Tuple

HERE      = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
if PAPER_DIR not in sys.path:
    sys.path.insert(0, PAPER_DIR)

from poker_math_exact import _evaluate_ranked, ROYAL_FLUSH, STRAIGHT_FLUSH  # noqa: E402

# Reuse samplers + condition manifest from the at-least script.
from agent.data.compute_extended_conditional_probs import (  # noqa: E402
    _build_conditions,
    _materialise_sampler,
    N_VALUES,
    MAX_N,
    RANK_NAMES,
)


# ---------------------------------------------------------------------------
# Bid table (same layout as compute_exact_rules_probs.py)
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


DEFAULT_SAMPLES = 10_000
BASE_SEED       = 4242
OUTPUT_FILE     = os.path.join(HERE, "extended_conditional_exact_probs.json")


def _exact_bids_in_pool(pool: List[int]) -> Set[Tuple[int, int]]:
    """Set of (ht, pr) bids that appear as the best hand of SOME 5-card subset."""
    n = len(pool)
    present: Set[Tuple[int, int]] = set()
    if n < 5:
        t, p = _evaluate_ranked(pool)
        if t == ROYAL_FLUSH:
            t, p = STRAIGHT_FLUSH, 12
        present.add((t, p))
        return present
    for combo in combinations(pool, 5):
        t, p = _evaluate_ranked(list(combo))
        if t == ROYAL_FLUSH:
            t, p = STRAIGHT_FLUSH, 12
        present.add((t, p))
    return present


def _simulate_exact(sampler: Callable, n_private: int, n_samples: int, seed: int) -> dict:
    """
    Per-sample: draw private + opponents; for each n in N_VALUES evaluate
    `_exact_bids_in_pool(pool[:n])` and increment each present bid's count.
    Returns {str(n): [prob]×NUM_BIDS}.
    """
    rng  = random.Random(seed)
    deck = list(range(52))
    counts = {str(n): [0] * NUM_BIDS for n in N_VALUES}
    max_opp = MAX_N - n_private

    for _ in range(n_samples):
        private = sampler(rng)
        priv_set = set(private)
        rest = [c for c in deck if c not in priv_set]
        opponents = rng.sample(rest, max_opp)

        for n in N_VALUES:
            pool = private + opponents[: n - n_private]
            present = _exact_bids_in_pool(pool)
            row = counts[str(n)]
            for (ht, pr) in present:
                idx = _BID_TO_INDEX.get((ht, pr))
                if idx is not None:
                    row[idx] += 1

    return {
        str(n): [c / n_samples for c in counts[str(n)]]
        for n in N_VALUES
    }


def _worker(job):
    key, factory_args, n_priv, n_samples, seed = job
    sampler = _materialise_sampler(factory_args)
    t0 = time.time()
    result = _simulate_exact(sampler, n_priv, n_samples, seed)
    dt = time.time() - t0
    return key, result, dt


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _load(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"n_samples": None, "conditions": {}}


def _save(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)  # no indent — these arrays are large
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force",   action="store_true")
    args = p.parse_args()

    conditions = _build_conditions()
    print(f"[config] {len(conditions)} conditions, {args.samples:,} samples each, "
          f"{args.workers} workers")

    os.makedirs(HERE, exist_ok=True)
    data = {"n_samples": args.samples, "conditions": {}} if args.force else _load(OUTPUT_FILE)
    data["n_samples"] = args.samples
    data.setdefault("conditions", {})

    existing = set(data["conditions"].keys())
    pending  = [(k, fa, np_) for (k, fa, np_) in conditions if k not in existing]

    print(f"[cache] {len(existing)} already computed, {len(pending)} pending")
    if not pending:
        print("[done] nothing to do.")
        return

    if args.dry_run:
        print("[dry-run] pending conditions:")
        for k, _, _ in pending:
            print(f"  - {k}")
        return

    jobs = []
    for i, (k, fa, np_) in enumerate(pending):
        seed = BASE_SEED + (hash(k) & 0xFFFF) + i * 2
        jobs.append((k, fa, np_, args.samples, seed))

    t_start   = time.time()
    completed = 0

    def _finalize(result):
        nonlocal completed
        key, per_n, dt = result
        data["conditions"][key] = per_n
        completed += 1
        print(f"[{completed}/{len(pending)}] {key}  ({dt:.1f}s)")
        _save(OUTPUT_FILE, data)

    if args.workers <= 1:
        for job in jobs:
            _finalize(_worker(job))
    else:
        with mp.Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(_worker, jobs):
                _finalize(result)

    total = time.time() - t_start
    print(f"[done] {completed} conditions in {total/60:.1f} min")
    print(f"[done] wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
