"""
compute_five_kings_probs.py — MC probability tables for 5-Kings easter egg mode.

When the "King of Beers" toggle is active the deck has 53 cards (standard 52 +
a 5th King, card index 52).  Five-of-a-Kind Kings (ht=9, pr=11) is a new hand
ranked above Straight Flush.  Standard MARGINAL_PAL is calibrated for a 52-card
deck and will be slightly off.

This script re-runs the marginal MC with the 53-card deck and saves a separate
probability table that agents should load in five-kings mode.

Output: agent/data/five_kings_probs.json
  {
    "n_samples": 1000000,
    "deck_size": 53,
    "5":  {"exact": [float×111], "at_least": [float×111]},
    ...
    "25": {...}
  }

The bid space gains one extra entry at index 110: Five of a Kind Kings (ht=9,pr=11).
Indices 0..109 are unchanged (same as standard ALL_BIDS ordering).

Usage:
  cd "papers/Liars poker/"
  python -m agent.data.compute_five_kings_probs
  python -m agent.data.compute_five_kings_probs --samples 2000000 --workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from typing import List, Tuple

HERE      = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
if PAPER_DIR not in sys.path:
    sys.path.insert(0, PAPER_DIR)

from poker_math_exact import _evaluate_ranked, ROYAL_FLUSH, STRAIGHT_FLUSH  # noqa: E402

DATA_DIR    = HERE
OUTPUT_FILE = os.path.join(DATA_DIR, "five_kings_probs.json")

N_VALUES = list(range(5, 26))
DEFAULT_SAMPLES = 1_000_000
BASE_SEED = 77

# ---------------------------------------------------------------------------
# Bid list — standard 110 bids + Five of a Kind Kings at index 110
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
_ALL_BIDS.append((9, 11))   # Five of a Kind Kings
NUM_BIDS_5K = len(_ALL_BIDS)   # 111

# Card 52 = King of Beers (rank=11/K, no suit)
FIVE_KINGS_CARD = 52


def _evaluate_five_kings(pool: List[int]) -> Tuple[int, int]:
    """Evaluate best hand, treating card 52 as an extra King (rank=11)."""
    # Count rank 11 (King) including card 52
    king_count = sum(1 for c in pool if (c != FIVE_KINGS_CARD and c >> 2 == 11)) + \
                 sum(1 for c in pool if c == FIVE_KINGS_CARD)
    if king_count >= 5:
        return (9, 11)  # Five of a Kind Kings

    # Remap card 52 to a regular King (any suit; use suit=0 for standard eval)
    remapped = [c if c != FIVE_KINGS_CARD else 44 for c in pool]  # 44 = K♣
    raw_t, raw_p = _evaluate_ranked(remapped)
    if raw_t == ROYAL_FLUSH:
        raw_t, raw_p = STRAIGHT_FLUSH, 12
    return (raw_t, raw_p)


def _simulate_n(args: Tuple) -> dict:
    n, n_samples, seed = args
    rng = random.Random(seed)
    deck = list(range(52)) + [FIVE_KINGS_CARD]  # 53-card deck

    counts = [0] * NUM_BIDS_5K
    t0 = time.time()

    for _ in range(n_samples):
        rng.shuffle(deck)
        pool = deck[:n]
        ht, pr = _evaluate_five_kings(pool)
        bid_idx = next(
            (i for i, (bht, bpr) in enumerate(_ALL_BIDS) if bht == ht and bpr == pr),
            None
        )
        if bid_idx is not None:
            counts[bid_idx] += 1

    elapsed = time.time() - t0
    exact_prob = [c / n_samples for c in counts]

    at_least = [0.0] * NUM_BIDS_5K
    running = 0.0
    for i in range(NUM_BIDS_5K - 1, -1, -1):
        running += exact_prob[i]
        at_least[i] = running

    print(f"  n={n}: {n_samples} samples in {elapsed:.1f}s", flush=True)
    return {"exact": exact_prob, "at_least": at_least}


def run(n_samples: int = DEFAULT_SAMPLES, workers: int = 1, force: bool = False) -> None:
    existing: dict = {}
    if os.path.exists(OUTPUT_FILE) and not force:
        with open(OUTPUT_FILE) as f:
            existing = json.load(f)
        print(f"[five_kings_probs] Loaded cache from {OUTPUT_FILE}")

    todo = [n for n in N_VALUES if str(n) not in existing or force]
    if not todo:
        print("[five_kings_probs] All n-values cached. Use --force to recompute.")
        return

    print(f"[five_kings_probs] Computing n={todo}, {n_samples} samples, {workers} worker(s)...")

    args_list = [(n, n_samples, BASE_SEED + n * 1000) for n in todo]
    if workers > 1:
        with mp.Pool(workers) as pool:
            results = pool.map(_simulate_n, args_list)
    else:
        results = [_simulate_n(a) for a in args_list]

    for n, result in zip(todo, results):
        existing[str(n)] = result

    existing["n_samples"] = n_samples
    existing["deck_size"] = 53
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[five_kings_probs] Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute 5-Kings mode hand probabilities")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(f"Would compute n={N_VALUES} × {args.samples} samples × {args.workers} workers")
    else:
        run(args.samples, args.workers, args.force)
