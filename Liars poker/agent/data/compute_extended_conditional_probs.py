"""
compute_extended_conditional_probs.py  (Stage 2 warm-start data)

Generates rank-specific conditional pool-hand distributions for use as
deterministic warm-start features by the RL agent.

Stage 1's compute_conditional_probs.py aggregates by condition TYPE only
("pair", "trips", "suited", ...). For the Stage 2 agent we need to condition
on the specific rank / rank-pattern, because "pair of Aces" and "pair of 2s"
induce meaningfully different posterior distributions over the pool.

Extended conditions (see AGENT_DESIGN.md §4.3):
  pair_{rank}          — 13 conditions  (pair of 2s..As)
  trips_{rank}         — 13 conditions  (trips of 2s..As)
  suited_high_{rank}   — 13 conditions  (2 same-suit cards, high card = rank)
  3suited_high_{rank}  — 13 conditions  (3 same-suit cards, high card = rank)
  adjacent_low_{rank}  — 12 conditions  (2 consecutive-rank cards, low = rank)
  3range_low_{rank}    — 10 conditions  (3 cards within 5-rank window,
                                         low = rank; 5-rank window requires
                                         low ∈ 0..8 and high ≤ low+4)

Total: 74 conditions × 21 pool sizes (n=5..25).

Output:
  ../data/extended_conditional_probs.json   (type-level: 10 counts per n)
  ../data/extended_conditional_probs_ranked.json  (rank-level: 10×13 per n)

The script is cache-aware: missing conditions are appended; existing ones
are skipped. To force a rerun for a specific condition, delete its entry.

Parallelism:
  Conditions are independent and run in a multiprocessing Pool.
  Each worker uses its own RNG seeded deterministically from (base_seed, key).

Performance notes:
  - The inner loop mirrors compute_conditional_probs._simulate: draw one set
    of opponents per trial (size = 25 - n_private) and slice for each n.
    This shares one draw across all 21 pool sizes (~21× amortization).
  - Default N_SAMPLES = 1_000_000 per condition. At ~500k hand-evals/sec
    single-threaded, each condition costs ~42s × 21 = ~15 min; 74 conditions
    → ~18.5 hours serial, ~2.5 hours on 8 cores. Adjust N_SAMPLES and workers
    via CLI.

Usage:
  cd "papers/Liars poker/"
  python -m agent.data.compute_extended_conditional_probs
  python -m agent.data.compute_extended_conditional_probs --samples 2000000 --workers 8
  python -m agent.data.compute_extended_conditional_probs --dry-run
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Callable

# Make poker_math_exact importable whether this is run as a module or a file.
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
if PAPER_DIR not in sys.path:
    sys.path.insert(0, PAPER_DIR)

from poker_math_exact import _evaluate, _evaluate_ranked, HAND_NAMES  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_VALUES = list(range(5, 26))                  # pool sizes
MAX_N = N_VALUES[-1]                           # 25
DEFAULT_SAMPLES = 1_000_000
BASE_SEED = 42

RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

DATA_CACHE = os.path.join(HERE, "extended_conditional_probs.json")
RANKED_CACHE = os.path.join(HERE, "extended_conditional_probs_ranked.json")


# ---------------------------------------------------------------------------
# Private-hand samplers (all parameterised by rank)
# ---------------------------------------------------------------------------
#
# Each sampler takes an rng and returns a list of card indices
# (card = rank*4 + suit). By construction the sampled cards satisfy the
# condition named in the key.


def _make_pair_sampler(rank: int) -> Callable:
    def sample(rng: random.Random):
        suits = rng.sample(range(4), 2)
        return [rank * 4 + suits[0], rank * 4 + suits[1]]
    return sample


def _make_trips_sampler(rank: int) -> Callable:
    def sample(rng: random.Random):
        suits = rng.sample(range(4), 3)
        return [rank * 4 + s for s in suits]
    return sample


def _make_suited_high_sampler(high_rank: int) -> Callable:
    """
    Two same-suit cards whose highest rank is exactly `high_rank`.
    The low card is chosen uniformly from ranks strictly below high_rank.
    Requires high_rank >= 1.
    """
    def sample(rng: random.Random):
        suit = rng.randint(0, 3)
        low = rng.randint(0, high_rank - 1)
        return [low * 4 + suit, high_rank * 4 + suit]
    return sample


def _make_3suited_high_sampler(high_rank: int) -> Callable:
    """
    Three same-suit cards whose highest rank is exactly `high_rank`.
    Requires high_rank >= 2.
    """
    def sample(rng: random.Random):
        suit = rng.randint(0, 3)
        low_two = rng.sample(range(high_rank), 2)
        return [low_two[0] * 4 + suit, low_two[1] * 4 + suit, high_rank * 4 + suit]
    return sample


def _make_adjacent_low_sampler(low_rank: int) -> Callable:
    """
    Two consecutive-rank cards (low_rank, low_rank+1), any suits.
    Requires low_rank in 0..11.
    """
    def sample(rng: random.Random):
        s1 = rng.randint(0, 3)
        s2 = rng.randint(0, 3)
        return [low_rank * 4 + s1, (low_rank + 1) * 4 + s2]
    return sample


def _make_3range_low_sampler(low_rank: int) -> Callable:
    """
    Three cards within a 5-rank window whose LOW rank is exactly `low_rank`.
    The other two ranks are drawn uniformly from (low_rank, low_rank+4] without
    replacement. Requires low_rank in 0..8 (so low+4 ≤ 12).
    """
    def sample(rng: random.Random):
        others = rng.sample(range(low_rank + 1, low_rank + 5), 2)
        ranks = [low_rank, others[0], others[1]]
        return [r * 4 + rng.randint(0, 3) for r in ranks]
    return sample


# ---------------------------------------------------------------------------
# Condition manifest
# ---------------------------------------------------------------------------


def _build_conditions():
    """
    Returns a list of (key, sampler_factory_args, n_private) tuples.
    We pass factory args (not the sampler itself) so the manifest is
    picklable for multiprocessing.
    """
    conditions = []

    # pair_{rank}  — 13 conditions, 2 private cards
    for r in range(13):
        conditions.append((f"pair_{RANK_NAMES[r]}", ("pair", r), 2))

    # trips_{rank}  — 13 conditions, 3 private cards
    for r in range(13):
        conditions.append((f"trips_{RANK_NAMES[r]}", ("trips", r), 3))

    # suited_high_{rank}  — ranks 1..12 (high must have something below it)
    for r in range(1, 13):
        conditions.append((f"suited_high_{RANK_NAMES[r]}", ("suited_high", r), 2))

    # 3suited_high_{rank}  — ranks 2..12 (need two ranks strictly below)
    for r in range(2, 13):
        conditions.append((f"3suited_high_{RANK_NAMES[r]}", ("3suited_high", r), 3))

    # adjacent_low_{rank}  — ranks 0..11
    for r in range(12):
        conditions.append((f"adjacent_low_{RANK_NAMES[r]}", ("adjacent_low", r), 2))

    # 3range_low_{rank}  — ranks 0..8
    for r in range(9):
        conditions.append((f"3range_low_{RANK_NAMES[r]}", ("3range_low", r), 3))

    return conditions


_FACTORIES = {
    "pair":          _make_pair_sampler,
    "trips":         _make_trips_sampler,
    "suited_high":   _make_suited_high_sampler,
    "3suited_high":  _make_3suited_high_sampler,
    "adjacent_low":  _make_adjacent_low_sampler,
    "3range_low":    _make_3range_low_sampler,
}


def _materialise_sampler(factory_args):
    name, rank = factory_args
    return _FACTORIES[name](rank)


# ---------------------------------------------------------------------------
# Inner simulation loops
# ---------------------------------------------------------------------------


def _simulate_type(sampler, n_private, n_samples, seed):
    """
    Share one opponents draw of size (MAX_N - n_private) across all n values.
    Returns {str(n): [10 counts]} for n in N_VALUES.
    """
    rng = random.Random(seed)
    deck = list(range(52))
    counts = {str(n): [0] * 10 for n in N_VALUES}
    max_opp = MAX_N - n_private

    for _ in range(n_samples):
        private = sampler(rng)
        priv_set = set(private)
        rest = [c for c in deck if c not in priv_set]
        opponents = rng.sample(rest, max_opp)

        for n in N_VALUES:
            pool = private + opponents[: n - n_private]
            counts[str(n)][_evaluate(pool)] += 1

    return counts


def _simulate_ranked(sampler, n_private, n_samples, seed):
    """
    Like _simulate_type but tracks (hand_type, primary_rank) via _evaluate_ranked.
    Returns {str(n): [[0]*13 for _ in range(10)]}.
    """
    rng = random.Random(seed)
    deck = list(range(52))
    counts = {str(n): [[0] * 13 for _ in range(10)] for n in N_VALUES}
    max_opp = MAX_N - n_private

    for _ in range(n_samples):
        private = sampler(rng)
        priv_set = set(private)
        rest = [c for c in deck if c not in priv_set]
        opponents = rng.sample(rest, max_opp)

        for n in N_VALUES:
            pool = private + opponents[: n - n_private]
            t, r = _evaluate_ranked(pool)
            counts[str(n)][t][r] += 1

    return counts


# ---------------------------------------------------------------------------
# Worker entrypoint (top-level for pickling)
# ---------------------------------------------------------------------------


def _worker(job):
    key, factory_args, n_priv, n_samples, seed = job
    sampler = _materialise_sampler(factory_args)

    t0 = time.time()
    type_counts = _simulate_type(sampler, n_priv, n_samples, seed)
    ranked_counts = _simulate_ranked(sampler, n_priv, n_samples, seed + 1)
    dt = time.time() - t0

    return key, type_counts, ranked_counts, dt


# ---------------------------------------------------------------------------
# Cache load / save
# ---------------------------------------------------------------------------


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"n_samples": None, "conditions": {}}


def _save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                   help=f"Monte Carlo samples per condition (default {DEFAULT_SAMPLES}).")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="Parallel worker processes (default: CPU-1).")
    p.add_argument("--dry-run", action="store_true",
                   help="List pending conditions and exit without computing.")
    p.add_argument("--force", action="store_true",
                   help="Ignore existing cache and recompute all conditions.")
    args = p.parse_args()

    conditions = _build_conditions()
    print(f"[config] {len(conditions)} conditions, {args.samples:,} samples each, "
          f"{args.workers} workers")

    os.makedirs(HERE, exist_ok=True)
    type_data = {"n_samples": args.samples, "conditions": {}} if args.force else _load(DATA_CACHE)
    ranked_data = {"n_samples": args.samples, "conditions": {}} if args.force else _load(RANKED_CACHE)
    # Ensure n_samples is always set (fixes load-from-cache case where it was None).
    type_data["n_samples"] = args.samples
    ranked_data["n_samples"] = args.samples
    type_data.setdefault("n_samples", args.samples)
    ranked_data.setdefault("n_samples", args.samples)
    type_data.setdefault("conditions", {})
    ranked_data.setdefault("conditions", {})

    existing = set(type_data["conditions"].keys()) & set(ranked_data["conditions"].keys())
    pending = [(k, fa, np_) for (k, fa, np_) in conditions if k not in existing]

    print(f"[cache] {len(existing)} already computed, {len(pending)} pending")
    if not pending:
        print("[done] nothing to do.")
        return

    if args.dry_run:
        print("[dry-run] pending conditions:")
        for k, _, _ in pending:
            print(f"  - {k}")
        return

    # Deterministic per-condition seeds derived from key hash.
    jobs = []
    for i, (k, fa, np_) in enumerate(pending):
        seed = BASE_SEED + (hash(k) & 0xFFFF) + i * 2
        jobs.append((k, fa, np_, args.samples, seed))

    t_start = time.time()
    completed = 0

    def _finalize(result):
        nonlocal completed
        key, type_counts, ranked_counts, dt = result
        type_data["conditions"][key] = type_counts
        ranked_data["conditions"][key] = ranked_counts
        completed += 1
        print(f"[{completed}/{len(pending)}] {key}  ({dt:.1f}s)")
        # Incremental save so a crash doesn't lose work.
        _save(DATA_CACHE, type_data)
        _save(RANKED_CACHE, ranked_data)

    if args.workers <= 1:
        for job in jobs:
            _finalize(_worker(job))
    else:
        with mp.Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(_worker, jobs):
                _finalize(result)

    total = time.time() - t_start
    print(f"[done] {completed} conditions in {total/60:.1f} min")
    print(f"[done] wrote {DATA_CACHE}")
    print(f"[done] wrote {RANKED_CACHE}")


if __name__ == "__main__":
    main()
