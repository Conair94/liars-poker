"""Stratified deal sampler (P5-#2 Phase A, S3).

Samples (P0_hand, P1_hand) pairs from a 52-card deck without replacement.
Both LBR and the sampled-subgame metric run their per-deal computation on
top of these draws; stratifying by P0 hand strength gives a more honest
exploitability estimate on a finite budget than pure i.i.d. sampling.

Stratification scheme
---------------------
Three coarse buckets by P0's 5-card hand_type:
  - "weak"   : HC                          (~50% of random 5-card hands)
  - "mid"    : Pair, Two Pair              (~47%)
  - "strong" : Three of a Kind through SF  (~3%)

Equal quota per bucket by default. Rarer fine-grained types (SF, 4K) are
not separately stratified — naive rejection sampling for them would burn
millions of draws to fill a quota. The exploitability metric is robust
to this; the fine breakdown is reported in the per-deal output so callers
can post-hoc reweight if they need a marginal-distribution-matched estimate.

Public API
----------
    sample_deals(n, seed, hand_size=5, num_players=2, stratified=True)
        Iterator yielding tuples of `num_players` hands; each hand is a
        sorted list[int] of card indices in the project encoding (0..51).
"""

from __future__ import annotations

import os
import random
import sys
from collections.abc import Iterator

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_PROBS = os.path.join(_SRC, "training", "probs")
if _PROBS not in sys.path:
    sys.path.insert(0, _PROBS)

from game.bids import (  # noqa: E402
    FLUSH,
    FOUR_OF_A_KIND,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    STRAIGHT,
    STRAIGHT_FLUSH,
    THREE_OF_A_KIND,
    TWO_PAIR,
    normalize_hand_type,
)
from poker_math_exact import _evaluate_ranked  # noqa: E402

_BUCKET_ORDER = ("weak", "mid", "strong")
_BUCKETS = {
    HIGH_CARD:       "weak",
    PAIR:            "mid",
    TWO_PAIR:        "mid",
    THREE_OF_A_KIND: "strong",
    STRAIGHT:        "strong",
    FLUSH:           "strong",
    FULL_HOUSE:      "strong",
    FOUR_OF_A_KIND:  "strong",
    STRAIGHT_FLUSH:  "strong",
}

# Safety cap on draws per bucket-fill attempt; a 5-card SF is ~1/65000 so
# the strong bucket's tail is the binding constraint.
_MAX_ATTEMPTS_MULTIPLIER = 200


def _hand_bucket(hand: list[int]) -> str:
    raw_t, raw_p = _evaluate_ranked(hand)
    t, _ = normalize_hand_type(raw_t, raw_p)
    return _BUCKETS.get(t, "weak")


def _draw_deal(rng: random.Random, hand_size: int, num_players: int) -> list[list[int]]:
    deck = list(range(52))
    rng.shuffle(deck)
    hands = []
    for p in range(num_players):
        start = p * hand_size
        hands.append(sorted(deck[start:start + hand_size]))
    return hands


def sample_deals(
    n: int,
    seed: int,
    hand_size: int = 5,
    num_players: int = 2,
    stratified: bool = True,
) -> Iterator[tuple[list[int], ...]]:
    """Yield up to `n` deals as tuples of per-player sorted card-index lists.

    With `stratified=True` (default), targets equal counts per coarse bucket
    measured on player 0's hand. Falls back to i.i.d. continuation once a
    safety cap is reached (rare strong hands won't always fill their quota
    on small `n`).

    Deterministic in `seed`.
    """
    if n <= 0:
        return
    if num_players * hand_size > 52:
        raise ValueError("hand size × players exceeds 52-card deck")

    rng = random.Random(seed)

    if not stratified:
        for _ in range(n):
            yield tuple(_draw_deal(rng, hand_size, num_players))
        return

    # Targets: even split across buckets, with leftovers going to "weak"
    # (most common, easiest to fill).
    base = n // len(_BUCKET_ORDER)
    targets = {b: base for b in _BUCKET_ORDER}
    targets["weak"] += n - base * len(_BUCKET_ORDER)
    counts = {b: 0 for b in _BUCKET_ORDER}

    yielded = 0
    attempts = 0
    cap = max(n, 1) * _MAX_ATTEMPTS_MULTIPLIER

    while yielded < n and attempts < cap:
        attempts += 1
        deal = _draw_deal(rng, hand_size, num_players)
        bucket = _hand_bucket(deal[0])
        if counts[bucket] >= targets[bucket]:
            continue
        counts[bucket] += 1
        yielded += 1
        yield tuple(deal)

    # Cap reached with quotas unfilled — top up with i.i.d. draws so the
    # caller always gets `n` deals (the alternative — yielding fewer — would
    # surprise consumers and complicate budget accounting).
    while yielded < n:
        yielded += 1
        yield tuple(_draw_deal(rng, hand_size, num_players))
