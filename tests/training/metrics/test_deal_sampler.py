"""Tests for the stratified deal sampler (P5-#2 Phase A, S3)."""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_PROBS = os.path.join(_SRC, "training", "probs")
if _PROBS not in sys.path:
    sys.path.insert(0, _PROBS)

from training.metrics.deal_sampler import (  # noqa: E402
    _BUCKET_ORDER,
    _hand_bucket,
    sample_deals,
)


def test_sample_yields_n_deals():
    deals = list(sample_deals(n=30, seed=0, stratified=False))
    assert len(deals) == 30


def test_deals_are_disjoint_per_deal():
    """Each deal's per-player hands must use distinct cards."""
    for hands in sample_deals(n=20, seed=1):
        all_cards = [c for h in hands for c in h]
        assert len(all_cards) == len(set(all_cards))
        assert all(0 <= c < 52 for c in all_cards)


def test_hand_size_and_player_count():
    deals = list(sample_deals(n=5, seed=2, hand_size=3, num_players=3))
    assert len(deals) == 5
    for hands in deals:
        assert len(hands) == 3
        for h in hands:
            assert len(h) == 3


def test_deterministic_seed():
    a = list(sample_deals(n=10, seed=42))
    b = list(sample_deals(n=10, seed=42))
    assert a == b


def test_seeds_differ():
    a = list(sample_deals(n=10, seed=0))
    b = list(sample_deals(n=10, seed=1))
    assert a != b


def test_stratified_balances_buckets_for_large_n():
    """With enough budget, weak/mid/strong buckets each see ~n/3 deals."""
    n = 90
    deals = list(sample_deals(n=n, seed=7, stratified=True))
    assert len(deals) == n
    counts = {b: 0 for b in _BUCKET_ORDER}
    for hands in deals:
        counts[_hand_bucket(list(hands[0]))] += 1
    target = n // 3
    # Each quota is exactly `target` (modulo the +leftovers bumping "weak").
    assert counts["mid"] == target
    # "strong" may underfill if cap hit + i.i.d. top-up. With n=90 and a
    # 200x attempts cap (=18000 draws) the strong bucket reliably fills.
    assert counts["strong"] == target


def test_iid_mode_does_not_fill_quotas():
    """With stratified=False the bucket distribution should follow the
    natural marginal — mostly weak (HC ~50%) and mid (~47%), barely any strong."""
    n = 200
    deals = list(sample_deals(n=n, seed=9, stratified=False))
    counts = {b: 0 for b in _BUCKET_ORDER}
    for hands in deals:
        counts[_hand_bucket(list(hands[0]))] += 1
    # Sanity: HC + Pair + TwoPair dominate.
    assert counts["weak"] + counts["mid"] >= int(0.85 * n)
