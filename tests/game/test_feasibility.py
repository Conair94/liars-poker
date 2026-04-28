"""Regression tests for the feasibility helper extracted from registry.py."""

from __future__ import annotations

import numpy as np

from agents.registry import _is_bid_feasible
from game.bids import NUM_ACTIONS, NUM_BIDS
from game.feasibility import (
    feasible_action_mask,
    feasible_bid_mask,
    is_bid_feasible,
)


def test_is_bid_feasible_agrees_with_registry():
    """Cross-check the new public helper against the legacy private one."""
    for n in range(0, 26):
        for a in range(NUM_ACTIONS):
            assert is_bid_feasible(a, n) == _is_bid_feasible(a, n), (a, n)


def test_call_and_hh_always_feasible():
    for n in range(0, 26):
        assert is_bid_feasible(NUM_BIDS, n)      # CALL
        assert is_bid_feasible(NUM_BIDS + 1, n)  # HH


def test_feasible_action_mask_matches_scalar():
    for n in range(0, 26):
        mask = feasible_action_mask(n)
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.dtype == np.bool_
        for a in range(NUM_ACTIONS):
            assert bool(mask[a]) == is_bid_feasible(a, n), (a, n)


def test_feasible_bid_mask_is_action_mask_prefix():
    for n in range(0, 26):
        bid_mask = feasible_bid_mask(n)
        act_mask = feasible_action_mask(n)
        assert bid_mask.shape == (NUM_BIDS,)
        np.testing.assert_array_equal(bid_mask, act_mask[:NUM_BIDS])


def test_low_n_filters_high_arity_hands():
    # n=2 → only HC (1 card) and Pair (2 cards) are feasible.
    mask = feasible_bid_mask(2)
    # HC bids occupy indices 0..12, Pairs 13..25.
    assert mask[:13].all()      # HC 2..A
    assert mask[13:26].all()    # Pairs
    # Two-pair (needs 4) onward should be False.
    assert not mask[26:].any()
