"""Bid feasibility helpers.

A bid is *feasible* at a given pool size when the bid's hand type can
physically be assembled from `pool_size` cards. E.g. a Straight requires
five cards; bidding it with a 4-card pool is a guaranteed bluff.

Extracted from a private helper in `agents.registry` so the contracts /
training stack can import without dragging the registry along.
"""

from __future__ import annotations

import numpy as np

from game.bids import (
    FLUSH,
    FOUR_OF_A_KIND,
    FULL_HOUSE,
    HIGH_CARD,
    NUM_ACTIONS,
    NUM_BIDS,
    PAIR,
    STRAIGHT,
    STRAIGHT_FLUSH,
    THREE_OF_A_KIND,
    TWO_PAIR,
    index_to_bid,
)

_MIN_CARDS_FOR_HAND: dict[int, int] = {
    HIGH_CARD:       1,
    PAIR:            2,
    TWO_PAIR:        4,
    THREE_OF_A_KIND: 3,
    STRAIGHT:        5,
    FLUSH:           5,
    FULL_HOUSE:      5,
    FOUR_OF_A_KIND:  4,
    STRAIGHT_FLUSH:  5,
}


def is_bid_feasible(action: int, pool_size: int) -> bool:
    """True if `action` could possibly hold given `pool_size` total pool cards.

    CALL and HH are always feasible-given-legal — they are responses, not bids.
    """
    if action >= NUM_BIDS:
        return True
    return pool_size >= _MIN_CARDS_FOR_HAND[index_to_bid(action).hand_type]


def feasible_action_mask(pool_size: int) -> np.ndarray:
    """Length-NUM_ACTIONS bool mask. CALL and HH are always True."""
    mask = np.empty(NUM_ACTIONS, dtype=np.bool_)
    for a in range(NUM_ACTIONS):
        mask[a] = is_bid_feasible(a, pool_size)
    return mask


def feasible_bid_mask(pool_size: int) -> np.ndarray:
    """Length-NUM_BIDS bool mask over bids only (no CALL/HH).

    Used by HandBelief whose support is the bid space, not the action space.
    """
    mask = np.empty(NUM_BIDS, dtype=np.bool_)
    for a in range(NUM_BIDS):
        mask[a] = is_bid_feasible(a, pool_size)
    return mask
