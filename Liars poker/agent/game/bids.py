"""
Bid space for card-based Liar's Poker.

A bid is a (hand_type, primary_rank) pair where hand_type ∈ 0..8
(see poker_math_exact.HAND_NAMES) and primary_rank ∈ 0..12 with
hand-type-specific semantics matching poker_math_exact._evaluate_ranked:

  HC  primary = max rank in the hand (0..12)
  Pair primary = rank of the pair (0..12)
  2P  primary = rank of the HIGHER pair (1..12, higher pair needs something below)
  3K  primary = rank of the trips (0..12)
  St  primary = high card of the straight (3..12: 5-high through A-high)
  Fl  primary = highest card in the flush (0..12 — we accept syntactically
                valid low-high-card flush bids; most will be dominated and
                never played in practice)
  FH  primary = rank of the trips (0..12)
  4K  primary = rank of the quads (0..12)
  SF  primary = high card of the straight flush (3..12: 5-high through A-high)

Note: Royal Flush is NOT a distinct bid — it is simply an Ace-high Straight
Flush, represented as Bid(STRAIGHT_FLUSH, 12). The engine normalizes the
evaluator's ROYAL_FLUSH output (hand_type=9) to STRAIGHT_FLUSH+A before any
bid comparison, so the bid space contains exactly one representation of the
strongest hand.

Bids are totally ordered: first by hand_type, then by primary_rank.
The ordering matches the natural comparator on (hand_type, primary_rank)
tuples, and the index of a bid in ascending order is stable and usable
directly as a discrete-action index for an RL policy head.

Action space for the game engine:
  actions 0..NUM_BIDS-1  → bids in ascending order
  action  NUM_BIDS       → CALL (accuse previous bidder of bluffing)
  NUM_ACTIONS = NUM_BIDS + 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Hand type constants (mirror poker_math_exact, avoiding a hard import to keep
# this module importable in isolation)
# ---------------------------------------------------------------------------

HIGH_CARD       = 0
PAIR            = 1
TWO_PAIR        = 2
THREE_OF_A_KIND = 3
STRAIGHT        = 4
FLUSH           = 5
FULL_HOUSE      = 6
FOUR_OF_A_KIND  = 7
STRAIGHT_FLUSH  = 8
ROYAL_FLUSH     = 9

HAND_NAMES = [
    "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
    "Flush", "Full House", "Four of a Kind", "Straight Flush", "Royal Flush",
]

HAND_ABBREV = ["HC", "Pr", "2P", "3K", "St", "Fl", "FH", "4K", "SF", "RF"]

RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

# Per hand-type legal primary_rank ranges, inclusive.
#
# Rationale for the ranges:
#   TWO_PAIR       : higher pair rank > some lower pair rank, so ≥ 1
#   STRAIGHT       : smallest straight is 5-high (A-2-3-4-5); rank index of 5 is 3
#   STRAIGHT_FLUSH : same range as STRAIGHT; A-high SF subsumes Royal Flush
#   ROYAL_FLUSH    : NOT a distinct bid — see module docstring
#   Others         : full 0..12
_PRIMARY_RANGES = {
    HIGH_CARD:       (0, 12),
    PAIR:            (0, 12),
    TWO_PAIR:        (1, 12),
    THREE_OF_A_KIND: (0, 12),
    STRAIGHT:        (3, 12),
    FLUSH:           (0, 12),
    FULL_HOUSE:      (0, 12),
    FOUR_OF_A_KIND:  (0, 12),
    STRAIGHT_FLUSH:  (3, 12),
}


@dataclass(frozen=True, order=True)
class Bid:
    """Total ordering: compare by (hand_type, primary_rank) tuple."""
    hand_type: int
    primary_rank: int

    def __str__(self) -> str:
        return f"{HAND_NAMES[self.hand_type]} {RANK_NAMES[self.primary_rank]}"

    def short(self) -> str:
        return f"{HAND_ABBREV[self.hand_type]} {RANK_NAMES[self.primary_rank]}"


def enumerate_bids() -> List[Bid]:
    """All legal bids in ascending total order (weakest first)."""
    out: List[Bid] = []
    for ht in range(STRAIGHT_FLUSH + 1):     # HC .. SF inclusive; no RF
        lo, hi = _PRIMARY_RANGES[ht]
        for pr in range(lo, hi + 1):
            out.append(Bid(ht, pr))
    return out


def normalize_hand_type(hand_type: int, primary_rank: int) -> Tuple[int, int]:
    """
    Collapse the evaluator's ROYAL_FLUSH output onto STRAIGHT_FLUSH+A.
    Call this before constructing a Bid from an evaluator's (type, rank) pair.
    """
    if hand_type == ROYAL_FLUSH:
        return (STRAIGHT_FLUSH, 12)
    return (hand_type, primary_rank)


_ALL_BIDS: List[Bid] = enumerate_bids()
_BID_TO_INDEX = {b: i for i, b in enumerate(_ALL_BIDS)}

NUM_BIDS     = len(_ALL_BIDS)
CALL_ACTION  = NUM_BIDS             # one past the last bid index
NUM_ACTIONS  = NUM_BIDS + 1


def bid_to_index(bid: Bid) -> int:
    return _BID_TO_INDEX[bid]


def index_to_bid(idx: int) -> Bid:
    if idx < 0 or idx >= NUM_BIDS:
        raise IndexError(f"bid index {idx} out of range [0, {NUM_BIDS})")
    return _ALL_BIDS[idx]


def all_bids() -> List[Bid]:
    """Return a fresh copy of the full ordered bid list."""
    return list(_ALL_BIDS)
