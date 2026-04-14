"""
Card-based Liar's Poker match engine (pure Python).

Match structure (see AGENT_DESIGN.md §2):

  - Match begins with N players (2..5), each holding hand_size = 1.
  - Each round:
      * Deal fresh private cards to each active player (size = their hand_size).
      * Starting from the designated first bidder, players alternate in seat
        order. Each turn the acting player either raises (any strictly
        stronger bid) or calls the current standing bid a lie.
      * The first bidder MUST bid (no call before any bid exists).
      * On a call, concatenate all active players' private cards into a pool
        and evaluate the best 5-card hand. If the pool ≥ bid, the caller
        loses; otherwise the bidder loses.
  - Loser's hand_size += 1. If that would exceed 5, the player is eliminated
    (they lost a round while holding 5 cards).
  - The first bidder of the next round is the loser — or, if the loser was
    eliminated, the next active seat clockwise from them.
  - Match terminates when only one active player remains; that player wins.

The engine is dependency-free and intentionally pure-Python. It does NOT
depend on OpenSpiel. An OpenSpiel wrapper can be added as a thin adapter
once the core rules are validated.

The hand evaluator is imported from the Stage 1 `poker_math_exact` module.
"""

from __future__ import annotations

import os
import random
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .bids import (
    Bid, CALL_ACTION, NUM_ACTIONS, NUM_BIDS,
    bid_to_index, index_to_bid, normalize_hand_type,
)

# Import the Stage 1 evaluator. Walk up to the paper directory on sys.path.
_PAPER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PAPER_DIR not in sys.path:
    sys.path.insert(0, _PAPER_DIR)

from poker_math_exact import _evaluate_ranked  # noqa: E402


MAX_HAND_SIZE = 5
MIN_PLAYERS = 2
MAX_PLAYERS = 5


# ---------------------------------------------------------------------------
# Per-round state
# ---------------------------------------------------------------------------


@dataclass
class RoundState:
    """State of the currently-active round (None between rounds / at terminal)."""
    # Private hands, indexed by ORIGINAL seat index. Eliminated / inactive seats
    # hold an empty list. Active seats hold exactly hand_sizes[seat] cards.
    hands: List[List[int]]
    # Bid history for this round, in turn order: (seat, action_idx)
    # action_idx ∈ {0..NUM_BIDS-1 (a bid), CALL_ACTION}
    history: List[Tuple[int, int]] = field(default_factory=list)
    # Current seat to act (original seat index).
    current_player: int = 0
    # Seat that made the most recent bid (for resolution). -1 before any bid.
    last_bidder: int = -1
    # The most recent bid made this round, None if no bid yet.
    current_bid: Optional[Bid] = None


@dataclass
class RoundResult:
    loser_seat: int
    winner_seat: int            # the player who won the call (bidder or caller)
    bid: Bid
    caller_seat: int
    pool: List[int]
    pool_best: Bid              # (hand_type, primary_rank) of the pool's best 5-card hand
    call_succeeded: bool        # True if the caller won (bid was a lie)


# ---------------------------------------------------------------------------
# Match state
# ---------------------------------------------------------------------------


@dataclass
class MatchState:
    num_players: int                   # original seat count (2..5)
    hand_sizes: List[int]              # per-seat; 0 means eliminated
    active: List[bool]                 # per-seat; True if still in the match
    first_bidder_next: int             # seat to lead off the next round
    round_state: Optional[RoundState]  # None between rounds / at terminal
    round_history: List[RoundResult] = field(default_factory=list)
    terminal: bool = False
    winner: Optional[int] = None
    rng: random.Random = field(default_factory=random.Random)

    # --- public API -------------------------------------------------------

    def active_seats(self) -> List[int]:
        return [i for i, a in enumerate(self.active) if a]

    def num_active(self) -> int:
        return sum(1 for a in self.active if a)

    def is_chance_node(self) -> bool:
        """
        Chance nodes occur between rounds — a fresh deal is needed. We resolve
        chance eagerly (in start_next_round()), so from the external API's
        perspective the game is fully deterministic at the player's view.
        Exposed for symmetry with OpenSpiel conventions.
        """
        return (not self.terminal) and self.round_state is None

    def current_player(self) -> int:
        if self.terminal:
            raise RuntimeError("match is terminal")
        if self.round_state is None:
            raise RuntimeError("between rounds; call start_next_round() first")
        return self.round_state.current_player

    def legal_actions(self) -> List[int]:
        """Legal action indices for the current player in the current round."""
        if self.terminal:
            return []
        if self.round_state is None:
            raise RuntimeError("between rounds; call start_next_round() first")

        rs = self.round_state
        # If no bid has been made yet, must bid (no call).
        if rs.current_bid is None:
            return list(range(NUM_BIDS))
        # Otherwise any strictly stronger bid, or a call.
        cur_idx = bid_to_index(rs.current_bid)
        return list(range(cur_idx + 1, NUM_BIDS)) + [CALL_ACTION]

    def apply_action(self, action: int) -> Optional[RoundResult]:
        """
        Apply `action` for the current player. On a terminal round (CALL),
        resolves the round and advances match state. Returns the RoundResult
        if the round ended, else None.

        If the match becomes terminal as a result of this action, this call
        also sets `self.terminal = True` and `self.winner`.
        """
        if self.terminal:
            raise RuntimeError("match is terminal")
        if self.round_state is None:
            raise RuntimeError("between rounds; call start_next_round() first")
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action}")

        rs = self.round_state
        actor = rs.current_player
        rs.history.append((actor, action))

        if action == CALL_ACTION:
            return self._resolve_call(actor)

        # It's a raise (bid).
        new_bid = index_to_bid(action)
        rs.current_bid = new_bid
        rs.last_bidder = actor
        rs.current_player = self._next_active_seat(actor)
        return None

    def start_next_round(self) -> None:
        """
        Deal a new round. Legal only when between rounds and not terminal.
        The first bidder is `self.first_bidder_next`.
        """
        if self.terminal:
            raise RuntimeError("match is terminal")
        if self.round_state is not None:
            raise RuntimeError("a round is already in progress")

        # Deal fresh hands to active players from a shuffled 52-card deck.
        deck = list(range(52))
        self.rng.shuffle(deck)
        hands: List[List[int]] = [[] for _ in range(self.num_players)]
        idx = 0
        for seat in range(self.num_players):
            if self.active[seat]:
                k = self.hand_sizes[seat]
                hands[seat] = sorted(deck[idx: idx + k])
                idx += k

        # First bidder: the designated seat (must still be active; if not,
        # walk forward until we find an active seat).
        first = self.first_bidder_next
        if not self.active[first]:
            first = self._next_active_seat(first)

        self.round_state = RoundState(
            hands=hands,
            history=[],
            current_player=first,
            last_bidder=-1,
            current_bid=None,
        )

    # --- information state ------------------------------------------------

    def info_state(self, seat: int) -> dict:
        """
        Everything seat `seat` can legally observe. Used by the network
        encoder in Stage 2. Returns a dict of primitive Python types so it
        is trivially serializable.
        """
        rs = self.round_state
        own_hand: List[int] = list(rs.hands[seat]) if (rs is not None) else []
        own_hand.sort()

        return {
            "num_players":       self.num_players,
            "seat":              seat,
            "hand_sizes":        list(self.hand_sizes),
            "active":            list(self.active),
            "own_hand":          own_hand,
            "current_player":    None if rs is None else rs.current_player,
            "current_bid":       None if (rs is None or rs.current_bid is None)
                                 else (rs.current_bid.hand_type, rs.current_bid.primary_rank),
            "last_bidder":       None if rs is None else rs.last_bidder,
            "bid_history":       [] if rs is None else list(rs.history),
            "round_index":       len(self.round_history),
            "round_history_summary": [
                (r.loser_seat, r.bid.hand_type, r.bid.primary_rank, r.caller_seat, r.call_succeeded)
                for r in self.round_history
            ],
            "terminal":          self.terminal,
            "winner":            self.winner,
        }

    def returns(self) -> List[float]:
        """
        Per-seat zero-sum returns. +1 to match winner, −1 to eliminated
        players, 0 to active non-winners (only present in non-terminal states).
        """
        out = [0.0] * self.num_players
        if not self.terminal:
            return out
        for s in range(self.num_players):
            if s == self.winner:
                out[s] = 1.0
            elif not self.active[s]:
                out[s] = -1.0
            else:
                out[s] = 0.0  # shouldn't happen at terminal
        return out

    def clone(self) -> "MatchState":
        """Deep clone, including RNG state (so simulated rollouts are reproducible)."""
        new = MatchState(
            num_players=self.num_players,
            hand_sizes=list(self.hand_sizes),
            active=list(self.active),
            first_bidder_next=self.first_bidder_next,
            round_state=deepcopy(self.round_state),
            round_history=list(self.round_history),
            terminal=self.terminal,
            winner=self.winner,
            rng=random.Random(),
        )
        new.rng.setstate(self.rng.getstate())
        return new

    # --- internal helpers -------------------------------------------------

    def _next_active_seat(self, seat: int) -> int:
        """Next seat in clockwise order that is still active."""
        n = self.num_players
        for step in range(1, n + 1):
            cand = (seat + step) % n
            if self.active[cand]:
                return cand
        raise RuntimeError("no active seats remaining")

    def _resolve_call(self, caller_seat: int) -> RoundResult:
        rs = self.round_state
        assert rs is not None and rs.current_bid is not None and rs.last_bidder >= 0

        bid = rs.current_bid
        bidder = rs.last_bidder

        # Build the pool from all active seats' hands.
        pool: List[int] = []
        for s in range(self.num_players):
            if self.active[s]:
                pool.extend(rs.hands[s])

        raw_type, raw_primary = _evaluate_ranked(pool)
        # Normalize: the evaluator returns ROYAL_FLUSH=9 for A-high SFs, but
        # the bid space collapses RF onto (STRAIGHT_FLUSH, A). Normalize before
        # comparing, so a bid of (SF, A) correctly "holds" against a pool that
        # evaluates as a Royal Flush.
        pool_type, pool_primary = normalize_hand_type(raw_type, raw_primary)
        pool_best = Bid(pool_type, pool_primary)

        # Compare: pool_best >= bid?
        bid_holds = (pool_type, pool_primary) >= (bid.hand_type, bid.primary_rank)

        if bid_holds:
            loser = caller_seat
            winner = bidder
        else:
            loser = bidder
            winner = caller_seat

        result = RoundResult(
            loser_seat=loser,
            winner_seat=winner,
            bid=bid,
            caller_seat=caller_seat,
            pool=list(pool),
            pool_best=pool_best,
            call_succeeded=(not bid_holds),
        )
        self.round_history.append(result)

        # Advance hand sizes / elimination.
        new_size = self.hand_sizes[loser] + 1
        if new_size > MAX_HAND_SIZE:
            # Elimination: loser lost a round while already at 5 cards.
            self.active[loser] = False
            self.hand_sizes[loser] = 0
            # Next first bidder: next active seat after the eliminated one.
            self.first_bidder_next = self._next_active_seat(loser) if self.num_active() > 0 else loser
        else:
            self.hand_sizes[loser] = new_size
            self.first_bidder_next = loser  # loser leads off next round

        # Clear round state.
        self.round_state = None

        # Check terminal.
        if self.num_active() == 1:
            self.terminal = True
            self.winner = self.active_seats()[0]

        return result


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def new_match(num_players: int, seed: Optional[int] = None) -> MatchState:
    if not (MIN_PLAYERS <= num_players <= MAX_PLAYERS):
        raise ValueError(f"num_players must be in [{MIN_PLAYERS}, {MAX_PLAYERS}]")

    rng = random.Random(seed)
    state = MatchState(
        num_players=num_players,
        hand_sizes=[1] * num_players,
        active=[True] * num_players,
        first_bidder_next=0,
        round_state=None,
        round_history=[],
        terminal=False,
        winner=None,
        rng=rng,
    )
    return state
