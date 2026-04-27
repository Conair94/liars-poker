"""Adapter ↔ engine state bridge (P5-#2 Phase A, S2).

The single-round OpenSpiel adapter (`LiarsPokerExactState`) and the project
engine (`MatchState`) maintain parallel state during a round. The
exploitability metrics walk the OpenSpiel tree but evaluate agents using
`MatchState`. This module reconstructs an equivalent `MatchState` from a
post-deal `LiarsPokerExactState`.

Single-round only: the engine's notion of `round_history`, `hand_sizes`
shrinkage, eliminations etc. don't apply. We always set
`exact_rules=True` and `high_hand=True` to match the adapter's wire layout.
"""

from __future__ import annotations

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from game.bids import (  # noqa: E402
    CALL_ACTION,
    HH_ACTION,
    bid_to_index,
    index_to_bid,
)
from game.engine import MatchState, RoundState  # noqa: E402

from interop.openspiel_adapter import LiarsPokerExactState  # noqa: E402


def adapter_state_to_match_state(s: LiarsPokerExactState) -> MatchState:
    """Reconstruct a `MatchState` equivalent to the given adapter state.

    Preconditions:
      - The deal must be complete (`s.current_player()` is not CHANCE).
      - The adapter and engine were defined to be parity-equivalent for
        single-round exact-rules+HH play (proven by the round-trip suite).

    The returned `MatchState` is non-terminal even if the adapter state is
    terminal — caller should check `s.is_terminal()` separately. (For the
    metric pipelines we only call this on non-terminal states.)
    """
    if len(s._dealt) < s._total_cards:
        raise ValueError("adapter state is pre-deal; cannot bridge")

    np_ = s._np
    hs  = s._hs

    hands: list[list[int]] = []
    for p in range(np_):
        start = p * hs
        hands.append(sorted(s._dealt[start:start + hs]))

    # Reconstruct seat-tagged history. Adapter alternates seats starting at 0,
    # but bids advance the seat while CALL/HH terminate — the seat sequence
    # here is the seat *that took* each action.
    history: list[tuple[int, int]] = []
    seat = 0
    last_bidder = -1
    current_bid = None
    for action in s._bets:
        history.append((seat, action))
        if action == CALL_ACTION or action == HH_ACTION:
            # Terminator — seat doesn't advance further; loop should end here.
            break
        current_bid = index_to_bid(action)
        last_bidder = seat
        seat = (seat + 1) % np_

    if current_bid is not None and s._current_bid is not None:
        # Sanity check parity with the adapter's own tracked bid.
        assert bid_to_index(current_bid) == bid_to_index(s._current_bid), (
            "history-derived current_bid disagrees with adapter state"
        )

    current_player = s._next_player

    rs = RoundState(
        hands=hands,
        history=list(history),
        current_player=current_player,
        last_bidder=last_bidder if last_bidder >= 0 else -1,
        current_bid=current_bid,
    )

    return MatchState(
        num_players=np_,
        hand_sizes=[hs] * np_,
        active=[True] * np_,
        first_bidder_next=0,
        round_state=rs,
        round_history=[],
        terminal=False,
        winner=None,
        rng=random.Random(),
        mode='countup',
        exact_rules=True,
        high_hand=True,
        five_kings=False,
    )
