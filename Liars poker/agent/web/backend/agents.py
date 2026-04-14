"""
Web agent implementations.

RandomAgent        — picks a uniformly random legal action
BlindBaselineAgent — N=2 backward-induction equilibrium (ignores private cards)
ConditionalAgent   — threshold strategy using private-card conditional priors
"""

from __future__ import annotations

import os
import random
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR   = os.path.abspath(os.path.join(_BACKEND_DIR, "..", ".."))
_PAPER_DIR   = os.path.abspath(os.path.join(_AGENT_DIR, ".."))
for _p in (_PAPER_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.game.engine import MatchState                      # noqa: E402
from agent.game.bids import CALL_ACTION, NUM_BIDS, bid_to_index   # noqa: E402
import numpy as np                                            # noqa: E402


# ---------------------------------------------------------------------------
class RandomAgent:
    """Picks a uniformly random legal action."""

    def choose_action(self, state: MatchState) -> int:
        return random.choice(state.legal_actions())


# ---------------------------------------------------------------------------
class BlindBaselineAgent:
    """
    Threshold strategy from the N=2 blind backward-induction equilibrium.
    Uses the cached equilibrium for n=2..25; raises to the threshold bid
    (where P(pool >= bid) ≈ 50%) and calls above it.

    Falls back to RandomAgent only if the equilibrium cache is unavailable
    or if n is out of range.
    """

    def __init__(self) -> None:
        self._rng = random.Random()

    def choose_action(self, state: MatchState) -> int:
        n = sum(state.hand_sizes[s] for s in state.active_seats())

        try:
            from agent.baseline.blind_equilibrium import get_blind_equilibrium
            eq = get_blind_equilibrium(n)
        except Exception:
            return self._rng.choice(state.legal_actions())

        rs     = state.round_state
        legal  = state.legal_actions()

        # First bid of the round — use the equilibrium's optimal initial bid.
        if rs.current_bid is None:
            action = eq["initial_bid"]
            return action if action in legal else legal[-1]

        # Responding to a standing bid — use the policy.
        bid_idx = bid_to_index(rs.current_bid)
        policy  = eq["policy"]

        # Try actor=0 first; if illegal (shouldn't happen) try actor=1 then call.
        for actor in (0, 1):
            action = policy[bid_idx][actor]
            if action in legal:
                return action

        # Fallback: call if legal, else highest legal bid.
        return CALL_ACTION if CALL_ACTION in legal else legal[-1]


# ---------------------------------------------------------------------------
# Module-level WarmStartLookup cache (loaded once on first use).
_WARM_START: Optional[object] = None


def _get_warm_start():
    global _WARM_START
    if _WARM_START is None:
        from agent.rnad.warm_start import WarmStartLookup
        _WARM_START = WarmStartLookup()
    return _WARM_START


class ConditionalAgent:
    """
    Threshold strategy using private-card conditional priors from WarmStartLookup.

    For each decision:
      - Computes P(pool >= bid | n, own_hand_condition) from the conditional vec.
      - Threshold bid = highest bid where that probability ≥ 50%.
      - If no standing bid: bid the threshold.
      - If standing bid is above threshold (P < 50%): call (it's likely a bluff).
      - If standing bid is at/below threshold: raise to the threshold.

    Falls back to BlindBaselineAgent for n outside the lookup range (2..4).
    """

    def __init__(self) -> None:
        self._blind = BlindBaselineAgent()

    def choose_action(self, state: MatchState) -> int:
        n     = sum(state.hand_sizes[s] for s in state.active_seats())
        rs    = state.round_state
        seat  = rs.current_player
        legal = state.legal_actions()

        # WarmStartLookup covers n=5..25; for small n use blind equilibrium.
        if n < 5 or n > 25:
            return self._blind.choose_action(state)

        try:
            lookup = _get_warm_start()
            _, cond_vec, _ = lookup.get_features(rs.hands[seat], n)
        except Exception:
            return self._blind.choose_action(state)

        # Cumulative: P(pool >= bid_i | condition) = sum(cond_vec[i:])
        cond_p_at_least: list = np.flip(np.cumsum(np.flip(cond_vec))).tolist()

        # Threshold = highest bid index where P(pool >= bid) >= 0.5
        threshold_idx = 0
        for i in range(NUM_BIDS - 1, -1, -1):
            if cond_p_at_least[i] >= 0.5:
                threshold_idx = i
                break

        # --- First bid ---
        if rs.current_bid is None:
            if threshold_idx in legal:
                return threshold_idx
            # All bids are legal when no standing bid exists.
            return legal[0]

        # --- Responding to a standing bid ---
        cur_idx = bid_to_index(rs.current_bid)
        cur_p   = cond_p_at_least[cur_idx]

        # Standing bid is above our threshold (likely a bluff): call.
        if CALL_ACTION in legal and cur_p < 0.5:
            return CALL_ACTION

        # Standing bid is still within range: raise to the threshold (or just above).
        bid_candidates = [a for a in legal if a != CALL_ACTION]
        if not bid_candidates:
            return CALL_ACTION

        for a in bid_candidates:
            if a >= threshold_idx:
                return a

        # All legal bids are already above threshold — smallest legal raise.
        return bid_candidates[0]
