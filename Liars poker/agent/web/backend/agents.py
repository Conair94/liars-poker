"""
Web agent implementations.

RandomAgent        — picks a uniformly random legal action
BlindBaselineAgent — marginal 50% threshold strategy (ignores private cards)
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

from agent.game.engine import MatchState                                     # noqa: E402
from agent.game.bids import CALL_ACTION, HH_ACTION, NUM_BIDS, bid_to_index   # noqa: E402
import numpy as np                                                             # noqa: E402


# ---------------------------------------------------------------------------
class RandomAgent:
    """Picks a uniformly random legal action."""

    def choose_action(self, state: MatchState) -> int:
        return random.choice(state.legal_actions())


# ---------------------------------------------------------------------------
class BlindBaselineAgent:
    """
    Marginal threshold strategy: bids the highest hand where
    P(pool >= bid | total cards = n) >= 50%, and calls when the standing
    bid exceeds that threshold.

    Data sources (in priority order):
      - WarmStartLookup marginal vector for n=5..25 (MC-derived)
      - blind_equilibrium p_at_least for n=2..4 (exact combinatorics)

    BLIND_EQ initial_bid is used as an opening-bid hint for n<=10 only
    when it does not exceed the 50% threshold.
    """

    def __init__(self) -> None:
        self._rng = random.Random()

    def _get_p_at_least(self, n: int, state: MatchState) -> Optional[np.ndarray]:
        """Return (NUM_BIDS,) P(pool >= bid_i | n) appropriate to game mode."""
        if n >= 5:
            try:
                lookup = _get_warm_start()
                # Exact rules mode: use dedicated exact-rules probability table if available
                if getattr(state, 'exact_rules', False):
                    exact_pal = lookup.get_exact_rules_at_least(n)
                    if exact_pal is not None:
                        return exact_pal
                # Five-kings mode: use 53-card deck table if available
                if getattr(state, 'five_kings', False):
                    fk_pal = lookup.get_five_kings_at_least(n)
                    if fk_pal is not None:
                        return fk_pal
                # Standard: marginal cumulative
                marginal, _, _ = lookup.get_features([], n)
                return np.flip(np.cumsum(np.flip(marginal))).astype(np.float32)
            except Exception:
                pass
        # n < 5 or WarmStart unavailable: exact blind equilibrium
        try:
            from agent.baseline.blind_equilibrium import get_blind_equilibrium
            eq = get_blind_equilibrium(n)
            return np.array(eq["p_at_least"], dtype=np.float32)
        except Exception:
            return None

    def choose_action(self, state: MatchState) -> int:
        n = sum(state.hand_sizes[s] for s in state.active_seats())

        p_at_least = self._get_p_at_least(n, state)
        if p_at_least is None:
            return self._rng.choice(state.legal_actions())

        rs    = state.round_state
        legal = state.legal_actions()

        # Threshold = highest bid index where P(pool >= bid) >= 50%
        threshold_idx = 0
        for i in range(NUM_BIDS - 1, -1, -1):
            if p_at_least[i] >= 0.5:
                threshold_idx = i
                break

        if rs.current_bid is None:
            # Use equilibrium initial_bid as opening hint (n<=10 only, must not exceed threshold)
            if n <= 10:
                try:
                    from agent.baseline.blind_equilibrium import get_blind_equilibrium
                    eq = get_blind_equilibrium(n)
                    initial = eq["initial_bid"]
                    if initial <= threshold_idx and initial in legal:
                        return initial
                except Exception:
                    pass
            return threshold_idx if threshold_idx in legal else legal[0]

        # Responding: call if standing bid is above threshold, else raise to threshold
        cur_idx = bid_to_index(rs.current_bid)
        if CALL_ACTION in legal and float(p_at_least[cur_idx]) < 0.5:
            return CALL_ACTION

        bid_candidates = [a for a in legal if a not in (CALL_ACTION, HH_ACTION)]
        if not bid_candidates:
            return CALL_ACTION

        for a in bid_candidates:
            if a >= threshold_idx:
                return a

        return bid_candidates[0]


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

    Falls back to BlindBaselineAgent (marginal threshold strategy) for n outside
    the WarmStartLookup range (n<5 or n>25).
    """

    def __init__(self) -> None:
        self._blind = BlindBaselineAgent()

    def choose_action(self, state: MatchState) -> int:
        n     = sum(state.hand_sizes[s] for s in state.active_seats())
        rs    = state.round_state
        seat  = rs.current_player
        legal = state.legal_actions()

        # Exact rules or five-kings: conditional tables aren't calibrated for these modes,
        # so fall back to BlindBaselineAgent which will use the mode-specific PAL tables.
        if getattr(state, 'exact_rules', False) or getattr(state, 'five_kings', False):
            return self._blind.choose_action(state)

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
        bid_candidates = [a for a in legal if a not in (CALL_ACTION, HH_ACTION)]
        if not bid_candidates:
            return CALL_ACTION

        for a in bid_candidates:
            if a >= threshold_idx:
                return a

        # All legal bids are already above threshold — smallest legal raise.
        return bid_candidates[0]
