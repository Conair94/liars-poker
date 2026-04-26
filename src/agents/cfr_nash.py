"""
CFRNashAgent — plays according to the average strategy from a trained CFR run.

Loads the checkpoint produced by cfr_1v1_overnight.py and serves each action
by looking up the infoset in the strategy_sum table, then sampling from the
normalised average strategy.

Key design notes
----------------
- The CFR solver stores infosets keyed by (player, card_rank, history_tuple)
  where history is a tuple of action ints (bid indices, CALL_ACTION, HH_ACTION).
- Card rank = card_index // 4 (0=2 .. 12=A).
- The agent was trained with a restricted bid_space (HC+Pair, 26 bids) and
  max_bids cap.  For any history or action outside that space it falls back to
  a simple heuristic (prefer CALL over HH over smallest legal bid).
- Sampling vs argmax: at Nash the agent must SAMPLE (not argmax) to prevent
  being exploited by a best-response opponent.
"""

from __future__ import annotations

import ast
import json
import os
import random
import sys

# Path setup: src/ for game.*
_HERE    = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from game.bids import CALL_ACTION, HH_ACTION  # noqa: E402
from game.engine import MatchState  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
_DEFAULT_CHECKPOINT = os.path.join(
    _REPO_ROOT, "data", "runs", "cfr_1v1", "mb3_hh_overnight", "checkpoint.json"
)


class CFRNashAgent:
    """
    Plays according to the CFR average strategy loaded from a checkpoint.
    Compatible with the web backend (implements choose_action(state)).
    """

    def __init__(self, checkpoint_path: str = _DEFAULT_CHECKPOINT) -> None:
        self._rng = random.Random()
        self._strategy_sum: dict[tuple, list[float]] = {}
        self._bid_space: tuple[int, ...] = ()
        self._max_bids: int = 6
        self._include_hh: bool = True
        self._loaded = False
        self._checkpoint_path = checkpoint_path
        self._load(checkpoint_path)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path) as f:
            d = json.load(f)
        self._max_bids  = int(d.get("max_bids", 6))
        self._bid_space = tuple(int(x) for x in d.get("bid_space", []))
        self._include_hh = bool(d.get("include_hh", True))
        self._strategy_sum = {
            ast.literal_eval(k): list(v)
            for k, v in d["strategy_sum"].items()
        }
        self._loaded = True

    # ------------------------------------------------------------------

    def _history_from_state(self, state: MatchState) -> tuple[int, ...]:
        """Convert engine round history to the CFR (action,) tuple."""
        rs = state.round_state
        if rs is None:
            return ()
        return tuple(action for _seat, action in rs.history)

    def _legal_cfr(self, history: tuple[int, ...]) -> list[int]:
        """Legal actions in the CFR action space at this history."""
        bid_space   = self._bid_space
        max_bids    = self._max_bids
        include_hh  = self._include_hh

        bids_placed = sum(
            1 for a in history if a not in (CALL_ACTION, HH_ACTION)
        )
        standing = None
        for a in reversed(history):
            if a not in (CALL_ACTION, HH_ACTION):
                standing = a
                break

        if standing is None:
            return list(bid_space)

        actions = [CALL_ACTION]
        if include_hh:
            actions.append(HH_ACTION)
        if bids_placed < max_bids:
            actions.extend(b for b in bid_space if b > standing)
        return actions

    def _average_strategy(
        self, key: tuple, legal: list[int]
    ) -> list[float]:
        ss = self._strategy_sum.get(key)
        if ss is None:
            n = len(legal)
            return [1.0 / n] * n
        total = sum(ss)
        if total > 0:
            return [s / total for s in ss]
        n = len(legal)
        return [1.0 / n] * n

    def _sample(self, probs: list[float], actions: list[int]) -> int:
        r = self._rng.random()
        cumulative = 0.0
        for p, a in zip(probs, actions):
            cumulative += p
            if r <= cumulative:
                return a
        return actions[-1]

    # ------------------------------------------------------------------

    def choose_action(self, state: MatchState) -> int:
        rs = state.round_state
        if rs is None:
            return CALL_ACTION

        seat    = rs.current_player
        hand    = rs.hands[seat]
        rank    = hand[0] // 4 if len(hand) == 1 else max(c // 4 for c in hand)
        history = self._history_from_state(state)

        # Intersect engine legal actions with CFR action space.
        engine_legal = set(state.legal_actions())

        if not self._loaded:
            # Fallback: prefer CALL, then HH, then first legal bid.
            for a in (CALL_ACTION, HH_ACTION):
                if a in engine_legal:
                    return a
            return next(iter(engine_legal))

        cfr_legal = [a for a in self._legal_cfr(history) if a in engine_legal]
        if not cfr_legal:
            # History extends beyond what was trained on; simple fallback.
            for a in (CALL_ACTION, HH_ACTION):
                if a in engine_legal:
                    return a
            return self._rng.choice(list(engine_legal))

        key   = (seat % 2, rank, history)
        probs = self._average_strategy(key, cfr_legal)
        return self._sample(probs, cfr_legal)
