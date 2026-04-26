"""
Non-invasive decision-capture wrapper for any agent that exposes
`choose_action(state) -> int`. Per design §6 we record one DecisionRecord
per turn; per plan P3 we wrap rather than refactor — `p`/`eu` per choice
are left None until the modular interface lands in P5.
"""

from __future__ import annotations

from typing import Protocol

from training.logging import (
    Choice,
    DecisionLogger,
    DecisionRecord,
    action_to_str,
    card_to_str,
)


class _ActAgent(Protocol):
    def choose_action(self, state) -> int: ...


class LoggingAgentWrapper:
    """
    Wraps an agent so each choose_action() call emits a DecisionRecord
    to the supplied logger. The wrapper forwards the chosen action
    unchanged. Other attributes (e.g. attached state on heuristic
    agents) are proxied via __getattr__.
    """

    def __init__(
        self,
        inner: _ActAgent,
        *,
        logger: DecisionLogger,
        run_id: str,
        agent_name: str,
        agent_seat: int,
        opponent_name: str,
        ruleset: dict,
        game_id_ref: list[int],
        turn_counter_ref: list[int],
    ):
        self._inner = inner
        self._logger = logger
        self._run_id = run_id
        self._agent_name = agent_name
        self._agent_seat = agent_seat
        self._opponent_name = opponent_name
        self._ruleset = dict(ruleset)
        # Mutable references shared with the match driver — avoids threading
        # game/turn ids through every method on the inner agent.
        self._game_id_ref = game_id_ref
        self._turn_counter_ref = turn_counter_ref

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def choose_action(self, state) -> int:
        legal = list(state.legal_actions())
        chosen = self._inner.choose_action(state)
        record = DecisionRecord(
            run_id=self._run_id,
            game_id=self._game_id_ref[0],
            turn=self._turn_counter_ref[0],
            agent=self._agent_name,
            agent_seat=self._agent_seat,
            opponent=self._opponent_name,
            ruleset=self._ruleset,
            state=_snapshot_state(state, self._agent_seat),
            choices=[Choice(action=action_to_str(a)) for a in legal],
            chosen=action_to_str(chosen),
            reasoning_tag=None,
        )
        self._logger.write(record)
        self._turn_counter_ref[0] += 1
        return chosen


def _snapshot_state(state, seat: int) -> dict:
    info = state.info_state(seat)
    own_hand = [card_to_str(c) for c in info["own_hand"]]
    cur_bid = info["current_bid"]
    if cur_bid is None:
        standing_bid = None
    else:
        # current_bid is (hand_type, primary_rank); reuse action_to_str via index
        from game.bids import Bid, bid_to_index

        standing_bid = action_to_str(bid_to_index(Bid(cur_bid[0], cur_bid[1])))
    return {
        "hand": own_hand,
        "hand_sizes": list(info["hand_sizes"]),
        "cards_in_pool": sum(info["hand_sizes"]),
        "standing_bid": standing_bid,
        "bid_history_len": len(info["bid_history"]),
        "round_index": info["round_index"],
    }
