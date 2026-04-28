"""
Decision-log schema and JSONL writer (TRAINING_PIPELINE_DESIGN §6).

One JSON object per line, per turn, per agent action. Records are written
incrementally as games are played; no in-memory buffering across the run.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Choice:
    action: str
    p: float | None = None
    eu: float | None = None
    feasible: bool = True


@dataclass
class Outcome:
    challenged: bool | None = None
    bid_existed: bool | None = None
    point_delta: int | None = None


@dataclass
class DecisionRecord:
    run_id: str
    game_id: int
    turn: int
    agent: str
    agent_seat: int
    opponent: str
    ruleset: dict[str, Any]
    state: dict[str, Any]
    choices: list[Choice]
    chosen: str
    reasoning_tag: str | None = None
    outcome: Outcome | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    # AR-0a: optional modular-agent trace fields. None for legacy agents.
    belief: dict[str, Any] | None = None
    call: dict[str, Any] | None = None
    bid: dict[str, Any] | None = None
    hh_fired: bool | None = None

    def to_json_line(self) -> str:
        d = asdict(self)
        # Drop None-valued top-level fields except where schema requires them.
        if d.get("outcome") is None:
            d.pop("outcome", None)
        if not d.get("extras"):
            d.pop("extras", None)
        for k in ("belief", "call", "bid", "hh_fired"):
            if d.get(k) is None:
                d.pop(k, None)
        return json.dumps(d, separators=(",", ":"))


class DecisionLogger:
    """Append-only JSONL writer. One file per run."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Truncate on open: a run owns its log file.
        self._fh = open(path, "w", buffering=1)

    def write(self, record: DecisionRecord) -> None:
        self._fh.write(record.to_json_line())
        self._fh.write("\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None  # type: ignore[assignment]

    def __enter__(self) -> DecisionLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Action / state encoding helpers
# ---------------------------------------------------------------------------

_RANK_NAME = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
_HAND_ABBREV = ["HC", "Pair", "2P", "3oK", "St", "Fl", "FH", "4oK", "SF"]


def action_to_str(action_idx: int) -> str:
    """Map an engine action index to a stable string token for the log."""
    from game.bids import CALL_ACTION, HH_ACTION, NUM_BIDS, index_to_bid

    if action_idx == CALL_ACTION:
        return "call"
    if action_idx == HH_ACTION:
        return "high_hand"
    if 0 <= action_idx < NUM_BIDS:
        b = index_to_bid(action_idx)
        return f"bid:{_HAND_ABBREV[b.hand_type]}:{_RANK_NAME[b.primary_rank]}"
    return f"action:{action_idx}"


def card_to_str(card_idx: int) -> str:
    """Inverse of poker_math_exact: rank*4+suit -> '<R><S>' (e.g. AS)."""
    rank = card_idx // 4
    suit = card_idx % 4
    return f"{_RANK_NAME[rank]}{'CDHS'[suit]}"
