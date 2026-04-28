"""Shared types for the modular agent stack (AR-0a).

Every component boundary in the agent redesign uses dataclasses defined
here. Callers import these types; they never re-derive shapes or dtypes.

See `docs-internal/design/agent_redesign.md` §8 and
`docs-internal/design/agent_redesign_ar0a.md` for the full contract
spec, dtype/tolerance pinning, and JSON canonical form.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from game.bids import NUM_ACTIONS, NUM_BIDS
from game.feasibility import feasible_action_mask

# Tolerances — see ar0a §1.
_SUM_TOL  = 1e-5
_MASK_TOL = 1e-6


# ---------------------------------------------------------------------------
# Infostate — the only thing any head ever sees
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Infostate:
    """Public infostate snapshot. Built by `Infostate.from_match_state(state)`."""

    own_hand:        tuple[int, ...]
    pool_size:       int
    hand_sizes:      tuple[int, ...]
    own_seat:        int
    current_player:  int
    standing_bid:    int | None
    bid_history:     tuple[tuple[int, int], ...]
    legal_actions:   tuple[int, ...]
    feasible_mask:   tuple[bool, ...]
    exact_rules:     bool
    high_hand:       bool
    five_kings:      bool

    @classmethod
    def from_match_state(cls, state) -> Infostate:
        """Build an Infostate from the engine's `MatchState`.

        Reads `state.info_state(seat)` plus `state.legal_actions()`. The
        `seat` is taken from `state.current_player()` — i.e. the
        Infostate is always from the actor's perspective at the current
        decision point.
        """
        seat = state.current_player()
        info = state.info_state(seat)

        cur_bid = info["current_bid"]
        if cur_bid is None:
            standing_bid: int | None = None
        else:
            from game.bids import Bid, bid_to_index
            standing_bid = bid_to_index(Bid(cur_bid[0], cur_bid[1]))

        hand_sizes = tuple(info["hand_sizes"])
        active     = info["active"]
        pool_size  = sum(hand_sizes[s] for s, a in enumerate(active) if a)

        legal       = tuple(state.legal_actions())
        legal_set   = set(legal)
        feasible    = feasible_action_mask(pool_size)
        # Joint mask: only legal *and* feasible. (Legal alone may include
        # an infeasible bid — engine doesn't filter on hand-type plausibility.)
        joint_mask  = tuple(bool(feasible[a] and a in legal_set) for a in range(NUM_ACTIONS))

        return cls(
            own_hand       = tuple(int(c) for c in info["own_hand"]),
            pool_size      = int(pool_size),
            hand_sizes     = hand_sizes,
            own_seat       = int(seat),
            current_player = int(info["current_player"]),
            standing_bid   = standing_bid,
            bid_history    = tuple((int(s), int(a)) for s, a in info["bid_history"]),
            legal_actions  = legal,
            feasible_mask  = joint_mask,
            exact_rules    = bool(state.exact_rules),
            high_hand      = bool(state.high_hand),
            five_kings     = bool(state.five_kings),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "own_hand":       list(self.own_hand),
            "pool_size":      self.pool_size,
            "hand_sizes":     list(self.hand_sizes),
            "own_seat":       self.own_seat,
            "current_player": self.current_player,
            "standing_bid":   self.standing_bid,
            "bid_history":    [list(p) for p in self.bid_history],
            "legal_actions":  list(self.legal_actions),
            "feasible_mask":  list(self.feasible_mask),
            "exact_rules":    self.exact_rules,
            "high_hand":      self.high_hand,
            "five_kings":     self.five_kings,
        }

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> Infostate:
        return cls(
            own_hand       = tuple(int(c) for c in obj["own_hand"]),
            pool_size      = int(obj["pool_size"]),
            hand_sizes     = tuple(int(h) for h in obj["hand_sizes"]),
            own_seat       = int(obj["own_seat"]),
            current_player = int(obj["current_player"]),
            standing_bid   = (None if obj["standing_bid"] is None else int(obj["standing_bid"])),
            bid_history    = tuple((int(s), int(a)) for s, a in obj["bid_history"]),
            legal_actions  = tuple(int(a) for a in obj["legal_actions"]),
            feasible_mask  = tuple(bool(x) for x in obj["feasible_mask"]),
            exact_rules    = bool(obj["exact_rules"]),
            high_hand      = bool(obj["high_hand"]),
            five_kings     = bool(obj["five_kings"]),
        )


# ---------------------------------------------------------------------------
# HandBelief — output of HandModel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HandBelief:
    q:             np.ndarray
    q_logits:      np.ndarray
    feasible_mask: np.ndarray
    n:             int

    def __post_init__(self) -> None:
        if self.q.shape != (NUM_BIDS,):
            raise ValueError(f"q shape {self.q.shape}, expected ({NUM_BIDS},)")
        if self.q_logits.shape != (NUM_BIDS,):
            raise ValueError(f"q_logits shape {self.q_logits.shape}")
        if self.feasible_mask.shape != (NUM_BIDS,):
            raise ValueError(f"feasible_mask shape {self.feasible_mask.shape}")
        if self.q.dtype != np.float32:
            raise TypeError(f"q dtype {self.q.dtype}, expected float32")
        if self.q_logits.dtype != np.float32:
            raise TypeError(f"q_logits dtype {self.q_logits.dtype}")
        if self.feasible_mask.dtype != np.bool_:
            raise TypeError(f"feasible_mask dtype {self.feasible_mask.dtype}")
        if abs(float(self.q.sum()) - 1.0) > _SUM_TOL:
            raise ValueError(f"q does not sum to 1: {self.q.sum()}")
        if float(self.q[~self.feasible_mask].sum()) > _MASK_TOL:
            raise ValueError("q has mass on infeasible bids")

    def top_k(self, k: int = 5) -> list[tuple[int, float]]:
        """Top-`k` (action_idx, prob) pairs by probability, descending."""
        idx = np.argsort(-self.q)[:k]
        return [(int(i), float(self.q[i])) for i in idx if self.q[i] > 0.0]


# ---------------------------------------------------------------------------
# CallDecision — output of CallPolicy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CallDecision:
    p_call:  float
    inputs:  Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.p_call <= 1.0):
            raise ValueError(f"p_call {self.p_call} out of [0, 1]")


# ---------------------------------------------------------------------------
# BidDistribution — output of BidPolicy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BidDistribution:
    pi:            np.ndarray   # (NUM_ACTIONS,)
    pi_logits:     np.ndarray
    legal_mask:    np.ndarray
    feasible_mask: np.ndarray
    entropy:       float
    support_size:  int

    def __post_init__(self) -> None:
        for name, arr in [
            ("pi", self.pi), ("pi_logits", self.pi_logits),
            ("legal_mask", self.legal_mask), ("feasible_mask", self.feasible_mask),
        ]:
            if arr.shape != (NUM_ACTIONS,):
                raise ValueError(f"{name} shape {arr.shape}, expected ({NUM_ACTIONS},)")
        if self.pi.dtype != np.float32 or self.pi_logits.dtype != np.float32:
            raise TypeError("pi / pi_logits must be float32")
        if self.legal_mask.dtype != np.bool_ or self.feasible_mask.dtype != np.bool_:
            raise TypeError("masks must be bool")
        # Empty distribution is allowed (e.g. hh_fired short-circuit) — flagged
        # by support_size == 0.
        if self.support_size > 0:
            if abs(float(self.pi.sum()) - 1.0) > _SUM_TOL:
                raise ValueError(f"pi does not sum to 1: {self.pi.sum()}")
            joint = self.legal_mask & self.feasible_mask
            if float(self.pi[~joint].sum()) > _MASK_TOL:
                raise ValueError("pi has mass outside legal ∩ feasible")
        else:
            if float(self.pi.sum()) > _MASK_TOL:
                raise ValueError("support_size=0 but pi has mass")

    def top_k(self, k: int = 5) -> list[tuple[int, float]]:
        idx = np.argsort(-self.pi)[:k]
        return [(int(i), float(self.pi[i])) for i in idx if self.pi[i] > 0.0]

    @classmethod
    def empty(cls, *, legal_mask: np.ndarray, feasible_mask: np.ndarray) -> BidDistribution:
        """Construct the no-bid sentinel (used when HH fires)."""
        pi     = np.zeros(NUM_ACTIONS, dtype=np.float32)
        logits = np.full(NUM_ACTIONS, -np.inf, dtype=np.float32)
        return cls(
            pi=pi, pi_logits=logits,
            legal_mask=legal_mask.astype(np.bool_),
            feasible_mask=feasible_mask.astype(np.bool_),
            entropy=0.0, support_size=0,
        )


# ---------------------------------------------------------------------------
# AgentDecision — public surface to benchmark / LBR / subgame / logger
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentDecision:
    action_probs: dict[int, float]
    chosen:       int | None = None
    belief:       HandBelief | None = None
    call:         CallDecision | None = None
    bid:          BidDistribution | None = None
    hh_fired:     bool = False

    def __post_init__(self) -> None:
        s = sum(self.action_probs.values())
        if abs(s - 1.0) > _SUM_TOL:
            raise ValueError(f"action_probs does not sum to 1: {s}")

    def trace_json(self) -> dict[str, Any]:
        """Compact JSON suitable for embedding in a per-turn DecisionRecord.

        Top-K only — full vectors are recoverable from a seed-pinned
        re-run.
        """
        out: dict[str, Any] = {"hh_fired": self.hh_fired}
        if self.belief is not None:
            out["belief"] = {
                "q_top5": self.belief.top_k(5),
                "entropy": float(_entropy(self.belief.q)),
                "n":       self.belief.n,
            }
        if self.call is not None:
            out["call"] = {
                "p_call": float(self.call.p_call),
                "inputs": dict(self.call.inputs),
            }
        if self.bid is not None:
            out["bid"] = {
                "support_size": self.bid.support_size,
                "entropy":      float(self.bid.entropy),
                "pi_top5":      self.bid.top_k(5),
            }
        return out


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats. Defined as 0 on the zero distribution."""
    nz = p[p > 0.0]
    if nz.size == 0:
        return 0.0
    return float(-(nz * np.log(nz)).sum())


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class HandModel(Protocol):
    def belief(self, info: Infostate) -> HandBelief: ...
    def belief_batch(self, infos: list[Infostate]) -> list[HandBelief]: ...


@runtime_checkable
class CallPolicy(Protocol):
    def call_prob(self, info: Infostate, q: HandBelief) -> CallDecision: ...


@runtime_checkable
class BidPolicy(Protocol):
    def bid_dist(
        self, info: Infostate, q: HandBelief, *, hh_fired: bool
    ) -> BidDistribution: ...


# ---------------------------------------------------------------------------
# Identity-style HandModel — uniform over feasible bids; used by tests / fallback
# ---------------------------------------------------------------------------

class IdentityHandModel:
    """Uniform-over-feasible HandModel. No learning, no warm-start.

    Concrete implementer of the `HandModel` Protocol — exists for tests
    and as the bottom of the `Analytic` / `Heuristic` / `Learned` ladder
    so the AR-0b ablation matrix has *something* to fill the slot from
    day one.
    """

    def belief(self, info: Infostate) -> HandBelief:
        from game.feasibility import feasible_bid_mask
        mask = feasible_bid_mask(info.pool_size)
        n_feasible = int(mask.sum())
        q = np.zeros(NUM_BIDS, dtype=np.float32)
        if n_feasible > 0:
            q[mask] = 1.0 / n_feasible
        logits = np.where(mask, 0.0, -np.inf).astype(np.float32)
        return HandBelief(q=q, q_logits=logits, feasible_mask=mask, n=info.pool_size)

    def belief_batch(self, infos: list[Infostate]) -> list[HandBelief]:
        return [self.belief(i) for i in infos]


# ---------------------------------------------------------------------------
# Trace JSON serialization for top-K entries (used by decision_capture)
# ---------------------------------------------------------------------------

def trace_to_str(trace: dict[str, Any]) -> str:
    """Stable JSON for an AgentDecision.trace_json() payload."""
    return json.dumps(trace, sort_keys=True, separators=(",", ":"))
