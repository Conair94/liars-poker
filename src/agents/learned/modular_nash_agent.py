"""ModularNashAgent — composes (HandModel, CallPolicy, BidPolicy, HH gate).

Wiring per AR-2 design §4.4 / §5.1 / §5.4:

  1. Build the `Infostate` from the engine's `MatchState`.
  2. Compute `belief = HandModel.belief(info)`.
  3. If HH is legal and `should_declare_hh(belief.q, standing_bid)` fires →
     return the degenerate `{HH_ACTION: 1.0}` distribution.
  4. Else compute `p_call = CallPolicy.call_prob(info, belief).p_call`
     (forced to 0 when CALL is not legal — e.g. opening bid).
  5. Sample bids from `BidPolicy.bid_dist(info, belief, hh_fired=False).pi`,
     scaled by `(1 - p_call)` so the joint sums to 1.

`action_probs` returns the full mixed distribution; `choose_action` samples
from it using the engine's match RNG (design §5.4) — the legacy 4-way mixing
fingerprint is gone.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE    = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from agents.contracts import (  # noqa: E402
    AgentDecision,
    BidPolicy,
    CallPolicy,
    HandModel,
    Infostate,
)
from agents.learned.hh_gate import should_declare_hh  # noqa: E402
from game.bids import CALL_ACTION, HH_ACTION, NUM_BIDS  # noqa: E402
from game.engine import MatchState  # noqa: E402


class ModularNashAgent:
    """Compose `HandModel → {HH gate, CallPolicy, BidPolicy}` into a Policy.

    Construct with already-loaded components (the registry factory loads them
    from checkpoints — see `registry._make_modular_nash`).

    `hh_band` matches the design's default (0.95); pass-through to the gate.
    """

    def __init__(
        self,
        hand_model: HandModel,
        call_policy: CallPolicy,
        bid_policy:  BidPolicy,
        *,
        hh_band: float = 0.95,
    ) -> None:
        self.hand_model  = hand_model
        self.call_policy = call_policy
        self.bid_policy  = bid_policy
        self.hh_band     = hh_band

    # ------------------------------------------------------------------
    # Decision API
    # ------------------------------------------------------------------

    def decision(self, state: MatchState) -> AgentDecision:
        info   = Infostate.from_match_state(state)
        belief = self.hand_model.belief(info)
        legal  = set(info.legal_actions)

        # 1. HH gate — single source of truth (design §4.4).
        if HH_ACTION in legal and info.standing_bid is not None \
                and should_declare_hh(belief.q, info.standing_bid, hh_band=self.hh_band):
            return AgentDecision(
                action_probs={HH_ACTION: 1.0},
                chosen=None,
                belief=belief,
                call=None,
                bid=None,
                hh_fired=True,
            )

        # 2. CallPolicy — only meaningful when CALL is legal.
        call = self.call_policy.call_prob(info, belief)
        p_call = float(call.p_call) if CALL_ACTION in legal else 0.0

        # 3. BidPolicy — pi is over NUM_ACTIONS with mass on bids only.
        bid = self.bid_policy.bid_dist(info, belief, hh_fired=False)

        # 4. Combine. If bid support is empty (no feasible bids), CALL gets all mass.
        if bid.support_size == 0:
            if CALL_ACTION in legal:
                probs = {CALL_ACTION: 1.0}
            else:
                # Pathological: no CALL, no bid support. Fall back to uniform over legal.
                probs = {int(a): 1.0 / len(legal) for a in legal}
            return AgentDecision(
                action_probs=probs,
                chosen=None,
                belief=belief,
                call=call,
                bid=bid,
                hh_fired=False,
            )

        probs: dict[int, float] = {}
        if CALL_ACTION in legal and p_call > 0.0:
            probs[CALL_ACTION] = p_call
        scale = 1.0 - (p_call if CALL_ACTION in legal else 0.0)
        for a in range(NUM_BIDS):
            mass = float(bid.pi[a])
            if mass > 0.0:
                probs[a] = probs.get(a, 0.0) + scale * mass

        # Defensive renorm to guard against float drift.
        total = sum(probs.values())
        if total > 0.0 and abs(total - 1.0) > 1e-6:
            probs = {a: p / total for a, p in probs.items()}

        return AgentDecision(
            action_probs=probs,
            chosen=None,
            belief=belief,
            call=call,
            bid=bid,
            hh_fired=False,
        )

    def action_probs(self, state: MatchState) -> dict[int, float]:
        return self.decision(state).action_probs

    def choose_action(self, state: MatchState) -> int:
        probs = self.action_probs(state)
        if not probs:
            return state.legal_actions()[0]
        actions = list(probs.keys())
        weights = np.array([probs[a] for a in actions], dtype=np.float64)
        s = weights.sum()
        if s <= 0.0:
            return actions[0]
        weights /= s
        # Use the engine's match RNG so sampling is reproducible per seed
        # (design §5.4). `random.Random.choices` accepts a weights list.
        return state.rng.choices(actions, weights=weights.tolist(), k=1)[0]


__all__ = ["ModularNashAgent"]
