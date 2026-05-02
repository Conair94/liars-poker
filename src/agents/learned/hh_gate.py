"""
Free-standing High-Hand declaration gate (AR-2 design §4.4).

Single source of truth for "should we declare HH?" — used by `ModularNashAgent`,
the CFR+ subgame solver's forced-HH labeling during distillation, and the
`ExactRulesConditional*` agents.

Fires iff the standing bid is exactly the argmax of the belief AND its mass is
within `hh_band` of the peak. Strict argmax (not "near-peak") because HH is
decision-theoretically optimal only when the standing bid is the *true* most
likely pool hand under the ±1 payoff model.
"""

from __future__ import annotations

import numpy as np


def should_declare_hh(
    q: np.ndarray,
    standing_bid: int | None,
    *,
    hh_band: float = 0.95,
) -> bool:
    if standing_bid is None:
        return False
    q_max = float(q.max())
    if q_max <= 0.0:
        return False
    if int(np.argmax(q)) != int(standing_bid):
        return False
    return float(q[int(standing_bid)]) >= hh_band * q_max
