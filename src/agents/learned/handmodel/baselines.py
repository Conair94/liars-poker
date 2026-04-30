"""Baseline `HandModel` adapters for the AR-1 ablation matrix.

Both adapters implement the AR-0a `HandModel` Protocol so they slot
into `ModularNashAgent` and the sweep harness without special-casing.

- `AnalyticHandModel` — wraps `WarmStartLookup` exact-rules conditional
  table. No bid-history conditioning. The Brier baseline AR-1 must
  Pareto-dominate.
- `HeuristicHandModel` — exposes the heuristic-ladder
  `ExactRulesOpponentModelAgent._compute_adj_exact` posterior, which
  *does* fold in opponent-bid signals. Useful as a third point in the
  calibration plot.

Neither adapter is parameterised; both are stateless (modulo the lazy
`WarmStartLookup` cache).
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE      = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_PROBS_DIR = os.path.join(_SRC_DIR, "training", "probs")
for _p in (_PROBS_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.contracts import HandBelief, Infostate  # noqa: E402
from game.bids import Bid, NUM_BIDS, index_to_bid  # noqa: E402
from game.feasibility import feasible_bid_mask  # noqa: E402

_TINY = 1e-12


# ---------------------------------------------------------------------------
# Lazy WarmStartLookup
# ---------------------------------------------------------------------------

_LOOKUP = None


def _get_lookup():
    global _LOOKUP
    if _LOOKUP is None:
        from agents.learned.rnad.warm_start import WarmStartLookup
        _LOOKUP = WarmStartLookup()
    return _LOOKUP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(q: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mask, renormalise, and return (q, q_logits) at AR-0a dtypes."""
    q = q.astype(np.float32) * mask
    s = float(q.sum())
    if s > 0:
        q /= s
    else:
        # Should never happen — fall back to uniform-over-feasible so the
        # AR-0a invariants still pass.
        n_feas = max(int(mask.sum()), 1)
        q = mask.astype(np.float32) / n_feas
    logits = np.where(mask, np.log(q + _TINY), -np.inf).astype(np.float32)
    return q.astype(np.float32), logits


def _uniform_belief(n: int) -> HandBelief:
    mask = feasible_bid_mask(n)
    n_feas = max(int(mask.sum()), 1)
    q = mask.astype(np.float32) / n_feas
    logits = np.where(mask, 0.0, -np.inf).astype(np.float32)
    return HandBelief(q=q, q_logits=logits, feasible_mask=mask, n=n)


# ---------------------------------------------------------------------------
# AnalyticHandModel
# ---------------------------------------------------------------------------

class AnalyticHandModel:
    """`q(pool | own_hand, n)` from the exact-rules conditional table.

    Bid-history is ignored by construction — this is the AR-1 floor.
    """

    def belief(self, info: Infostate) -> HandBelief:
        n = info.pool_size
        mask = feasible_bid_mask(n)
        # The exact-rules conditional table only covers n ∈ {5..25}. For
        # smaller pools, fall back to uniform-over-feasible.
        if n < 5 or n > 25:
            return _uniform_belief(n)
        try:
            lookup = _get_lookup()
            _, _, cond_key = lookup.get_features(list(info.own_hand), n)
            cond = None
            get_exact_cond = getattr(lookup, "get_exact_rules_conditional", None)
            if get_exact_cond is not None and cond_key is not None:
                cond = get_exact_cond(n, cond_key)
            if cond is None:
                # No conditional row matched — use the marginal exact-rules table.
                cond = lookup.get_exact_rules_exact(n)
        except Exception:
            return _uniform_belief(n)
        if cond is None:
            return _uniform_belief(n)
        q_arr = np.asarray(cond, dtype=np.float32)
        if q_arr.shape[0] < NUM_BIDS:
            padded = np.zeros(NUM_BIDS, dtype=np.float32)
            padded[: q_arr.shape[0]] = q_arr
            q_arr = padded
        else:
            q_arr = q_arr[:NUM_BIDS]
        q, logits = _normalise(q_arr, mask)
        return HandBelief(q=q, q_logits=logits, feasible_mask=mask, n=n)

    def belief_batch(self, infos: list[Infostate]) -> list[HandBelief]:
        return [self.belief(i) for i in infos]


# ---------------------------------------------------------------------------
# HeuristicHandModel
# ---------------------------------------------------------------------------

class HeuristicHandModel:
    """Adapter over `ExactRulesOpponentModelAgent._compute_adj_exact`.

    Includes the heuristic ladder's opponent-bid bumps, so this is the
    "best closed-form prior with bid-history conditioning" point in the
    calibration plot.
    """

    def __init__(self) -> None:
        # Late import keeps the agents.registry side effects out of
        # tests that only need contracts.
        from agents.registry import ExactRulesOpponentModelAgent
        self._inner = ExactRulesOpponentModelAgent()

    def belief(self, info: Infostate) -> HandBelief:
        n = info.pool_size
        mask = feasible_bid_mask(n)
        if n < 5 or n > 25:
            return _uniform_belief(n)

        try:
            lookup     = _get_lookup()
            exact_prob = lookup.get_exact_rules_exact(n)
        except Exception:
            return _uniform_belief(n)
        if exact_prob is None:
            return _uniform_belief(n)

        cur_bid: Bid | None
        if info.standing_bid is None:
            cur_bid = None
        else:
            cur_bid = index_to_bid(info.standing_bid)

        adj = self._inner._compute_adj_exact(
            np.asarray(exact_prob, dtype=np.float32),
            lookup,
            list(info.own_hand),
            cur_bid,
            n,
        )
        if adj is None:
            return _uniform_belief(n)
        if adj.shape[0] < NUM_BIDS:
            padded = np.zeros(NUM_BIDS, dtype=np.float32)
            padded[: adj.shape[0]] = adj
            adj = padded
        else:
            adj = adj[:NUM_BIDS]
        q, logits = _normalise(adj.astype(np.float32), mask)
        return HandBelief(q=q, q_logits=logits, feasible_mask=mask, n=n)

    def belief_batch(self, infos: list[Infostate]) -> list[HandBelief]:
        return [self.belief(i) for i in infos]


__all__ = ["AnalyticHandModel", "HeuristicHandModel"]
