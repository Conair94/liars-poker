"""CFR+ subgame solver and shared bidding-tree primitives (AR-2 §2)."""

from training.cfr.subgame_solver import (
    CFRPlusSubgameSolver,
    SubgameSolution,
    build_canonical_match_state,
    legal_subgame_actions,
    pool_best_bid_idx,
    resolve_call_returns,
    resolve_hh_returns,
)

__all__ = [
    "CFRPlusSubgameSolver",
    "SubgameSolution",
    "build_canonical_match_state",
    "legal_subgame_actions",
    "pool_best_bid_idx",
    "resolve_call_returns",
    "resolve_hh_returns",
]
