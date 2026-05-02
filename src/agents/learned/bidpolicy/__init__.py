"""BidPolicy — AR-2 distillation head producing π(action | info, q, ¬call)."""

from agents.learned.bidpolicy.config import BidPolicyConfig
from agents.learned.bidpolicy.network import (
    BidPolicyNet,
    DistilledBidPolicy,
    build_bid_features,
)

__all__ = [
    "BidPolicyConfig",
    "BidPolicyNet",
    "DistilledBidPolicy",
    "build_bid_features",
]
