"""BidPolicy trainer — Phase 3 stub.

Skeleton fixes the optimizer wiring and `--load-trunk` plumbing so Phase 4 can
fill in the loss + dataset path without re-deciding scaffolding.

Phase 4 (design §5.3): forward KL `KL(target ‖ pi)` + entropy regularizer
`-β(n)·H(pi)` with `β(n) = β_max · max(0, 1 - n/5)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import optim

from agents.learned.bidpolicy.config import BidPolicyConfig
from agents.learned.bidpolicy.network import BidPolicyNet
from agents.learned.handmodel.network import LearnedHandModel


@dataclass
class BidPolicyTrainState:
    net:       BidPolicyNet
    trunk:     LearnedHandModel
    optimizer: optim.Optimizer
    step:      int = 0


def build_train_state(config: BidPolicyConfig) -> BidPolicyTrainState:
    """Construct head + frozen trunk + optimizer per config."""
    if config.load_trunk is None:
        raise ValueError("BidPolicyConfig.load_trunk must point to an AR-1 HandModel checkpoint")
    trunk = LearnedHandModel.from_checkpoint(config.load_trunk, device=config.device)
    for p in trunk.net.parameters():
        p.requires_grad_(False)
    net = BidPolicyNet(config).to(config.device)
    optimizer = optim.AdamW(
        net.parameters(),                 # trunk params deliberately excluded
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    return BidPolicyTrainState(net=net, trunk=trunk, optimizer=optimizer, step=0)


def loss_step(
    state:    BidPolicyTrainState,
    features: torch.Tensor,    # (B, input_dim)
    log_q:    torch.Tensor,    # (B, NUM_BIDS)
    bid_mask: torch.Tensor,    # (B, NUM_BIDS) bool
    targets:  torch.Tensor,    # (B, NUM_BIDS) — CFR+ avg policy over bids (rows sum to 1)
    pool_size: torch.Tensor,   # (B,) int — for β(n) schedule
) -> torch.Tensor:
    """Phase 4 home: forward KL + entropy regularizer per design §5.3."""
    raise NotImplementedError("BidPolicy loss lands in AR-2 Phase 4 (design §5.3)")


__all__ = ["BidPolicyTrainState", "build_train_state", "loss_step"]


# Silence unused-import warnings on the stubbed-out path.
_ = nn
