"""BidPolicy trainer — Phase 3 stub.

Skeleton fixes the optimizer wiring and `--load-trunk` plumbing so Phase 4 can
fill in the loss + dataset path without re-deciding scaffolding.

Phase 4 (design §5.3): forward KL `KL(target ‖ pi)` + entropy regularizer
`-β(n)·H(pi)` with `β(n) = β_max · max(0, 1 - n/5)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
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
    """Forward KL + schedule-dependent entropy regularizer per AR-2 §5.3.

    L = E[ -Σ_a target * log_pi(a) ]  -  β(n) · E[H(pi)]

    The constant target-entropy term is dropped (KL → cross-entropy form).
    `targets` is zero on infeasible actions (solver respects the same mask),
    so the `0 * -inf` contributions vanish exactly.
    """
    masked_logits = state.net(features, log_q, bid_mask)        # (B, NUM_BIDS)
    log_pi        = F.log_softmax(masked_logits, dim=-1)
    # Cross-entropy form. Where targets==0, the term contributes 0 even if
    # log_pi == -inf (PyTorch defines 0 * -inf as 0 in masked_fill paths
    # only if the zero is exact; we guard with masked_select to be safe).
    safe_log_pi = torch.where(torch.isfinite(log_pi), log_pi, torch.zeros_like(log_pi))
    ce = -(targets * safe_log_pi).sum(dim=-1)                   # (B,)

    pi = log_pi.exp()                                           # zeros on infeasible
    # Defensive clamp: H = -Σ pi * log_pi; on infeasible rows pi==0 and
    # log_pi==-inf → 0*-inf = NaN unless we clamp. Clamp log_pi to ≥ -30.
    log_pi_clamped = log_pi.clamp_min(-30.0)
    H = -(pi * log_pi_clamped).sum(dim=-1)                      # (B,)

    beta_max = state.net.config.beta_max
    beta = beta_max * (1.0 - pool_size.float() / 5.0).clamp_min(0.0)   # (B,)
    return (ce - beta * H).mean()


__all__ = ["BidPolicyTrainState", "build_train_state", "loss_step"]
