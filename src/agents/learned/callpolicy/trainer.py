"""CallPolicy trainer — Phase 3 stub.

Skeleton fixes the optimizer wiring and `--load-trunk` plumbing so Phase 4 can
fill in the loss + dataset path without re-deciding scaffolding. The loss body
is intentionally NotImplementedError until Phase 4 (design §5.2: BCE vs
`avg_call_prob`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import optim

from agents.learned.callpolicy.config import CallPolicyConfig
from agents.learned.callpolicy.network import CallPolicyNet
from agents.learned.handmodel.network import LearnedHandModel


@dataclass
class CallPolicyTrainState:
    net:       CallPolicyNet
    trunk:     LearnedHandModel
    optimizer: optim.Optimizer
    step:      int = 0


def build_train_state(config: CallPolicyConfig) -> CallPolicyTrainState:
    """Construct head + frozen trunk + optimizer per config."""
    if config.load_trunk is None:
        raise ValueError("CallPolicyConfig.load_trunk must point to an AR-1 HandModel checkpoint")
    trunk = LearnedHandModel.from_checkpoint(config.load_trunk, device=config.device)
    for p in trunk.net.parameters():
        p.requires_grad_(False)
    net = CallPolicyNet(config).to(config.device)
    optimizer = optim.AdamW(
        net.parameters(),                 # trunk params deliberately excluded
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    return CallPolicyTrainState(net=net, trunk=trunk, optimizer=optimizer, step=0)


def loss_step(
    state:    CallPolicyTrainState,
    features: torch.Tensor,    # (B, input_dim)
    targets:  torch.Tensor,    # (B,) avg_call_prob ∈ [0, 1] — soft labels
) -> torch.Tensor:
    """BCE-with-logits against soft targets per AR-2 §5.2."""
    logits = state.net._raw_logits(features)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


__all__ = ["CallPolicyTrainState", "build_train_state", "loss_step"]
