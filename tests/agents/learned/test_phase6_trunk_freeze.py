"""AR-2 Phase 6 §7.5 — trunk-freeze invariance after one training epoch.

Phase 3 already verifies trunk params have `requires_grad=False` at
construction. §7.5 is the stronger invariant: after the head completes a real
training loop (`loss.backward()` + `optimizer.step()` × N), the trunk
parameters' L2 norm is bitwise unchanged. Catches bugs like accidentally
registering trunk params with the optimizer or a future refactor that re-enables
trunk gradients without updating the freeze.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import optim

from agents.learned.bidpolicy import BidPolicyConfig, BidPolicyNet
from agents.learned.bidpolicy.trainer import BidPolicyTrainState, loss_step
from agents.learned.handmodel.config import HandModelConfig
from agents.learned.handmodel.network import LearnedHandModel, LearnedHandModelNet
from game.bids import NUM_BIDS


def _tiny_handmodel(*, hidden_dim: int = 32) -> LearnedHandModel:
    cfg = HandModelConfig(
        card_emb_dim=8,
        bid_emb_dim=8,
        bid_hist_len=4,
        bid_hist_dim=16,
        transformer_heads=2,
        transformer_ffn_dim=16,
        transformer_layers=1,
        hidden_dim=hidden_dim,
        num_trunk_layers=1,
        num_seats=2,
    )
    return LearnedHandModel(LearnedHandModelNet(cfg), device="cpu")


def _trunk_l2(trunk: LearnedHandModel) -> float:
    return sum(p.data.norm(p=2).item() ** 2 for p in trunk.net.parameters()) ** 0.5


def test_trunk_freeze_after_epoch() -> None:
    """50 BidPolicy `loss_step` + `optimizer.step()` iters do not move trunk params."""
    torch.manual_seed(0)
    trunk = _tiny_handmodel(hidden_dim=32)
    cfg = BidPolicyConfig(trunk_dim=32, hidden=64, load_trunk=None, device="cpu")
    net = BidPolicyNet(cfg)
    opt = optim.AdamW(net.parameters(), lr=1e-3)
    state = BidPolicyTrainState(net=net, trunk=trunk, optimizer=opt, step=0)

    before = _trunk_l2(trunk)

    # Synthetic batch fixed across iterations (a real epoch behaves the same way).
    B = 32
    feats     = torch.randn(B, cfg.input_dim)
    log_q     = torch.randn(B, NUM_BIDS)
    bid_mask  = torch.ones(B, NUM_BIDS, dtype=torch.bool)
    targets   = F.softmax(torch.randn(B, NUM_BIDS), dim=-1)
    pool_size = torch.full((B,), 5, dtype=torch.long)   # β=0 → pure KL loss

    for _ in range(50):
        opt.zero_grad()
        loss = loss_step(state, feats, log_q, bid_mask, targets, pool_size)
        loss.backward()
        opt.step()

    after = _trunk_l2(trunk)

    assert before == after, (
        f"trunk L2 norm drifted after 50 gradient steps: "
        f"before={before!r}, after={after!r}, delta={after - before:.3e}"
    )

    # Belt-and-braces: also verify per-param exact equality on grad.
    for p in trunk.net.parameters():
        assert p.grad is None or p.grad.abs().sum().item() == 0.0, (
            "trunk parameter accumulated a nonzero gradient"
        )
