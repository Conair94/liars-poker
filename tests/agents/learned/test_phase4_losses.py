"""AR-2 Phase 4 — loss-body tests + trunk-freeze invariance (§7.5)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_SRC  = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agents.learned.bidpolicy import trainer as bp_trainer
from agents.learned.bidpolicy.config import BidPolicyConfig
from agents.learned.callpolicy import trainer as cp_trainer
from agents.learned.callpolicy.config import CallPolicyConfig
from game.bids import NUM_BIDS

_CKPT = os.path.join(
    _REPO, "data/runs/ar1-20260430T030730Z-b64-h256-n2-fb17d031/handmodel/best.pt"
)


def _maybe_skip_no_trunk():
    if not os.path.exists(_CKPT):
        pytest.skip(f"AR-1 trunk checkpoint not present at {_CKPT}")


# ---------------------------------------------------------------------------
# CallPolicy
# ---------------------------------------------------------------------------

def test_callpolicy_loss_decreases():
    _maybe_skip_no_trunk()
    torch.manual_seed(0)
    cfg = CallPolicyConfig(load_trunk=_CKPT, lr=1e-2)
    state = cp_trainer.build_train_state(cfg)

    B = 256
    feats = torch.randn(B, cfg.input_dim)
    targets = torch.rand(B)

    init_loss = float(cp_trainer.loss_step(state, feats, targets).detach())
    for _ in range(50):
        state.optimizer.zero_grad()
        loss = cp_trainer.loss_step(state, feats, targets)
        loss.backward()
        state.optimizer.step()
    final_loss = float(loss.detach())
    assert final_loss < init_loss, f"call loss did not decrease: {init_loss:.4f} -> {final_loss:.4f}"


# ---------------------------------------------------------------------------
# BidPolicy
# ---------------------------------------------------------------------------

def _synthetic_bid_batch(B: int, cfg: BidPolicyConfig, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(B, cfg.input_dim, generator=g)
    log_q = torch.log(torch.softmax(torch.randn(B, NUM_BIDS, generator=g), dim=-1) + 1e-12)
    # Random feasible mask: each row has ~50% feasible bids.
    bid_mask = torch.rand(B, NUM_BIDS, generator=g) > 0.5
    # Guarantee at least 2 feasible bids per row.
    for i in range(B):
        if bid_mask[i].sum() < 2:
            bid_mask[i, :2] = True
    # Random target: softmax over feasible.
    raw = torch.randn(B, NUM_BIDS, generator=g)
    raw = raw.masked_fill(~bid_mask, float("-inf"))
    targets = torch.softmax(raw, dim=-1)
    targets = torch.nan_to_num(targets, nan=0.0)
    pool_size = torch.randint(2, 11, (B,), generator=g)
    return feats, log_q, bid_mask, targets, pool_size


def test_bidpolicy_loss_decreases():
    _maybe_skip_no_trunk()
    torch.manual_seed(0)
    cfg = BidPolicyConfig(load_trunk=_CKPT, lr=1e-2)
    state = bp_trainer.build_train_state(cfg)
    feats, log_q, bid_mask, targets, pool_size = _synthetic_bid_batch(128, cfg, seed=0)

    init_loss = float(bp_trainer.loss_step(state, feats, log_q, bid_mask, targets, pool_size).detach())
    for _ in range(50):
        state.optimizer.zero_grad()
        loss = bp_trainer.loss_step(state, feats, log_q, bid_mask, targets, pool_size)
        loss.backward()
        state.optimizer.step()
    final_loss = float(loss.detach())
    assert final_loss < init_loss, f"bid loss did not decrease: {init_loss:.4f} -> {final_loss:.4f}"


def test_bidpolicy_no_nan_on_infeasible():
    """All-zero target on infeasible actions must produce finite gradients."""
    _maybe_skip_no_trunk()
    torch.manual_seed(0)
    cfg = BidPolicyConfig(load_trunk=_CKPT)
    state = bp_trainer.build_train_state(cfg)
    feats, log_q, bid_mask, targets, pool_size = _synthetic_bid_batch(64, cfg, seed=1)

    state.optimizer.zero_grad()
    loss = bp_trainer.loss_step(state, feats, log_q, bid_mask, targets, pool_size)
    loss.backward()
    assert torch.isfinite(loss).item()
    for p in state.net.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all().item(), "non-finite gradient"


# ---------------------------------------------------------------------------
# Trunk-freeze invariance (§7.5)
# ---------------------------------------------------------------------------

def _trunk_param_state(trunk) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in trunk.net.state_dict().items()}


def _assert_trunk_unchanged(before: dict, trunk) -> None:
    after = trunk.net.state_dict()
    for k, v_before in before.items():
        v_after = after[k]
        assert torch.equal(v_before, v_after), f"trunk param '{k}' changed after training step"


def test_trunk_freeze_invariance_callpolicy():
    _maybe_skip_no_trunk()
    torch.manual_seed(0)
    cfg = CallPolicyConfig(load_trunk=_CKPT, lr=1e-2)
    state = cp_trainer.build_train_state(cfg)
    snap = _trunk_param_state(state.trunk)

    feats = torch.randn(128, cfg.input_dim)
    targets = torch.rand(128)
    for _ in range(10):
        state.optimizer.zero_grad()
        loss = cp_trainer.loss_step(state, feats, targets)
        loss.backward()
        state.optimizer.step()

    _assert_trunk_unchanged(snap, state.trunk)


def test_trunk_freeze_invariance_bidpolicy():
    _maybe_skip_no_trunk()
    torch.manual_seed(0)
    cfg = BidPolicyConfig(load_trunk=_CKPT, lr=1e-2)
    state = bp_trainer.build_train_state(cfg)
    snap = _trunk_param_state(state.trunk)

    feats, log_q, bid_mask, targets, pool_size = _synthetic_bid_batch(64, cfg, seed=2)
    for _ in range(10):
        state.optimizer.zero_grad()
        loss = bp_trainer.loss_step(state, feats, log_q, bid_mask, targets, pool_size)
        loss.backward()
        state.optimizer.step()

    _assert_trunk_unchanged(snap, state.trunk)


# Silence unused-import warnings on platforms missing the trunk checkpoint.
_ = np
