"""AR-2 Phase 3: smoke tests for CallPolicy / BidPolicy heads.

These tests do not exercise training (Phase 4) — they confirm:
- the heads construct,
- a forward pass produces a valid CallDecision / BidDistribution,
- the trunk is genuinely frozen (no grads on its params),
- HandModel byte-equivalence after the trunk_forward refactor (delegated to
  the existing AR-1 handmodel test suite).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agents.contracts import HandBelief, Infostate
from agents.learned.bidpolicy import BidPolicyConfig, BidPolicyNet, DistilledBidPolicy
from agents.learned.callpolicy import CallPolicyConfig, CallPolicyNet, DistilledCallPolicy
from agents.learned.handmodel.config import HandModelConfig
from agents.learned.handmodel.network import LearnedHandModel, LearnedHandModelNet
from game.bids import NUM_ACTIONS, NUM_BIDS
from game.feasibility import feasible_action_mask, feasible_bid_mask


def _tiny_handmodel(*, hidden_dim: int = 32) -> LearnedHandModel:
    """A small, untrained HandModel suitable for shape/wiring tests."""
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
    net = LearnedHandModelNet(cfg)
    return LearnedHandModel(net, device="cpu")


def _make_infostate(*, n: int = 5, standing_bid: int | None = 30) -> Infostate:
    legal = list(range(standing_bid + 1 if standing_bid is not None else 0, NUM_BIDS))
    if standing_bid is not None:
        legal.append(NUM_BIDS)        # CALL
    feasible_full = feasible_action_mask(n)
    legal_set = set(legal)
    joint = tuple(bool(feasible_full[a] and a in legal_set) for a in range(NUM_ACTIONS))
    return Infostate(
        own_hand=tuple([0, 13]),
        pool_size=n,
        hand_sizes=(2, 3),
        own_seat=0,
        current_player=0,
        standing_bid=standing_bid,
        bid_history=() if standing_bid is None else ((1, standing_bid),),
        legal_actions=tuple(legal),
        feasible_mask=joint,
        exact_rules=True,
        high_hand=True,
        five_kings=False,
    )


def _make_belief(info: Infostate) -> HandBelief:
    mask = feasible_bid_mask(info.pool_size)
    q = mask.astype(np.float32)
    q /= q.sum()
    logits = np.log(q + 1e-12).astype(np.float32)
    return HandBelief(q=q, q_logits=logits, feasible_mask=mask, n=info.pool_size)


# ---------------------------------------------------------------------------
# Trunk-forward parity (refactor guard)
# ---------------------------------------------------------------------------

def test_trunk_forward_matches_full_forward() -> None:
    """`forward` after refactor must equal `head(trunk_forward(...))` byte-identically."""
    torch.manual_seed(0)
    trunk = _tiny_handmodel(hidden_dim=32)
    info  = _make_infostate(n=5, standing_bid=30)
    batch = trunk._collate([info])

    with torch.no_grad():
        full   = trunk.net(**batch)
        hidden = trunk.net.trunk_forward(
            batch["own_hand"], batch["bid_tokens"], batch["seat_tokens"],
            batch["pos_tokens"], batch["hist_mask"], batch["scalars"],
        )
        manual = trunk.net.head(hidden).masked_fill(~batch["feasible"], float("-inf"))

    assert torch.equal(full, manual)


# ---------------------------------------------------------------------------
# CallPolicy smoke
# ---------------------------------------------------------------------------

def test_callpolicy_constructs_and_runs() -> None:
    trunk = _tiny_handmodel(hidden_dim=32)
    cfg   = CallPolicyConfig(trunk_dim=32, hidden=16)
    net   = CallPolicyNet(cfg)
    head  = DistilledCallPolicy(net, trunk)

    info = _make_infostate(n=5, standing_bid=30)
    q    = _make_belief(info)
    out  = head.call_prob(info, q)

    assert 0.0 <= out.p_call <= 1.0
    assert "n" in out.inputs
    assert "q_at_bid" in out.inputs


def test_callpolicy_dim_mismatch_raises() -> None:
    trunk = _tiny_handmodel(hidden_dim=32)
    cfg   = CallPolicyConfig(trunk_dim=64)   # mismatch
    net   = CallPolicyNet(cfg)
    with pytest.raises(ValueError, match="trunk_dim"):
        DistilledCallPolicy(net, trunk)


def test_callpolicy_input_dim() -> None:
    cfg = CallPolicyConfig(trunk_dim=256)
    assert cfg.input_dim == 478   # 256 + 110 + 110 + 2


# ---------------------------------------------------------------------------
# BidPolicy smoke
# ---------------------------------------------------------------------------

def test_bidpolicy_constructs_and_runs() -> None:
    trunk = _tiny_handmodel(hidden_dim=32)
    cfg   = BidPolicyConfig(trunk_dim=32, hidden=16)
    net   = BidPolicyNet(cfg)
    head  = DistilledBidPolicy(net, trunk)

    info = _make_infostate(n=5, standing_bid=30)
    q    = _make_belief(info)
    dist = head.bid_dist(info, q, hh_fired=False)

    # Contract: pi sums to 1, mass only on legal ∩ feasible bids.
    assert dist.support_size > 0
    assert abs(float(dist.pi.sum()) - 1.0) < 1e-5
    bid_mask = np.array(info.feasible_mask, dtype=np.bool_)
    assert float(dist.pi[~bid_mask].sum()) < 1e-6
    assert dist.entropy >= 0.0


def test_bidpolicy_warm_start_approximates_log_q() -> None:
    """At init (orthogonal gain 0.01), pi mode ≈ q mode and TV(pi, q) is small.

    Uses a peaked belief — uniform `q` makes the mode test meaningless because
    `softmax(log q)` is itself uniform.
    """
    torch.manual_seed(0)
    trunk = _tiny_handmodel(hidden_dim=32)
    cfg   = BidPolicyConfig(trunk_dim=32, hidden=16, final_init_gain=0.01)
    net   = BidPolicyNet(cfg)
    head  = DistilledBidPolicy(net, trunk)

    info = _make_infostate(n=5, standing_bid=None)
    mask = feasible_bid_mask(info.pool_size)

    # Peaked belief: bid 42 dominates by a wide margin.
    q_arr = mask.astype(np.float32) * 0.001
    q_arr[42] = 1.0
    q_arr = q_arr / q_arr.sum()
    belief = HandBelief(
        q=q_arr,
        q_logits=np.log(q_arr + 1e-12).astype(np.float32),
        feasible_mask=mask,
        n=info.pool_size,
    )
    dist = head.bid_dist(info, belief, hh_fired=False)

    bid_mask = np.array(info.feasible_mask, dtype=np.bool_)[:NUM_BIDS]
    expected = q_arr[:NUM_BIDS] * bid_mask
    expected = expected / expected.sum()
    actual   = dist.pi[:NUM_BIDS]

    assert int(actual.argmax()) == int(expected.argmax())
    tv = 0.5 * np.abs(actual - expected).sum()
    assert tv < 0.25, f"TV(actual, warm_start) = {tv:.3f} too large"


def test_bidpolicy_hh_fired_returns_empty() -> None:
    trunk = _tiny_handmodel(hidden_dim=32)
    cfg   = BidPolicyConfig(trunk_dim=32, hidden=16)
    net   = BidPolicyNet(cfg)
    head  = DistilledBidPolicy(net, trunk)

    info = _make_infostate(n=5, standing_bid=30)
    q    = _make_belief(info)
    dist = head.bid_dist(info, q, hh_fired=True)

    assert dist.support_size == 0
    assert float(dist.pi.sum()) == 0.0


def test_bidpolicy_input_dim() -> None:
    cfg = BidPolicyConfig(trunk_dim=256)
    assert cfg.input_dim == 367   # 256 + 110 + 1


# ---------------------------------------------------------------------------
# Trunk-freeze invariance (light version of design §7.5)
# ---------------------------------------------------------------------------

def test_trunk_params_frozen_after_wrap() -> None:
    trunk = _tiny_handmodel(hidden_dim=32)
    cfg   = CallPolicyConfig(trunk_dim=32)
    net   = CallPolicyNet(cfg)
    DistilledCallPolicy(net, trunk)
    assert all(not p.requires_grad for p in trunk.net.parameters())


def test_trunk_excluded_from_optimizer() -> None:
    """build_train_state should only register head params."""
    from agents.learned.callpolicy.trainer import build_train_state

    # Round-trip a tiny HandModel checkpoint to disk so the trainer's
    # `from_checkpoint` path is exercised.
    import tempfile

    trunk = _tiny_handmodel(hidden_dim=32)
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = f"{td}/handmodel.pt"
        trunk.save(ckpt_path, iter=0)
        cfg = CallPolicyConfig(trunk_dim=32, hidden=16, load_trunk=ckpt_path)
        state = build_train_state(cfg)

    optim_param_ids  = {id(p) for group in state.optimizer.param_groups for p in group["params"]}
    head_param_ids   = {id(p) for p in state.net.parameters()}
    trunk_param_ids  = {id(p) for p in state.trunk.net.parameters()}
    assert optim_param_ids == head_param_ids
    assert optim_param_ids.isdisjoint(trunk_param_ids)
