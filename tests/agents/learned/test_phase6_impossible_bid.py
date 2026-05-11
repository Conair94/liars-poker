"""AR-2 Phase 6 §7.1 — impossible-bid deal → `p_call > 0.95`.

Two assertions:

1. The CFR+ solver assigns `avg_call_prob ≈ 1.0` at an infeasible-bid state.
2. A `CallPolicyNet` mini-trained on that solver label converges to
   `p_call > 0.95` — confirming the (features → head → BCE loss) pipeline
   transmits the obvious signal.

No checkpoint is required: the trunk is a randomly-initialised
`_tiny_handmodel`, kept frozen.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from agents.contracts import HandBelief, Infostate
from agents.learned.callpolicy import (
    CallPolicyConfig,
    CallPolicyNet,
    DistilledCallPolicy,
)
from agents.learned.callpolicy.network import _trunk_forward, build_call_features
from agents.learned.callpolicy.trainer import CallPolicyTrainState, loss_step
from agents.learned.handmodel.config import HandModelConfig
from agents.learned.handmodel.network import LearnedHandModel, LearnedHandModelNet
from game.bids import CALL_ACTION, NUM_ACTIONS, NUM_BIDS
from game.feasibility import feasible_action_mask, feasible_bid_mask
from training.cfr.subgame_solver import CFRPlusSubgameSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_POOL_SIZE = 4   # 2 cards per seat × 2 seats — STRAIGHT bids (require 5) are infeasible
_HANDS     = ([0, 1], [2, 3])


def _tiny_handmodel(*, hidden_dim: int = 32) -> LearnedHandModel:
    """Small, untrained HandModel. Matches the helper in test_phase3_smoke.py."""
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


def _find_first_infeasible_bid(pool_size: int) -> int:
    """First bid index `i` with `feasible_bid_mask(pool_size)[i] == False`."""
    mask = feasible_bid_mask(pool_size)
    for i in range(NUM_BIDS):
        if not mask[i]:
            return i
    raise AssertionError(f"no infeasible bid at pool_size={pool_size}")


def _make_infostate_at_standing(standing_bid: int, pool_size: int) -> Infostate:
    """Infostate for P1's decision after P0 opened with `standing_bid`."""
    legal = list(range(standing_bid + 1, NUM_BIDS)) + [CALL_ACTION]
    feas  = feasible_action_mask(pool_size)
    joint = tuple(bool(feas[a] and a in set(legal)) for a in range(NUM_ACTIONS))
    return Infostate(
        own_hand       = (2, 3),
        pool_size      = pool_size,
        hand_sizes     = (2, 2),
        own_seat       = 1,
        current_player = 1,
        standing_bid   = standing_bid,
        bid_history    = ((0, standing_bid),),
        legal_actions  = tuple(legal),
        feasible_mask  = joint,
        exact_rules    = True,
        high_hand      = True,
        five_kings     = False,
    )


def _belief_uniform_over_feasible(pool_size: int) -> HandBelief:
    mask   = feasible_bid_mask(pool_size)
    q      = mask.astype(np.float32) / float(mask.sum())
    logits = np.log(q + 1e-12).astype(np.float32)
    return HandBelief(q=q, q_logits=logits, feasible_mask=mask, n=pool_size)


# ---------------------------------------------------------------------------
# Test 1 — solver target at the impossible-bid state
# ---------------------------------------------------------------------------

def test_impossible_bid_solver_target() -> None:
    """CFR+ must learn that calling an infeasible bid wins with certainty."""
    standing_bid = _find_first_infeasible_bid(_POOL_SIZE)
    solver = CFRPlusSubgameSolver(max_iters=500, eps=1e-4, seed=0)
    solution = solver.solve(_HANDS)

    key = (standing_bid, 1)   # P1's turn after P0 opens with standing_bid
    assert key in solution.avg_call_prob, (
        f"state {key} not visited by solver — strat_sum was zero"
    )
    assert solution.avg_call_prob[key] >= 0.98, (
        f"avg_call_prob[{key}] = {solution.avg_call_prob[key]:.4f}, expected ≥ 0.98"
    )


# ---------------------------------------------------------------------------
# Test 2 — mini-distillation reaches p_call > 0.95
# ---------------------------------------------------------------------------

def test_impossible_bid_mini_distillation() -> None:
    """A CallPolicyNet mini-trained on the solver label hits p_call > 0.95."""
    standing_bid = _find_first_infeasible_bid(_POOL_SIZE)

    # Solver label.
    solver = CFRPlusSubgameSolver(max_iters=500, eps=1e-4, seed=0)
    solution = solver.solve(_HANDS)
    target = float(solution.avg_call_prob[(standing_bid, 1)])
    assert target >= 0.98   # precondition; Test 1 covers it independently.

    # Components: random trunk + fresh head.
    trunk = _tiny_handmodel(hidden_dim=32)
    cfg = CallPolicyConfig(trunk_dim=32, hidden=64, load_trunk=None, device="cpu")
    net = CallPolicyNet(cfg)
    opt = optim.AdamW(net.parameters(), lr=1e-3)
    state = CallPolicyTrainState(net=net, trunk=trunk, optimizer=opt, step=0)

    # Features for the single impossible-bid infostate.
    info   = _make_infostate_at_standing(standing_bid, _POOL_SIZE)
    belief = _belief_uniform_over_feasible(_POOL_SIZE)
    assert belief.q[standing_bid] == 0.0   # infeasible → zero mass

    trunk_repr_t = _trunk_forward(trunk, [info], device=torch.device("cpu"))   # (1, 32)
    feats_np = build_call_features(
        trunk_repr_t.cpu().numpy(),
        belief.q[None, :],
        np.array([standing_bid], dtype=np.int64),
        np.array([_POOL_SIZE], dtype=np.int64),
    )
    feats   = torch.from_numpy(feats_np).expand(32, -1).contiguous()
    targets = torch.full((32,), target, dtype=torch.float32)

    # Mini-distillation loop.
    for _ in range(300):
        opt.zero_grad()
        loss = loss_step(state, feats, targets)
        loss.backward()
        opt.step()

    distilled = DistilledCallPolicy(net=net, trunk=trunk, device="cpu")
    decision  = distilled.call_prob(info, belief)
    assert decision.p_call > 0.95, (
        f"after 300 mini-distillation steps, p_call = {decision.p_call:.4f}"
    )
