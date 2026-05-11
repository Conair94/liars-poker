"""AR-2 Phase 6 §7.2 — entropy-floor property test at n=2.

1000 random infostates at `pool_size = 2`, run through
`DistilledBidPolicy.bid_dist()`. Every output must satisfy
`entropy >= floor_frac[2] · log(feasible_count)`.

Tests the full inference path (network → masked softmax →
`apply_entropy_floor`), not the floor helper in isolation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from agents.contracts import HandBelief, Infostate
from agents.learned.bidpolicy import BidPolicyConfig, BidPolicyNet, DistilledBidPolicy
from agents.learned.handmodel.config import HandModelConfig
from agents.learned.handmodel.network import LearnedHandModel, LearnedHandModelNet
from game.bids import CALL_ACTION, NUM_ACTIONS, NUM_BIDS
from game.feasibility import feasible_action_mask, feasible_bid_mask

_POOL_SIZE = 2
_N_DEALS   = 1000


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


def _make_infostate(standing_bid: int | None, pool_size: int) -> Infostate:
    if standing_bid is None:
        legal = list(range(NUM_BIDS))
    else:
        legal = list(range(standing_bid + 1, NUM_BIDS)) + [CALL_ACTION]
    feas  = feasible_action_mask(pool_size)
    joint = tuple(bool(feas[a] and a in set(legal)) for a in range(NUM_ACTIONS))
    bid_history: tuple[tuple[int, int], ...] = (
        () if standing_bid is None else ((0, standing_bid),)
    )
    own_seat = 0 if standing_bid is None else 1
    return Infostate(
        own_hand       = (0,),
        pool_size      = pool_size,
        hand_sizes     = (1, 1),
        own_seat       = own_seat,
        current_player = own_seat,
        standing_bid   = standing_bid,
        bid_history    = bid_history,
        legal_actions  = tuple(legal),
        feasible_mask  = joint,
        exact_rules    = True,
        high_hand      = True,
        five_kings     = False,
    )


def _uniform_belief(pool_size: int) -> HandBelief:
    mask   = feasible_bid_mask(pool_size)
    q      = mask.astype(np.float32) / float(mask.sum())
    logits = np.log(q + 1e-12).astype(np.float32)
    return HandBelief(q=q, q_logits=logits, feasible_mask=mask, n=pool_size)


@pytest.mark.slow
def test_entropy_floor_n2_property() -> None:
    """`DistilledBidPolicy` at n=2: every output meets the entropy floor."""
    trunk = _tiny_handmodel(hidden_dim=32)
    cfg = BidPolicyConfig(trunk_dim=32, hidden=64, load_trunk=None, device="cpu")
    net = BidPolicyNet(cfg)
    bid_policy = DistilledBidPolicy(net=net, trunk=trunk, device="cpu")

    rng = np.random.default_rng(42)
    belief = _uniform_belief(_POOL_SIZE)
    feasible_count = int(np.array(belief.feasible_mask, dtype=np.bool_).sum())
    assert feasible_count > 1
    h_target = cfg.floor_frac[_POOL_SIZE] * math.log(feasible_count)

    failures: list[tuple[int, float]] = []
    for i in range(_N_DEALS):
        if i < _N_DEALS // 2:
            standing_bid: int | None = None       # opener
        else:
            # mid-round: any HC bid (HC bids are 0..12 at n=2) that leaves
            # at least 2 legal+feasible bids so the entropy floor is meaningful.
            standing_bid = int(rng.integers(0, 11))   # 0..10 inclusive
        info  = _make_infostate(standing_bid, _POOL_SIZE)

        feasible_bids = np.array(info.feasible_mask[:NUM_BIDS], dtype=np.bool_)
        feas_n = int(feasible_bids.sum())
        if feas_n <= 1:
            continue   # degenerate support — no floor applies

        dist = bid_policy.bid_dist(info, belief, hh_fired=False)

        local_target = cfg.floor_frac[_POOL_SIZE] * math.log(feas_n)
        if dist.entropy < local_target - 1e-3:
            failures.append((i, dist.entropy))

    assert not failures, (
        f"{len(failures)}/{_N_DEALS} infostates failed the entropy floor "
        f"(target ≈ {h_target:.4f} nats); first failure: {failures[0]}"
    )
