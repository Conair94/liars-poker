"""AR-2 Phase 4 — distillation pipeline smoke + KL-reduction test (§7.6)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC  = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from training.cfr_distillation import (
    sample_deals_mixture,
    walk_avg_strategy,
)

_CKPT = os.path.join(
    _REPO, "data/runs/ar1-20260430T030730Z-b64-h256-n2-fb17d031/handmodel/best.pt"
)


def _maybe_skip_no_trunk():
    if not os.path.exists(_CKPT):
        pytest.skip(f"AR-1 trunk checkpoint not present at {_CKPT}")


# ---------------------------------------------------------------------------
# Cheap tests
# ---------------------------------------------------------------------------

def test_mixture_sampler_counts():
    deals = list(sample_deals_mixture(400, seed=0))
    assert len(deals) == 400
    counts = {2: 0, 3: 0, 4: 0, 5: 0}
    for hands, _n in deals:
        counts[len(hands[0])] += 1
    assert counts == {2: 100, 3: 100, 4: 100, 5: 100}, counts

    # Determinism in seed.
    again = list(sample_deals_mixture(400, seed=0))
    for (h1, _), (h2, _) in zip(deals, again, strict=True):
        assert h1 == h2


def test_walk_avg_strategy_basic():
    """One n=4 deal yields ≥1 CallRow, ≥1 BidRow; bid targets sum to 1."""
    from training.cfr.subgame_solver import CFRPlusSubgameSolver
    deals = list(sample_deals_mixture(4, seed=0))
    hands, _n = deals[0]
    sol = CFRPlusSubgameSolver(max_iters=100).solve(hands)
    rows = list(walk_avg_strategy(sol, hands, deal_idx=0))
    assert len(rows) >= 1
    bid_rows = [r for r in rows if r.target_bid is not None]
    assert len(bid_rows) >= 1
    for r in bid_rows:
        s = float(r.target_bid.sum())
        assert abs(s - 1.0) < 1e-4, f"target_bid sum {s}"
        # All mass on feasible.
        infeasible = ~r.feasible_mask
        assert float(r.target_bid[infeasible].sum()) < 1e-6


# ---------------------------------------------------------------------------
# End-to-end smoke (slow): pipeline produces shards + KL drops
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_pipeline_end_to_end_and_kl_reduction(tmp_path):
    _maybe_skip_no_trunk()
    import torch

    from agents.learned.bidpolicy import trainer as bp_trainer
    from agents.learned.bidpolicy.config import BidPolicyConfig
    from training.cfr_distillation import run_distillation

    summary = run_distillation(
        N=32,
        seed=0,
        run_id="phase4_smoke_test",
        trunk_ckpt=_CKPT,
        max_iters=200,
        shard_count=4,
        device="cpu",
        data_root=str(tmp_path),
    )

    out_dir = summary["out_dir"]
    bid_files   = sorted(f for f in os.listdir(out_dir) if f.startswith("bid_"))
    trunk_files = sorted(f for f in os.listdir(out_dir) if f.startswith("trunk_"))
    call_files  = sorted(f for f in os.listdir(out_dir) if f.startswith("call_"))
    assert bid_files and trunk_files and call_files
    assert len(bid_files) == len(trunk_files), "bid and trunk shard counts must match"

    # Aggregate all bid + trunk rows into a single training tensor.
    feats_all   = []
    log_q_all   = []
    mask_all    = []
    targets_all = []
    pool_all    = []
    for bf, tf in zip(bid_files, trunk_files, strict=True):
        b = np.load(os.path.join(out_dir, bf))
        t = np.load(os.path.join(out_dir, tf))
        assert b["target_bid"].shape[0] == t["trunk_repr"].shape[0], (bf, tf)
        n_scalar = (b["pool_size"].astype(np.float32) / 25.0)[:, None]
        feats = np.concatenate([t["trunk_repr"], t["q"], n_scalar], axis=1)
        log_q = np.log(t["q"] + 1e-12)
        feats_all.append(feats)
        log_q_all.append(log_q)
        mask_all.append(b["feasible_mask"])
        targets_all.append(b["target_bid"])
        pool_all.append(b["pool_size"])

    feats_t   = torch.from_numpy(np.concatenate(feats_all)).float()
    log_q_t   = torch.from_numpy(np.concatenate(log_q_all)).float()
    mask_t    = torch.from_numpy(np.concatenate(mask_all)).bool()
    targets_t = torch.from_numpy(np.concatenate(targets_all)).float()
    pool_t    = torch.from_numpy(np.concatenate(pool_all)).long()

    # Build a fresh BidPolicy + train for a few epochs; expect cross-entropy
    # to drop substantially below the log-q warm-start init.
    torch.manual_seed(0)
    cfg = BidPolicyConfig(load_trunk=_CKPT, lr=3e-3)
    state = bp_trainer.build_train_state(cfg)

    def kl_only(state) -> float:
        """Pure cross-entropy term (no entropy regularizer)."""
        with torch.no_grad():
            masked_logits = state.net(feats_t, log_q_t, mask_t)
            log_pi = torch.log_softmax(masked_logits, dim=-1)
            log_pi = torch.where(torch.isfinite(log_pi), log_pi, torch.zeros_like(log_pi))
            ce = -(targets_t * log_pi).sum(dim=-1).mean()
        return float(ce)

    init_kl = kl_only(state)

    # Train: 5 passes over the full (small) batch.
    for _ in range(50):
        state.optimizer.zero_grad()
        loss = bp_trainer.loss_step(
            state, feats_t, log_q_t, mask_t, targets_t, pool_t,
        )
        loss.backward()
        state.optimizer.step()

    final_kl = kl_only(state)
    print(f"phase4 KL: init={init_kl:.4f} final={final_kl:.4f}")
    assert final_kl < 0.5 * init_kl, (
        f"§7.6 KL-reduction gate failed: init={init_kl:.4f} final={final_kl:.4f}"
    )
