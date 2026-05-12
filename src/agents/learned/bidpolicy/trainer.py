"""BidPolicy trainer — Phase 3 stub.

Skeleton fixes the optimizer wiring and `--load-trunk` plumbing so Phase 4 can
fill in the loss + dataset path without re-deciding scaffolding.

Phase 4 (design §5.3): forward KL `KL(target ‖ pi)` + entropy regularizer
`-β(n)·H(pi)` with `β(n) = β_max · max(0, 1 - n/5)`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agents.learned.bidpolicy.config import BidPolicyConfig
from agents.learned.bidpolicy.network import BidPolicyNet, build_bid_features
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


def _batch_tensors(
    batch:  dict[str, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (features, log_q, bid_mask, targets, pool_size) from an iter_shards batch."""
    feats = build_bid_features(
        batch["trunk_repr"].astype(np.float32),
        batch["q"].astype(np.float32),
        batch["pool_size"].astype(np.int64),
    )
    log_q     = np.log(batch["q"].astype(np.float32) + 1e-12)
    bid_mask  = batch["feasible_mask"].astype(np.bool_)
    targets   = batch["target_bid"].astype(np.float32)
    pool_size = batch["pool_size"].astype(np.int64)
    return (
        torch.from_numpy(feats).to(device).float(),
        torch.from_numpy(log_q).to(device).float(),
        torch.from_numpy(bid_mask).to(device).bool(),
        torch.from_numpy(targets).to(device).float(),
        torch.from_numpy(pool_size).to(device).long(),
    )


def _eval_val_kl(
    state:      BidPolicyTrainState,
    val_run_id: str,
    *,
    splits:     list[str],
    batch_size: int,
    device:     torch.device,
    data_root:  str | None,
) -> dict:
    """Compute val_ce, val_kl, and val_kl_per_n over the requested splits."""
    from training.cfr_distillation import iter_shards
    state.net.eval()
    total_ce  = 0.0
    total_target_entropy = 0.0
    n_rows = 0
    per_n_ce: dict[int, float] = {}
    per_n_target_entropy: dict[int, float] = {}
    per_n_count: dict[int, int] = {}

    with torch.no_grad():
        for split in splits:
            for batch in iter_shards(
                val_run_id, split, head="bid",
                data_root=data_root, batch_size=batch_size,
            ):
                feats, log_q, bid_mask, targets, pool_size = _batch_tensors(batch, device)
                masked_logits = state.net(feats, log_q, bid_mask)
                log_pi = F.log_softmax(masked_logits, dim=-1)
                safe_log_pi = torch.where(
                    torch.isfinite(log_pi), log_pi, torch.zeros_like(log_pi),
                )
                ce = -(targets * safe_log_pi).sum(dim=-1)  # (B,)
                # Target entropy (constant in data; KL = CE - H(target)).
                safe_log_t = torch.where(
                    targets > 0,
                    torch.log(targets.clamp_min(1e-12)),
                    torch.zeros_like(targets),
                )
                th = -(targets * safe_log_t).sum(dim=-1)   # (B,)

                ce_np = ce.cpu().numpy()
                th_np = th.cpu().numpy()
                pn_np = pool_size.cpu().numpy()
                total_ce += float(ce_np.sum())
                total_target_entropy += float(th_np.sum())
                n_rows += int(ce_np.shape[0])
                for n_val in (2, 4, 6, 8, 10):
                    mask = (pn_np == n_val)
                    if mask.any():
                        per_n_ce[n_val] = per_n_ce.get(n_val, 0.0) + float(ce_np[mask].sum())
                        per_n_target_entropy[n_val] = per_n_target_entropy.get(n_val, 0.0) + float(th_np[mask].sum())
                        per_n_count[n_val] = per_n_count.get(n_val, 0) + int(mask.sum())

    state.net.train()
    val_ce = total_ce / max(1, n_rows)
    target_entropy_mean = total_target_entropy / max(1, n_rows)
    val_kl = val_ce - target_entropy_mean
    val_kl_per_n: dict[str, float | None] = {}
    for n_val in (2, 4, 6, 8, 10):
        cnt = per_n_count.get(n_val, 0)
        if cnt == 0:
            val_kl_per_n[str(n_val)] = None
        else:
            ce_n = per_n_ce[n_val] / cnt
            te_n = per_n_target_entropy[n_val] / cnt
            val_kl_per_n[str(n_val)] = ce_n - te_n
    return {
        "val_ce":         val_ce,
        "val_kl":         val_kl,
        "val_kl_per_n":   val_kl_per_n,
    }


def _atomic_save(payload: dict, path: str) -> None:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="AR-2 Phase 7 BidPolicy trainer")
    p.add_argument("--run-id",     type=str, required=True)
    p.add_argument("--load-trunk", type=str, required=True)
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--out-dir",    type=str, default=None)
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--external-val-run-id", type=str, default=None)
    p.add_argument("--data-root",  type=str, default=None)
    p.add_argument("--config",     type=str, default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    from training.cfr_distillation import iter_shards

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = args.data_root
    if data_root is None:
        data_root = os.path.abspath(os.path.join(_SRC, "..", "data"))
    out_dir = args.out_dir or os.path.join(
        data_root, "runs", args.run_id, "bidpolicy"
    )
    os.makedirs(out_dir, exist_ok=True)

    config = BidPolicyConfig(
        load_trunk=args.load_trunk,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
    )
    state = build_train_state(config)
    device = torch.device(args.device)

    if args.external_val_run_id:
        val_run_id = args.external_val_run_id
        val_splits = ["val", "test"]
    else:
        val_run_id = args.run_id
        val_splits = ["val"]

    curve: list[dict] = []
    best_val: float | None = None
    t_start = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        state.net.train()
        train_loss_sum = 0.0
        n_batches = 0
        for batch in iter_shards(
            args.run_id, "train", head="bid",
            data_root=data_root, batch_size=args.batch_size,
        ):
            feats, log_q, bid_mask, targets, pool_size = _batch_tensors(batch, device)
            state.optimizer.zero_grad()
            loss = loss_step(state, feats, log_q, bid_mask, targets, pool_size)
            loss.backward()
            state.optimizer.step()
            train_loss_sum += float(loss.item())
            n_batches += 1
            state.step += 1

        train_loss = train_loss_sum / max(1, n_batches)
        val_metrics = _eval_val_kl(
            state, val_run_id, splits=val_splits,
            batch_size=args.batch_size, device=device, data_root=data_root,
        )
        wall = time.monotonic() - t_start
        curve.append({
            "epoch":        epoch,
            "train_loss":   train_loss,
            "val_ce":       val_metrics["val_ce"],
            "val_kl":       val_metrics["val_kl"],
            "val_kl_per_n": val_metrics["val_kl_per_n"],
            "wall_clock_s": wall,
        })
        with open(os.path.join(out_dir, "training_curve.json"), "w") as f:
            json.dump(curve, f, indent=2)

        if best_val is None or val_metrics["val_kl"] < best_val:
            best_val = val_metrics["val_kl"]
            _atomic_save(
                {
                    "state_dict":     state.net.state_dict(),
                    "config_dict":    config.to_dict(),
                    "val_kl":         val_metrics["val_kl"],
                    "val_kl_per_n":   val_metrics["val_kl_per_n"],
                    "epoch":          epoch,
                },
                os.path.join(out_dir, "best.pt"),
            )

        print(f"[bidpolicy] epoch={epoch} train_loss={train_loss:.4f} "
              f"val_kl={val_metrics['val_kl']:.4f} best={best_val:.4f}", flush=True)


if __name__ == "__main__":
    _main()


__all__ = ["BidPolicyTrainState", "build_train_state", "loss_step", "_main"]
