"""CallPolicy trainer — Phase 3 stub.

Skeleton fixes the optimizer wiring and `--load-trunk` plumbing so Phase 4 can
fill in the loss + dataset path without re-deciding scaffolding. The loss body
is intentionally NotImplementedError until Phase 4 (design §5.2: BCE vs
`avg_call_prob`).
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

from agents.learned.callpolicy.config import CallPolicyConfig
from agents.learned.callpolicy.network import CallPolicyNet, build_call_features
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


def _batch_features(batch: dict[str, np.ndarray]) -> np.ndarray:
    """Build the 478-d call features from an iter_shards batch."""
    return build_call_features(
        batch["trunk_repr"].astype(np.float32),
        batch["q"].astype(np.float32),
        batch["standing_bid"].astype(np.int64),
        batch["pool_size"].astype(np.int64),
    )


def _eval_val_bce(
    state:        CallPolicyTrainState,
    val_run_id:   str,
    *,
    splits:       list[str],
    batch_size:   int,
    device:       torch.device,
    data_root:    str | None,
) -> float:
    """Mean BCE over the requested splits of `val_run_id`."""
    from training.cfr_distillation import iter_shards
    state.net.eval()
    total = 0.0
    n_rows = 0
    with torch.no_grad():
        for split in splits:
            for batch in iter_shards(
                val_run_id, split, head="call",
                data_root=data_root, batch_size=batch_size,
            ):
                feats = torch.from_numpy(_batch_features(batch)).to(device).float()
                targets = torch.from_numpy(
                    batch["target_call_prob"].astype(np.float32)
                ).to(device)
                logits = state.net._raw_logits(feats)
                bce = F.binary_cross_entropy_with_logits(
                    logits, targets, reduction="sum",
                )
                total += float(bce.item())
                n_rows += int(targets.shape[0])
    state.net.train()
    return total / max(1, n_rows)


def _atomic_save(payload: dict, path: str) -> None:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="AR-2 Phase 7 CallPolicy trainer")
    p.add_argument("--run-id",     type=str, required=True)
    p.add_argument("--load-trunk", type=str, required=True)
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--out-dir",    type=str, default=None)
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--external-val-run-id", type=str, default=None,
                   help="If set, evaluate val on this run's val+test splits.")
    p.add_argument("--data-root",  type=str, default=None,
                   help="Override data/ root.")
    # Sweep harness passes --config <path>; accept and ignore.
    p.add_argument("--config",     type=str, default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    from training.cfr_distillation import iter_shards

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = args.data_root
    if data_root is None:
        data_root = os.path.abspath(os.path.join(_SRC, "..", "data"))
    out_dir = args.out_dir or os.path.join(
        data_root, "runs", args.run_id, "callpolicy"
    )
    os.makedirs(out_dir, exist_ok=True)

    config = CallPolicyConfig(
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
            args.run_id, "train", head="call",
            data_root=data_root, batch_size=args.batch_size,
        ):
            feats   = torch.from_numpy(_batch_features(batch)).to(device).float()
            targets = torch.from_numpy(
                batch["target_call_prob"].astype(np.float32)
            ).to(device)
            state.optimizer.zero_grad()
            loss = loss_step(state, feats, targets)
            loss.backward()
            state.optimizer.step()
            train_loss_sum += float(loss.item())
            n_batches += 1
            state.step += 1

        train_loss = train_loss_sum / max(1, n_batches)
        val_bce = _eval_val_bce(
            state, val_run_id, splits=val_splits,
            batch_size=args.batch_size, device=device, data_root=data_root,
        )
        wall = time.monotonic() - t_start
        curve.append({
            "epoch":        epoch,
            "train_loss":   train_loss,
            "val_bce":      val_bce,
            "wall_clock_s": wall,
        })
        with open(os.path.join(out_dir, "training_curve.json"), "w") as f:
            json.dump(curve, f, indent=2)

        if best_val is None or val_bce < best_val:
            best_val = val_bce
            _atomic_save(
                {
                    "state_dict":  state.net.state_dict(),
                    "config_dict": config.to_dict(),
                    "val_bce":     val_bce,
                    "epoch":       epoch,
                },
                os.path.join(out_dir, "best.pt"),
            )

        print(f"[callpolicy] epoch={epoch} train_loss={train_loss:.4f} "
              f"val_bce={val_bce:.4f} best={best_val:.4f}", flush=True)


if __name__ == "__main__":
    _main()


__all__ = ["CallPolicyTrainState", "build_train_state", "loss_step", "_main"]
