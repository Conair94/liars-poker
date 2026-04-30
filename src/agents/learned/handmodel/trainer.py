"""AR-1 Phase-A trainer.

End-to-end:

    python -m agents.learned.handmodel.trainer \\
        --run-id ar1-... \\
        [--config <yaml>] [--device cpu] [--max-epochs 30]

Steps:
1. (Optional) collect rollouts under `config.rollout_agent` if the run
   directory does not already have a `rollouts/` tree.
2. Train `LearnedHandModelNet` with masked cross-entropy and linear
   warmup + early stop on val NLL.
3. Evaluate per-`n` Brier on the held-out test split. Compare against
   `AnalyticHandModel` and write `metrics.json` + best checkpoint.

This module is invokable as a script and importable for tests (the
unit tests drive a small in-process run with a tiny dataset).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE      = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_PROBS_DIR = os.path.join(_SRC_DIR, "training", "probs")
_REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
for _p in (_PROBS_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.learned.handmodel.baselines import AnalyticHandModel  # noqa: E402
from agents.learned.handmodel.config import HandModelConfig  # noqa: E402
from agents.learned.handmodel.dataset import (  # noqa: E402
    HandModelDataset,
    brier_per_n,
    build_phase_a_dataset,
    collate_batch,
    load_split,
)
from agents.learned.handmodel.network import (  # noqa: E402
    LearnedHandModel,
    LearnedHandModelNet,
    make_lr_lambda,
)
from agents.contracts import Infostate  # noqa: E402

_DATA_RUNS = Path(_REPO_ROOT) / "data" / "runs"


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _eval_loss(net: LearnedHandModelNet, loader: DataLoader, device: torch.device) -> float:
    net.eval()
    total_loss = 0.0
    total_n    = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = net(
                own_hand=batch["own_hand"],
                bid_tokens=batch["bid_tokens"],
                seat_tokens=batch["seat_tokens"],
                pos_tokens=batch["pos_tokens"],
                hist_mask=batch["hist_mask"],
                scalars=batch["scalars"],
                feasible=batch["feasible"],
            )
            loss = F.cross_entropy(logits, batch["label"], reduction="sum")
            total_loss += float(loss.item())
            total_n    += int(batch["label"].numel())
    return total_loss / max(1, total_n)


def _eval_per_n_brier(
    model: LearnedHandModel,
    rows,
    *,
    batch_size: int = 256,
) -> dict[int, float]:
    """Compute per-`n` Brier of `model` on `rows` (list[RolloutDecision])."""
    qs:    list[np.ndarray] = []
    ys:    list[int]        = []
    ns:    list[int]        = []
    for i in range(0, len(rows), batch_size):
        chunk  = rows[i:i + batch_size]
        infos  = [r.infostate for r in chunk]
        beliefs = model.belief_batch(infos)
        for r, b in zip(chunk, beliefs):
            qs.append(b.q)
            ys.append(int(r.label))
            ns.append(int(r.infostate.pool_size))
    return brier_per_n(np.stack(qs), np.asarray(ys, dtype=np.int64), np.asarray(ns, dtype=np.int64))


def _eval_analytic_brier(rows, *, batch_size: int = 256) -> dict[int, float]:
    qs:    list[np.ndarray] = []
    ys:    list[int]        = []
    ns:    list[int]        = []
    base = AnalyticHandModel()
    for i in range(0, len(rows), batch_size):
        chunk  = rows[i:i + batch_size]
        for r in chunk:
            b = base.belief(r.infostate)
            qs.append(b.q)
            ys.append(int(r.label))
            ns.append(int(r.infostate.pool_size))
    return brier_per_n(np.stack(qs), np.asarray(ys, dtype=np.int64), np.asarray(ns, dtype=np.int64))


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    best_epoch:    int
    best_val_loss: float
    test_loss:     float
    test_brier_per_n:    dict[int, float]
    analytic_brier_per_n: dict[int, float]


def train(
    config: HandModelConfig,
    run_dir: str,
    *,
    skip_rollouts: bool = False,
) -> TrainResult:
    """Run end-to-end Phase-A training.

    Side effects:
    - writes `<run_dir>/rollouts/{train,val,test}/n*.jsonl.gz` (unless
      `skip_rollouts`).
    - writes `<run_dir>/handmodel/best.pt` (best-val checkpoint).
    - writes `<run_dir>/metrics.json` summarising the result.
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    rollout_dir = run_path / "rollouts"
    if not skip_rollouts and not rollout_dir.exists():
        build_phase_a_dataset(
            out_dir=str(rollout_dir),
            agent_key=config.rollout_agent,
            hand_sizes=config.hand_sizes,
            train_per_n=config.train_decisions_per_n,
            val_per_n=config.val_decisions_per_n,
            test_per_n=config.test_decisions_per_n,
            seed=config.seed,
        )

    # Resolve split paths from the directory tree.
    splits: dict[str, dict[int, str]] = {"train": {}, "val": {}, "test": {}}
    for split in splits:
        for n in config.hand_sizes:
            p = rollout_dir / split / f"n{n}.jsonl.gz"
            if p.exists():
                splits[split][n] = str(p)

    train_rows = load_split(splits["train"])
    val_rows   = load_split(splits["val"])
    test_rows  = load_split(splits["test"])
    if not train_rows or not val_rows:
        raise RuntimeError(f"empty train/val rollouts at {rollout_dir}")

    train_ds = HandModelDataset(train_rows, bid_hist_len=config.bid_hist_len, num_seats=config.num_seats)
    val_ds   = HandModelDataset(val_rows,   bid_hist_len=config.bid_hist_len, num_seats=config.num_seats)
    test_ds  = HandModelDataset(test_rows,  bid_hist_len=config.bid_hist_len, num_seats=config.num_seats)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_batch, drop_last=False, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_batch, drop_last=False, num_workers=0,
    )

    net = LearnedHandModelNet(config).to(device)
    optim = torch.optim.Adam(
        net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=make_lr_lambda(config.warmup_steps),
    )

    best_val   = math.inf
    best_epoch = 0
    best_state: dict | None = None
    no_improve = 0

    ckpt_dir = run_path / "handmodel"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    for epoch in range(1, config.max_epochs + 1):
        net.train()
        epoch_loss = 0.0
        epoch_n    = 0
        t0 = time.time()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = net(
                own_hand=batch["own_hand"],
                bid_tokens=batch["bid_tokens"],
                seat_tokens=batch["seat_tokens"],
                pos_tokens=batch["pos_tokens"],
                hist_mask=batch["hist_mask"],
                scalars=batch["scalars"],
                feasible=batch["feasible"],
            )
            loss = F.cross_entropy(logits, batch["label"])
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            bn = int(batch["label"].numel())
            epoch_loss += float(loss.item()) * bn
            epoch_n    += bn

        train_loss = epoch_loss / max(1, epoch_n)
        val_loss   = _eval_loss(net, val_loader, device)
        elapsed    = time.time() - t0
        print(
            f"epoch {epoch:>3} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"lr {sched.get_last_lr()[0]:.2e} | {elapsed:.1f}s",
            flush=True,
        )

        improved = val_loss < best_val * (1.0 - config.early_stop_rel_tol)
        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            # Persist best checkpoint via the AR-0a contract.
            tmp_model = LearnedHandModel(LearnedHandModelNet(config), device="cpu")
            tmp_model.net.load_state_dict(best_state)
            tmp_model.save(str(best_path), iter=epoch)

        if improved:
            no_improve = 0
        else:
            no_improve += 1

        if epoch >= config.min_epochs and no_improve >= config.early_stop_patience:
            print(f"early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    if best_state is None:
        raise RuntimeError("no successful epoch — training did not progress")

    # Final eval on test set.
    eval_model = LearnedHandModel.from_checkpoint(str(best_path), device=str(device))
    test_brier = _eval_per_n_brier(eval_model, test_rows)
    analytic_brier = _eval_analytic_brier(test_rows)
    test_loss = _eval_loss(eval_model.net, DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_batch, num_workers=0,
    ), device)

    metrics = {
        "best_epoch":     best_epoch,
        "best_val_loss":  best_val,
        "test_loss":      test_loss,
        "test_brier_per_n":     {str(k): v for k, v in test_brier.items()},
        "analytic_brier_per_n": {str(k): v for k, v in analytic_brier.items()},
    }
    (run_path / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Update manifest so the sweep driver's --resume can detect completion.
    manifest_path = run_path / "manifest.json"
    if manifest_path.exists():
        try:
            mdata = json.loads(manifest_path.read_text())
        except Exception:
            mdata = {}
        mdata["completed"] = True
        mdata["last_completed_iter"] = best_epoch
        mdata["components"] = {"handmodel": str(run_path / "handmodel" / "best.pt"),
                               "callpolicy": None, "bidpolicy": None}
        tmp = manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(mdata, indent=2))
        os.replace(tmp, manifest_path)

    return TrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val,
        test_loss=test_loss,
        test_brier_per_n=test_brier,
        analytic_brier_per_n=analytic_brier,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AR-1 Phase-A HandModel trainer.")
    p.add_argument("--run-id", required=True, help="Run id under data/runs/")
    p.add_argument("--config", default=None, help="Optional config dict YAML/JSON; missing keys default.")
    p.add_argument("--device", default=None)
    p.add_argument("--seed",   default=None, type=int)
    p.add_argument("--max-epochs", default=None, type=int)
    p.add_argument("--hidden-dim", default=None, type=int)
    p.add_argument("--num-trunk-layers", default=None, type=int)
    p.add_argument("--bid-hist-dim", default=None, type=int)
    p.add_argument("--train-decisions-per-n", default=None, type=int)
    p.add_argument("--val-decisions-per-n",   default=None, type=int)
    p.add_argument("--test-decisions-per-n",  default=None, type=int)
    p.add_argument("--rollout-agent", default=None)
    p.add_argument("--skip-rollouts", action="store_true")
    return p.parse_args(argv)


def _load_config_dict(path: str | None) -> dict:
    if path is None:
        return {}
    text = Path(path).read_text()
    if path.endswith((".yml", ".yaml")):
        import yaml
        return yaml.safe_load(text) or {}
    return json.loads(text)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    base = _load_config_dict(args.config)

    overrides = {}
    for k in (
        "device", "seed", "max_epochs", "hidden_dim", "num_trunk_layers",
        "bid_hist_dim", "train_decisions_per_n", "val_decisions_per_n",
        "test_decisions_per_n", "rollout_agent",
    ):
        cli_val = getattr(args, k.replace("_", "_") if k != "max_epochs" else "max_epochs", None)
        # argparse uses dashes; translate.
        cli_val = getattr(args, k, None)
        if cli_val is not None:
            overrides[k] = cli_val
    base.update(overrides)
    config = HandModelConfig(**{k: v for k, v in base.items() if k in HandModelConfig.__dataclass_fields__})

    run_dir = _DATA_RUNS / args.run_id
    print(f"run_dir = {run_dir}")
    print(f"config  = {config.to_dict()}")
    train(config, str(run_dir), skip_rollouts=args.skip_rollouts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
