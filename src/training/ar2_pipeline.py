"""AR-2 Phase 7 — per-cell pipeline driver.

Single entry point the sweep harness invokes once per cell. Runs distillation,
trains both heads as subprocesses, and writes `ar2_summary.json` +
`manifest.json` (so AR-0b's `_manifest_completed` flips to True).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_SRC, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from training.cfr_distillation import run_distillation  # noqa: E402


def _run_module(module: str, args: list[str], cwd: str, env: dict) -> int:
    cmd = [sys.executable, "-m", module] + args
    print(f"[ar2_pipeline] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


def _update_manifest(manifest_path: str, *, completed: bool, exit_code: int | None) -> None:
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            data = json.load(f)
    else:
        data = {}
    data["completed"] = completed
    data["exit_code"] = exit_code
    if completed:
        data["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, manifest_path)


def _read_best_payload(path: str) -> dict:
    import torch
    return torch.load(path, weights_only=False, map_location="cpu")


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AR-2 Phase 7 per-cell pipeline")
    p.add_argument("--run-id",            type=str, required=True)
    p.add_argument("--N",                 type=int, required=True)
    p.add_argument("--seed",              type=int, default=0)
    p.add_argument("--trunk-ckpt",        type=str, required=True)
    p.add_argument("--holdout-run-id",    type=str, required=True)
    p.add_argument("--max-iters",         type=int, default=500)
    p.add_argument("--callpolicy-epochs", type=int, default=20)
    p.add_argument("--bidpolicy-epochs",  type=int, default=30)
    p.add_argument("--device",            type=str, default="cpu")
    p.add_argument("--data-root",         type=str, default=None)
    # Accept-and-ignore: sweep harness passes --config.
    p.add_argument("--config",            type=str, default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    data_root = args.data_root or os.path.join(_REPO_ROOT, "data")
    run_dir = os.path.abspath(os.path.join(data_root, "runs", args.run_id))
    os.makedirs(run_dir, exist_ok=True)
    manifest_path = os.path.join(run_dir, "manifest.json")

    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")

    # 1. Distillation
    t0 = time.monotonic()
    summary = run_distillation(
        N=args.N, seed=args.seed, run_id=args.run_id, trunk_ckpt=args.trunk_ckpt,
        max_iters=args.max_iters, device=args.device, data_root=data_root,
    )
    t_distill_s = time.monotonic() - t0

    cp_out = os.path.join(run_dir, "callpolicy")
    bp_out = os.path.join(run_dir, "bidpolicy")

    common = [
        "--run-id",                args.run_id,
        "--load-trunk",            args.trunk_ckpt,
        "--external-val-run-id",   args.holdout_run_id,
        "--device",                args.device,
        "--seed",                  str(args.seed),
        "--data-root",             data_root,
    ]

    # 2. CallPolicy
    rc = _run_module(
        "agents.learned.callpolicy.trainer",
        common + [
            "--epochs",  str(args.callpolicy_epochs),
            "--out-dir", cp_out,
        ],
        cwd=_REPO_ROOT, env=env,
    )
    if rc != 0:
        _update_manifest(manifest_path, completed=False, exit_code=rc)
        print(f"[ar2_pipeline] callpolicy trainer failed rc={rc}", file=sys.stderr)
        return rc

    # 3. BidPolicy
    rc = _run_module(
        "agents.learned.bidpolicy.trainer",
        common + [
            "--epochs",  str(args.bidpolicy_epochs),
            "--out-dir", bp_out,
        ],
        cwd=_REPO_ROOT, env=env,
    )
    if rc != 0:
        _update_manifest(manifest_path, completed=False, exit_code=rc)
        print(f"[ar2_pipeline] bidpolicy trainer failed rc={rc}", file=sys.stderr)
        return rc

    # 4. Aggregate per-cell summary
    cp_best = _read_best_payload(os.path.join(cp_out, "best.pt"))
    bp_best = _read_best_payload(os.path.join(bp_out, "best.pt"))

    summary_path = os.path.join(run_dir, "ar2_summary.json")
    out = {
        "run_id":                     args.run_id,
        "N":                          args.N,
        "seed":                       args.seed,
        "distillation_wall_clock_s":  t_distill_s,
        "n_deals":                    summary["n_deals"],
        "n_rows":                     summary["n_rows"],
        "n_bid_rows":                 summary["n_bid_rows"],
        "callpolicy_val_bce_best":    float(cp_best["val_bce"]),
        "bidpolicy_val_kl_best":      float(bp_best["val_kl"]),
        "bidpolicy_val_kl_per_n":     bp_best["val_kl_per_n"],
        "callpolicy_ckpt":            os.path.join(cp_out, "best.pt"),
        "bidpolicy_ckpt":             os.path.join(bp_out, "best.pt"),
    }
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)

    _update_manifest(manifest_path, completed=True, exit_code=0)
    print(f"[ar2_pipeline] cell {args.run_id} OK — summary at {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_main())


__all__ = ["_main"]
