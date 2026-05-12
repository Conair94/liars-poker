"""AR-2 Phase 7 — pilot pre-flight wrapper.

Runs `run_distillation(N=1000, ...)`, records wall-clock + per-deal solver
stats, prints a one-line GO/NO-GO verdict against the three gates in design
§1. Also generates `ar2_holdout_2k` when `--holdout` is set (idempotent).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from training.cfr_distillation import (  # noqa: E402
    run_distillation,
)

_REPO_ROOT = os.path.abspath(os.path.join(_SRC, ".."))
_DEFAULT_DATA_ROOT = os.path.join(_REPO_ROOT, "data")


def _hist(values: list[float], bins: int = 10) -> dict:
    if not values:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=bins)
    return {"edges": edges.tolist(), "counts": counts.tolist()}


def run_pilot(
    run_id:     str,
    N:          int,
    seed:       int,
    trunk_ckpt: str,
    *,
    max_iters:  int = 200,
    device:     str = "cpu",
    data_root:  str | None = None,
    workers:    int = 1,
    progress:   bool = True,
) -> dict:
    """Run distillation, harvest solver stats inline, write pilot JSONs, return GO/NO-GO dict."""
    if data_root is None:
        data_root = _DEFAULT_DATA_ROOT
    run_dir = os.path.abspath(os.path.join(data_root, "runs", run_id))

    t0 = time.monotonic()
    summary = run_distillation(
        N=N, seed=seed, run_id=run_id, trunk_ckpt=trunk_ckpt,
        max_iters=max_iters, device=device, data_root=data_root,
        workers=workers, progress=progress,
    )
    wall_clock_s = time.monotonic() - t0

    iters_list: list[int]   = list(summary.get("iters_used", []))
    eps_list:   list[float] = list(summary.get("final_eps",  []))

    eps_arr = np.asarray(eps_list, dtype=np.float64)
    frac_converged = float((eps_arr < 1e-3).mean()) if len(eps_arr) else 0.0

    rows_per_deal = summary["n_rows"] / max(1, summary["n_deals"])
    projected_50k_hours = wall_clock_s * (50000.0 / max(1, N)) / 3600.0

    timing = {
        "wall_clock_seconds":   wall_clock_s,
        "n_deals":              summary["n_deals"],
        "n_rows":               summary["n_rows"],
        "n_bid_rows":           summary["n_bid_rows"],
        "rows_per_deal":        rows_per_deal,
        "projected_50k_hours":  projected_50k_hours,
        "max_iters":            max_iters,
        "N":                    N,
        "seed":                 seed,
        "run_id":               run_id,
    }
    stats = {
        "iters_hist":     _hist([float(x) for x in iters_list]),
        "eps_hist":       _hist(eps_list),
        "frac_converged": frac_converged,
        "max_iters":      max_iters,
    }

    with open(os.path.join(run_dir, "pilot_timing.json"), "w") as f:
        json.dump(timing, f, indent=2)
    with open(os.path.join(run_dir, "pilot_solver_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Gates (design §1 "Pass criteria")
    gate_wall  = wall_clock_s <= 30 * 60
    gate_conv  = frac_converged >= 0.95
    gate_rows  = 1.5 * N <= summary["n_rows"] <= 4 * N

    reasons = []
    if not gate_wall:
        reasons.append(f"wall_clock_s={wall_clock_s:.1f}>1800")
    if not gate_conv:
        reasons.append(f"frac_converged={frac_converged:.3f}<0.95")
    if not gate_rows:
        reasons.append(f"n_rows={summary['n_rows']} outside [{int(1.5*N)},{4*N}]")

    verdict = "GO" if not reasons else "NO-GO"
    return {
        "verdict":         verdict,
        "reasons":         reasons,
        "wall_clock_s":    wall_clock_s,
        "frac_converged":  frac_converged,
        "n_rows":          summary["n_rows"],
        "run_id":          run_id,
    }


def _main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="AR-2 Phase 7 pilot pre-flight")
    p.add_argument("--run-id",      type=str, required=True)
    p.add_argument("--N",           type=int, default=1000)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--trunk-ckpt",  type=str, required=True)
    p.add_argument("--max-iters",   type=int, default=200)
    p.add_argument("--device",      type=str, default="cpu")
    p.add_argument("--workers",     type=int, default=1,
                   help="Process-pool workers for parallel solver. 1 = serial.")
    p.add_argument("--no-progress", action="store_true",
                   help="Suppress per-batch progress lines on stderr.")
    p.add_argument("--holdout",     action="store_true",
                   help="After the primary pilot, also generate ar2_holdout_2k (seed=1000, N=2000).")
    p.add_argument("--data-root",   type=str, default=None,
                   help="Override data/ root (test/debug).")
    args = p.parse_args(argv)

    result = run_pilot(
        run_id=args.run_id, N=args.N, seed=args.seed,
        trunk_ckpt=args.trunk_ckpt, max_iters=args.max_iters,
        device=args.device, data_root=args.data_root,
        workers=args.workers, progress=(not args.no_progress),
    )

    if result["verdict"] == "GO":
        print(f"[GO] wall_clock={result['wall_clock_s']:.1f}s "
              f"frac_converged={result['frac_converged']:.3f} "
              f"n_rows={result['n_rows']}")
    else:
        print(f"[NO-GO: {'; '.join(result['reasons'])}]")

    if args.holdout:
        data_root = args.data_root or _DEFAULT_DATA_ROOT
        holdout_split = os.path.join(data_root, "runs", "ar2_holdout_2k", "cfr_deals", "split.json")
        if os.path.exists(holdout_split):
            print("[holdout] ar2_holdout_2k already present; skipping.")
        else:
            print("[holdout] generating ar2_holdout_2k (N=2000, seed=1000) ...")
            run_distillation(
                N=2000, seed=1000, run_id="ar2_holdout_2k", trunk_ckpt=args.trunk_ckpt,
                max_iters=args.max_iters, device=args.device, data_root=data_root,
                workers=args.workers, progress=(not args.no_progress),
            )


if __name__ == "__main__":
    _main()


__all__ = ["run_pilot", "_main"]
