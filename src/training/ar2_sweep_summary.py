"""AR-2 Phase 7 — sweep summary + elbow detector.

Reads each cell's `ar2_summary.json`, builds `kl_per_n_per_N`, computes
per-n slopes per doubling, picks the elbow per design §3.3, and writes
`ar2_kl_curve.json` + `elbow.json` under `data/sweeps/<sweep_id>/`.

Phase-7 deviation: AR-0b sweep harness writes `data/sweeps/<sweep_id>/index.json`
(cell_key → run_id), not `cells.json`. This summariser reads `index.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_SRC, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_N_BUCKETS = (4, 6, 8, 10)
_SWEEP_NS  = (1000, 5000, 10000, 50000)


def compute_slopes(
    kl_per_n_per_N: dict[int, dict[int, float]],
    *,
    n_buckets: tuple[int, ...] = _N_BUCKETS,
) -> dict[int, dict[int, float | None]]:
    """slope_n(N) = (kl_n(N/2) - kl_n(N)) / kl_n(N/2), for N with N/2 in the data."""
    all_Ns = sorted({N for d in kl_per_n_per_N.values() for N in d.keys()})
    slopes: dict[int, dict[int, float | None]] = {n: {} for n in n_buckets}
    for n in n_buckets:
        for N in all_Ns:
            half = N // 2
            if half not in kl_per_n_per_N.get(n, {}):
                slopes[n][N] = None
                continue
            kl_prev = kl_per_n_per_N[n][half]
            kl_cur  = kl_per_n_per_N[n].get(N)
            if kl_cur is None or kl_prev is None or kl_prev <= 0:
                slopes[n][N] = None
                continue
            slopes[n][N] = (kl_prev - kl_cur) / kl_prev
    return slopes


def select_elbow(
    kl_per_n_per_N: dict[int, dict[int, float]],
    *,
    threshold: float = 0.05,
    n_buckets: tuple[int, ...] = _N_BUCKETS,
    fallback_N: int = 10000,
) -> tuple[int, str, dict[int, dict[int, float | None]]]:
    """Smallest N where slope_n(N) < threshold for every n; else fallback."""
    slopes = compute_slopes(kl_per_n_per_N, n_buckets=n_buckets)
    all_Ns = sorted({N for d in kl_per_n_per_N.values() for N in d.keys()})
    for N in all_Ns:
        per_n_slopes = [slopes[n].get(N) for n in n_buckets]
        if any(s is None for s in per_n_slopes):
            continue
        if max(per_n_slopes) < threshold:  # type: ignore[type-var]
            return N, f"elbow_at_N={N}", slopes
    return fallback_N, "fallback_per_design_3.3", slopes


def _discover_run_ids(sweep_dir: Path) -> list[str]:
    index_path = sweep_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No index.json at {index_path}")
    with open(index_path) as f:
        index = json.load(f)
    return list(index.values())


def _load_cell_summaries(data_root: Path, run_ids: list[str]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for run_id in run_ids:
        sp = data_root / "runs" / run_id / "ar2_summary.json"
        if not sp.exists():
            print(f"[ar2_sweep_summary] WARN: missing {sp}", file=sys.stderr)
            continue
        with open(sp) as f:
            s = json.load(f)
        out[int(s["N"])] = s
    return out


def _build_kl_per_n_per_N(cells: dict[int, dict]) -> dict[int, dict[int, float]]:
    kl: dict[int, dict[int, float]] = {n: {} for n in _N_BUCKETS}
    for N, s in cells.items():
        per_n = s.get("bidpolicy_val_kl_per_n", {}) or {}
        for n in _N_BUCKETS:
            v = per_n.get(str(n))
            if v is None:
                continue
            kl[n][N] = float(v)
    return kl


def _maybe_plot(kl_per_n_per_N: dict[int, dict[int, float]], path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    for n, series in kl_per_n_per_N.items():
        if not series:
            continue
        xs = sorted(series.keys())
        ys = [series[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=f"n={n}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (deal count)")
    ax.set_ylabel("val KL")
    ax.set_title("AR-2 distillation: per-n val KL vs N")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def summarise_sweep(sweep_id: str, *, data_root: str | None = None) -> dict:
    root = Path(data_root) if data_root else Path(_REPO_ROOT) / "data"
    sweep_dir = root / "sweeps" / sweep_id
    run_ids = _discover_run_ids(sweep_dir)
    cells   = _load_cell_summaries(root, run_ids)
    kl_per_n_per_N = _build_kl_per_n_per_N(cells)

    chosen_N, reason, slopes = select_elbow(kl_per_n_per_N)

    # JSON-friendly forms (string-keyed inner dicts).
    kl_out = {
        str(n): {str(N): v for N, v in d.items()}
        for n, d in kl_per_n_per_N.items()
    }
    slopes_out = {
        str(n): {str(N): v for N, v in d.items()}
        for n, d in slopes.items()
    }

    curve_path = sweep_dir / "ar2_kl_curve.json"
    with open(curve_path, "w") as f:
        json.dump({"kl_per_n_per_N": kl_out, "slopes": slopes_out}, f, indent=2)

    chosen_cell = cells.get(chosen_N)
    elbow = {
        "chosen_N":         chosen_N,
        "reason":           reason,
        "slopes":           slopes_out,
        "kl_per_n_per_N":   kl_out,
        "callpolicy_ckpt":  chosen_cell["callpolicy_ckpt"] if chosen_cell else None,
        "bidpolicy_ckpt":   chosen_cell["bidpolicy_ckpt"]  if chosen_cell else None,
    }
    with open(sweep_dir / "elbow.json", "w") as f:
        json.dump(elbow, f, indent=2)

    _maybe_plot(kl_per_n_per_N, str(sweep_dir / "ar2_kl_curve.png"))
    return elbow


def _main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="AR-2 Phase 7 sweep summariser")
    p.add_argument("--sweep-id",  type=str, required=True)
    p.add_argument("--data-root", type=str, default=None)
    args = p.parse_args(argv)
    elbow = summarise_sweep(args.sweep_id, data_root=args.data_root)
    print(json.dumps({"chosen_N": elbow["chosen_N"], "reason": elbow["reason"]}, indent=2))


if __name__ == "__main__":
    _main()


__all__ = [
    "compute_slopes",
    "select_elbow",
    "summarise_sweep",
    "_main",
]
