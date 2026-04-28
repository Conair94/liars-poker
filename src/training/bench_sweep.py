"""
AR-0b bench_sweep: pairwise-benchmark every completed run in a sweep.

Usage:
    python -m training.bench_sweep <sweep_id> [--games-per-pair 200]

Reads data/sweeps/<sweep_id>/index.json to enumerate runs.
For each completed run reads its results.json (agent_a, agent_b, win_rate_a).
Writes long-format Parquet (or CSV fallback) to:
    data/sweeps/<sweep_id>/bench/results.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_RUNS = _REPO_ROOT / "data" / "runs"
_DATA_SWEEPS = _REPO_ROOT / "data" / "sweeps"
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from training.bench_only import run_bench_only  # noqa: E402


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _write_manifest(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _write_results(rows: list[dict], out_path: Path) -> None:
    """Write results rows to Parquet (preferred) or CSV (fallback)."""
    columns = [
        "run_id", "cell", "opponent", "metric",
        "value", "ci_low", "ci_high", "n_games",
        "games_per_pair", "computed_at",
    ]
    try:
        import pandas as pd  # type: ignore[import]
        df = pd.DataFrame(rows, columns=columns)
        try:
            df.to_parquet(out_path, index=False)
        except Exception:
            # pyarrow not available — fall back to CSV
            csv_path = out_path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            print(f"[bench_sweep] pyarrow unavailable; wrote CSV to {csv_path}", flush=True)
    except ImportError:
        # pandas not available — write minimal CSV by hand
        csv_path = out_path.with_suffix(".csv")
        with open(csv_path, "w") as f:
            f.write(",".join(columns) + "\n")
            for row in rows:
                f.write(",".join(str(row.get(c, "")) for c in columns) + "\n")
        print(f"[bench_sweep] pandas unavailable; wrote CSV to {csv_path}", flush=True)


def run_bench_sweep(sweep_id: str, games_per_pair: int = 200, seed: int = 0) -> Path:
    sweep_dir = _DATA_SWEEPS / sweep_id
    index_path = sweep_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"sweep index not found: {index_path}")

    index: dict[str, str] = json.loads(index_path.read_text())

    bench_dir = sweep_dir / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    now = _utc_now()

    for cell_json, run_id in index.items():
        run_dir = _DATA_RUNS / run_id
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"[bench_sweep] skip {run_id} (no manifest)", flush=True)
            continue
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("completed", False):
            print(f"[bench_sweep] skip {run_id} (not completed)", flush=True)
            continue

        results_path = run_dir / "results.json"
        if not results_path.exists():
            print(f"[bench_sweep] skip {run_id} (no results.json)", flush=True)
            continue

        results = json.loads(results_path.read_text())
        agent_a = results["agent_a"]
        agent_b = results["agent_b"]
        wins_a = results["wins_a"]
        actual_games = results["games_per_pair"]
        win_rate = results["win_rate_a"]
        ci_low = results.get("ci_low", float("nan"))
        ci_high = results.get("ci_high", float("nan"))

        rows.append({
            "run_id": run_id,
            "cell": cell_json,
            "opponent": agent_b,
            "metric": "winrate",
            "value": win_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_games": actual_games,
            "games_per_pair": games_per_pair,
            "computed_at": now,
        })
        # Symmetric row for agent_b's perspective
        rows.append({
            "run_id": run_id,
            "cell": cell_json,
            "opponent": agent_a,
            "metric": "winrate",
            "value": round(1.0 - win_rate, 6),
            "ci_low": round(1.0 - ci_high, 6),
            "ci_high": round(1.0 - ci_low, 6),
            "n_games": actual_games,
            "games_per_pair": games_per_pair,
            "computed_at": now,
        })

    out_path = bench_dir / "results.parquet"
    _write_results(rows, out_path)

    _write_manifest(
        bench_dir / "manifest.json",
        {
            "sweep_id": sweep_id,
            "rows": len(rows),
            "completed": True,
            "computed_at": now,
        },
    )
    print(f"[bench_sweep] wrote {len(rows)} rows to {out_path}", flush=True)
    return out_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="AR-0b bench_sweep")
    parser.add_argument("sweep_id")
    parser.add_argument("--games-per-pair", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_bench_sweep(args.sweep_id, games_per_pair=args.games_per_pair, seed=args.seed)


if __name__ == "__main__":
    _main()
