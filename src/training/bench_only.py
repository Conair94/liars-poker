"""
AR-0b smoketest cell runner.

Plays a single agent matchup (handmodel vs callpolicy as agent keys),
writes results.json + manifest.json to data/runs/<run_id>/.

Usage (called by sweep driver):
    python -m training.bench_only \\
        --config <resolved_cell_config.yaml> \\
        --run-id <run_id> \\
        --handmodel exactconditional \\
        --callpolicy exact_adaptive \\
        --games-per-pair 50
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
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agents.registry import build_agent  # noqa: E402
from training.benchmark import run_match  # noqa: E402


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


def run_bench_only(
    run_id: str,
    agent_a_key: str,
    agent_b_key: str,
    games_per_pair: int,
    seed: int,
    sweep_id: str = "",
    git_sha: str = "",
) -> dict:
    run_dir = _DATA_RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    match_kwargs = {"exact_rules": True, "high_hand": True, "five_kings": False}
    wins_a = 0
    for g in range(games_per_pair):
        if g % 2 == 0:
            winner = run_match(agent_a_key, agent_b_key, match_kwargs, seed=seed + g)
            if winner == 0:
                wins_a += 1
        else:
            winner = run_match(agent_b_key, agent_a_key, match_kwargs, seed=seed + g)
            if winner == 1:
                wins_a += 1

    win_rate = wins_a / games_per_pair
    ci_low, ci_high = _wilson_ci(wins_a, games_per_pair)
    now = _utc_now()
    results = {
        "run_id": run_id,
        "agent_a": agent_a_key,
        "agent_b": agent_b_key,
        "games_per_pair": games_per_pair,
        "wins_a": wins_a,
        "win_rate_a": round(win_rate, 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "computed_at": now,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    _write_manifest(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "sweep_id": sweep_id,
            "git_sha": git_sha,
            "started_at": now,
            "last_completed_iter": 1,
            "completed": True,
            "exit_code": 0,
            "components": {"handmodel": None, "callpolicy": None, "bidpolicy": None},
        },
    )
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(description="AR-0b single-cell bench runner")
    parser.add_argument("--config", required=True, help="resolved cell config YAML")
    parser.add_argument("--run-id", required=True)
    # These args come from the sweep config runner.args template
    parser.add_argument("--handmodel", required=True, help="agent key for player A")
    parser.add_argument("--callpolicy", required=True, help="agent key for player B")
    parser.add_argument("--bidpolicy", default="analytic", help="ignored in AR-0b")
    parser.add_argument("--games-per-pair", type=int, default=50)
    args = parser.parse_args()

    # Read seed from resolved config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = cfg.get("random_seed", 0)

    run_dir = _DATA_RUNS / args.run_id

    try:
        run_bench_only(
            run_id=args.run_id,
            agent_a_key=args.handmodel,
            agent_b_key=args.callpolicy,
            games_per_pair=args.games_per_pair,
            seed=seed,
            sweep_id=cfg.get("sweep_id", ""),
            git_sha=cfg.get("git_sha", ""),
        )
    except Exception as exc:
        print(f"[bench_only] ERROR: {exc}", file=sys.stderr)
        _write_manifest(
            run_dir / "manifest.json",
            {
                "run_id": args.run_id,
                "sweep_id": cfg.get("sweep_id", ""),
                "git_sha": cfg.get("git_sha", ""),
                "started_at": _utc_now(),
                "last_completed_iter": 0,
                "completed": False,
                "exit_code": 1,
                "components": {"handmodel": None, "callpolicy": None, "bidpolicy": None},
            },
        )
        sys.exit(1)


if __name__ == "__main__":
    _main()
