"""AR-0b tests for bench_sweep."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.bench_only import run_bench_only
from training.bench_sweep import run_bench_sweep


def _make_sweep_fixture(tmp_path: Path, run_id: str, agent_a: str, agent_b: str,
                         games: int, seed: int) -> tuple[Path, Path]:
    """Set up a fake sweep directory with one completed run."""
    runs_dir = tmp_path / "data" / "runs"
    sweeps_dir = tmp_path / "data" / "sweeps"
    sweep_id = "ar0b-test-sweep"
    sweep_dir = sweeps_dir / sweep_id
    bench_dir = sweep_dir / "bench"
    bench_dir.mkdir(parents=True)

    # Write index.json
    index = {json.dumps({"handmodel": agent_a, "callpolicy": agent_b}, sort_keys=True): run_id}
    (sweep_dir / "index.json").write_text(json.dumps(index))

    # Run the actual bench_only to produce results.json + manifest
    import training.bench_only as bo_mod
    old_runs = bo_mod._DATA_RUNS
    bo_mod._DATA_RUNS = runs_dir
    try:
        run_bench_only(run_id=run_id, agent_a_key=agent_a, agent_b_key=agent_b,
                       games_per_pair=games, seed=seed)
    finally:
        bo_mod._DATA_RUNS = old_runs

    return runs_dir, sweeps_dir


def test_bench_sweep_produces_rows(tmp_path):
    runs_dir, sweeps_dir = _make_sweep_fixture(
        tmp_path, "ar0b-test-run-abc12345",
        agent_a="random", agent_b="random",
        games=100, seed=7,
    )

    import training.bench_sweep as bs_mod
    old_runs = bs_mod._DATA_RUNS
    old_sweeps = bs_mod._DATA_SWEEPS
    bs_mod._DATA_RUNS = runs_dir
    bs_mod._DATA_SWEEPS = sweeps_dir
    try:
        out = run_bench_sweep("ar0b-test-sweep")
    finally:
        bs_mod._DATA_RUNS = old_runs
        bs_mod._DATA_SWEEPS = old_sweeps

    # Should have written parquet or csv
    assert out.exists() or out.with_suffix(".csv").exists()


def test_bench_sweep_winrate_near_half_for_random_vs_random(tmp_path):
    runs_dir, sweeps_dir = _make_sweep_fixture(
        tmp_path, "ar0b-test-run-rnd99",
        agent_a="random", agent_b="random",
        games=100, seed=99,
    )

    import training.bench_sweep as bs_mod
    old_runs = bs_mod._DATA_RUNS
    old_sweeps = bs_mod._DATA_SWEEPS
    bs_mod._DATA_RUNS = runs_dir
    bs_mod._DATA_SWEEPS = sweeps_dir
    try:
        out = run_bench_sweep("ar0b-test-sweep")
    finally:
        bs_mod._DATA_RUNS = old_runs
        bs_mod._DATA_SWEEPS = old_sweeps

    # Load results and check win-rate
    results_path = out if out.exists() else out.with_suffix(".csv")
    try:
        import pandas as pd
        df = pd.read_parquet(out) if out.exists() else pd.read_csv(results_path)
        winrate_rows = df[df["metric"] == "winrate"]
        assert len(winrate_rows) >= 1
        # random vs random: win rate should be in [0.35, 0.65] with high probability
        val = float(winrate_rows["value"].iloc[0])
        assert 0.35 <= val <= 0.65, f"win rate {val} outside expected range for random vs random"
    except ImportError:
        # pandas not available — just verify file exists
        assert results_path.exists()


def test_bench_sweep_skips_incomplete_runs(tmp_path):
    runs_dir = tmp_path / "data" / "runs"
    sweeps_dir = tmp_path / "data" / "sweeps"
    sweep_id = "ar0b-skip-test"
    sweep_dir = sweeps_dir / sweep_id
    bench_dir = sweep_dir / "bench"
    bench_dir.mkdir(parents=True)

    # Write index pointing to a run that isn't completed
    run_id = "ar0b-incomplete-run"
    index = {json.dumps({"handmodel": "random"}, sort_keys=True): run_id}
    (sweep_dir / "index.json").write_text(json.dumps(index))

    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"completed": False, "exit_code": None})
    )

    import training.bench_sweep as bs_mod
    old_runs = bs_mod._DATA_RUNS
    old_sweeps = bs_mod._DATA_SWEEPS
    bs_mod._DATA_RUNS = runs_dir
    bs_mod._DATA_SWEEPS = sweeps_dir
    try:
        out = run_bench_sweep(sweep_id)
    finally:
        bs_mod._DATA_RUNS = old_runs
        bs_mod._DATA_SWEEPS = old_sweeps

    # Output should be empty (0 rows)
    results_path = out if out.exists() else out.with_suffix(".csv")
    try:
        import pandas as pd
        df = pd.read_parquet(out) if out.exists() else pd.read_csv(results_path)
        assert len(df) == 0
    except ImportError:
        pass
