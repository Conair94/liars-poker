"""AR-0b tests for the compare reporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.compare import generate_report, write_report


def _write_fake_results_csv(bench_dir: Path, rows: list[dict]) -> None:
    columns = [
        "run_id", "cell", "opponent", "metric",
        "value", "ci_low", "ci_high", "n_games",
        "games_per_pair", "computed_at",
    ]
    bench_dir.mkdir(parents=True, exist_ok=True)
    with open(bench_dir / "results.csv", "w") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(c, "")) for c in columns) + "\n")


def _make_sweep(tmp_path: Path, sweep_id: str, rows: list[dict]) -> Path:
    sweeps_dir = tmp_path / "data" / "sweeps"
    sweep_dir = sweeps_dir / sweep_id
    bench_dir = sweep_dir / "bench"
    _write_fake_results_csv(bench_dir, rows)
    return sweeps_dir


def test_compare_markdown_has_heading(tmp_path, monkeypatch):
    import training.compare as cmp_mod
    sweep_id = "ar0b-cmp-test"
    sweeps_dir = _make_sweep(tmp_path, sweep_id, rows=[
        {
            "run_id": "run-abc", "cell": '{"a":1}', "opponent": "agent_b",
            "metric": "winrate", "value": 0.72, "ci_low": 0.62, "ci_high": 0.81,
            "n_games": 50, "games_per_pair": 50, "computed_at": "2026-04-28T00:00:00Z",
        }
    ])
    monkeypatch.setattr(cmp_mod, "_DATA_SWEEPS", sweeps_dir)
    content = generate_report(sweep_id)
    assert f"# Sweep {sweep_id}" in content
    assert "## Win-rate matrix" in content


def test_compare_markdown_row_count(tmp_path, monkeypatch):
    import training.compare as cmp_mod
    sweep_id = "ar0b-cmp-rows"
    rows = [
        {
            "run_id": f"run-{i}", "cell": f'{{"i":{i}}}', "opponent": "opp",
            "metric": "winrate", "value": 0.5 + i * 0.05,
            "ci_low": 0.4, "ci_high": 0.6,
            "n_games": 50, "games_per_pair": 50, "computed_at": "2026-04-28T00:00:00Z",
        }
        for i in range(4)
    ]
    sweeps_dir = _make_sweep(tmp_path, sweep_id, rows=rows)
    monkeypatch.setattr(cmp_mod, "_DATA_SWEEPS", sweeps_dir)
    content = generate_report(sweep_id)
    # 4 data rows + header + separator = 6 table lines, but we just check 4 run-ids appear
    for i in range(4):
        assert f"run-{i}" in content


def test_compare_write_report_creates_file(tmp_path, monkeypatch):
    import training.compare as cmp_mod
    sweep_id = "ar0b-write-test"
    sweeps_dir = _make_sweep(tmp_path, sweep_id, rows=[
        {
            "run_id": "run-x", "cell": '{}', "opponent": "opp",
            "metric": "winrate", "value": 0.6, "ci_low": 0.5, "ci_high": 0.7,
            "n_games": 50, "games_per_pair": 50, "computed_at": "2026-04-28T00:00:00Z",
        }
    ])
    monkeypatch.setattr(cmp_mod, "_DATA_SWEEPS", sweeps_dir)
    out = write_report(sweep_id)
    assert out.exists()
    assert out.suffix == ".md"
    content = out.read_text()
    assert "# Sweep" in content


def test_compare_lbr_not_yet(tmp_path, monkeypatch):
    import training.compare as cmp_mod
    sweep_id = "ar0b-lbr-test"
    sweeps_dir = _make_sweep(tmp_path, sweep_id, rows=[
        {
            "run_id": "run-y", "cell": '{}', "opponent": "opp",
            "metric": "winrate", "value": 0.5, "ci_low": 0.4, "ci_high": 0.6,
            "n_games": 50, "games_per_pair": 50, "computed_at": "2026-04-28T00:00:00Z",
        }
    ])
    monkeypatch.setattr(cmp_mod, "_DATA_SWEEPS", sweeps_dir)
    content = generate_report(sweep_id)
    assert "N/A" in content  # LBR placeholder
