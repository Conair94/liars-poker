"""AR-2 Phase 7 — pilot smoke test."""

from __future__ import annotations

import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC  = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from training.ar2_pilot import _main as pilot_main  # noqa: E402

_CKPT = os.path.join(
    _REPO, "data/runs/ar1-20260430T030730Z-b64-h256-n2-fb17d031/handmodel/best.pt"
)


@pytest.mark.slow
def test_pilot_smoke(tmp_path):
    if not os.path.exists(_CKPT):
        pytest.skip(f"AR-1 trunk checkpoint not present at {_CKPT}")

    run_id = "phase7_pilot_smoke"
    pilot_main([
        "--run-id",     run_id,
        "--N",          "8",
        "--max-iters",  "20",
        "--trunk-ckpt", _CKPT,
        "--data-root",  str(tmp_path),
    ])

    run_dir = tmp_path / "runs" / run_id
    timing = json.loads((run_dir / "pilot_timing.json").read_text())
    stats  = json.loads((run_dir / "pilot_solver_stats.json").read_text())

    assert timing["N"] == 8
    assert timing["n_deals"] == 8
    assert 0.0 <= stats["frac_converged"] <= 1.0
    assert "edges" in stats["iters_hist"] and "counts" in stats["iters_hist"]
