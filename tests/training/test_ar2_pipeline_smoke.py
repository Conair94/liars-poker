"""AR-2 Phase 7 — end-to-end per-cell pipeline smoke."""

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

from training.ar2_pipeline import _main as pipeline_main  # noqa: E402
from training.cfr_distillation import run_distillation  # noqa: E402

_CKPT = os.path.join(
    _REPO, "data/runs/ar1-20260430T030730Z-b64-h256-n2-fb17d031/handmodel/best.pt"
)


@pytest.mark.slow
def test_ar2_pipeline_smoke(tmp_path):
    if not os.path.exists(_CKPT):
        pytest.skip(f"AR-1 trunk checkpoint not present at {_CKPT}")

    # Pre-stage the holdout run (so --external-val-run-id resolves).
    holdout_run = "phase7_pipeline_holdout"
    run_distillation(
        N=8, seed=1000, run_id=holdout_run, trunk_ckpt=_CKPT,
        max_iters=20, shard_count=2, device="cpu",
        data_root=str(tmp_path),
    )

    cell_run = "phase7_pipeline_cell"
    rc = pipeline_main([
        "--run-id",            cell_run,
        "--N",                 "8",
        "--trunk-ckpt",        _CKPT,
        "--holdout-run-id",    holdout_run,
        "--max-iters",         "20",
        "--callpolicy-epochs", "1",
        "--bidpolicy-epochs",  "1",
        "--data-root",         str(tmp_path),
    ])
    assert rc == 0

    run_dir = tmp_path / "runs" / cell_run
    summary = json.loads((run_dir / "ar2_summary.json").read_text())
    assert summary["N"] == 8
    assert os.path.exists(summary["callpolicy_ckpt"])
    assert os.path.exists(summary["bidpolicy_ckpt"])
    # Manifest flipped to completed.
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["completed"] is True
    assert manifest["exit_code"] == 0
