"""AR-2 Phase 7 — BidPolicy training CLI smoke."""

from __future__ import annotations

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_SRC  = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agents.learned.bidpolicy.trainer import _main as bp_main  # noqa: E402
from training.cfr_distillation import run_distillation  # noqa: E402

_CKPT = os.path.join(
    _REPO, "data/runs/ar1-20260430T030730Z-b64-h256-n2-fb17d031/handmodel/best.pt"
)


@pytest.mark.slow
def test_bidpolicy_cli_smoke(tmp_path):
    if not os.path.exists(_CKPT):
        pytest.skip(f"AR-1 trunk checkpoint not present at {_CKPT}")

    run_id = "phase7_bidpolicy_cli_smoke"
    run_distillation(
        N=16, seed=0, run_id=run_id, trunk_ckpt=_CKPT,
        max_iters=30, shard_count=4, device="cpu",
        data_root=str(tmp_path),
    )

    out_dir = tmp_path / "out_bidpolicy"
    bp_main([
        "--run-id",     run_id,
        "--load-trunk", _CKPT,
        "--epochs",     "1",
        "--batch-size", "8",
        "--out-dir",    str(out_dir),
        "--data-root",  str(tmp_path),
    ])

    best_path = out_dir / "best.pt"
    assert best_path.exists()
    payload = torch.load(str(best_path), weights_only=False, map_location="cpu")
    assert "state_dict" in payload
    assert "val_kl" in payload
    assert "val_kl_per_n" in payload
    # All five n keys present (as strings); values may be null if no rows.
    assert set(payload["val_kl_per_n"].keys()) == {"2", "4", "6", "8", "10"}
