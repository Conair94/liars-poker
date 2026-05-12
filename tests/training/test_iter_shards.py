"""AR-2 Phase 7 — iter_shards unit test."""

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

from training.cfr_distillation import iter_shards, run_distillation

_CKPT = os.path.join(
    _REPO, "data/runs/ar1-20260430T030730Z-b64-h256-n2-fb17d031/handmodel/best.pt"
)


def _maybe_skip_no_trunk():
    if not os.path.exists(_CKPT):
        pytest.skip(f"AR-1 trunk checkpoint not present at {_CKPT}")


@pytest.mark.slow
def test_iter_shards_split_and_keys(tmp_path):
    _maybe_skip_no_trunk()
    run_id = "phase7_iter_shards_smoke"
    summary = run_distillation(
        N=8, seed=0, run_id=run_id, trunk_ckpt=_CKPT,
        max_iters=20, shard_count=4, device="cpu",
        data_root=str(tmp_path),
    )
    assert summary["n_bid_rows"] > 0

    with open(os.path.join(summary["out_dir"], "split.json")) as f:
        splits = json.load(f)
    assert set(splits.keys()) == {"train", "val", "test"}
    all_deals = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    # Disjointness.
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(all_deals)

    # Call head rows.
    call_rows = list(
        iter_shards(run_id, "train", head="call", data_root=str(tmp_path))
    ) + list(iter_shards(run_id, "val", head="call", data_root=str(tmp_path))) \
      + list(iter_shards(run_id, "test", head="call", data_root=str(tmp_path)))
    if call_rows:
        keys = set(call_rows[0].keys())
        assert keys == {"trunk_repr", "q", "standing_bid", "pool_size", "target_call_prob", "deal_idx"}
        assert "target_bid" not in keys

    # Bid head rows.
    bid_rows = list(
        iter_shards(run_id, "train", head="bid", data_root=str(tmp_path))
    ) + list(iter_shards(run_id, "val", head="bid", data_root=str(tmp_path))) \
      + list(iter_shards(run_id, "test", head="bid", data_root=str(tmp_path)))
    if bid_rows:
        keys = set(bid_rows[0].keys())
        assert keys == {"trunk_repr", "q", "pool_size", "feasible_mask", "target_bid", "deal_idx"}
        assert "target_call_prob" not in keys

    # Call rows ⊇ bid rows under our deviation (both subset to bid-eligible).
    assert len(call_rows) == len(bid_rows)
    assert len(call_rows) > 0


@pytest.mark.slow
def test_iter_shards_batching(tmp_path):
    _maybe_skip_no_trunk()
    run_id = "phase7_iter_shards_batch"
    run_distillation(
        N=8, seed=0, run_id=run_id, trunk_ckpt=_CKPT,
        max_iters=20, shard_count=4, device="cpu",
        data_root=str(tmp_path),
    )
    batches = list(iter_shards(
        run_id, "train", head="bid", data_root=str(tmp_path), batch_size=4,
    ))
    if not batches:
        pytest.skip("no train rows in synthetic split")
    for b in batches:
        assert b["trunk_repr"].ndim == 2
        assert b["target_bid"].ndim == 2
        assert b["trunk_repr"].shape[0] == b["target_bid"].shape[0]
