"""AR-2 Phase 4 — CFR+ distillation pipeline.

Produces (CallPolicy, BidPolicy) training rows by:

1. Sampling deals from a hand-size mixture (parent §3.1).
2. Solving each subgame with `CFRPlusSubgameSolver`.
3. Walking the average-strategy support to emit one row per visited
   decision state per head (no reach-weighting — see Phase 4 design §2.2).
4. Sharding rows by `deal_idx % shard_count` to `.npz`.
5. Precomputing the frozen-trunk activation + HandModel posterior `q`
   once per row (`trunk_<shard>.npz`, aligned 1:1 with `bid_<shard>.npz`).

This module is a *data producer*. Training is invoked separately via the
heads' trainers (Phase 4 step 10 wires that adapter in tests).

See `docs-internal/design/agent_redesign_ar2_phase4_design.md` for the
row schemas and the full set of decisions this module operationalizes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agents.contracts import Infostate  # noqa: E402
from agents.learned.handmodel.network import LearnedHandModel  # noqa: E402
from game.bids import NUM_BIDS  # noqa: E402
from training.cfr.subgame_solver import (  # noqa: E402
    CFRPlusSubgameSolver,
    SubgameSolution,
    build_canonical_match_state,
    pool_best_bid_idx,
)
from training.metrics.deal_sampler import sample_deals  # noqa: E402

# Maximum per-seat hand size we ever pad to in the row schema.
_MAX_HAND = 5


# ---------------------------------------------------------------------------
# Step 1 — Hand-size mixture sampler
# ---------------------------------------------------------------------------

def sample_deals_mixture(
    N: int,
    seed: int,
    mix: dict[int, float] | None = None,
) -> Iterator[tuple[tuple[list[int], list[int]], int]]:
    """Yield `N` deals interleaved across per-seat hand sizes per `mix`.

    `mix` keys are per-seat hand size (`hand_size`); pool size `n = 2 * hand_size`.
    Default `mix = {2: .25, 3: .25, 4: .25, 5: .25}` → pool sizes 4, 6, 8, 10.

    Per-bucket seeds are deterministic in `seed` (`seed + k * 7919`). Buckets
    are filled lazily and yielded round-robin so consumers can early-stop on
    a balanced prefix.
    """
    if mix is None:
        mix = {2: 0.25, 3: 0.25, 4: 0.25, 5: 0.25}
    sizes = sorted(mix.keys())

    counts = {h: int(round(N * mix[h])) for h in sizes}
    # Adjust rounding drift so totals == N exactly.
    drift = N - sum(counts.values())
    if drift != 0:
        counts[sizes[0]] += drift

    iters = {
        h: sample_deals(counts[h], seed=seed + i * 7919, hand_size=h, num_players=2)
        for i, h in enumerate(sizes)
    }
    remaining = dict(counts)

    while sum(remaining.values()) > 0:
        for h in sizes:
            if remaining[h] <= 0:
                continue
            try:
                hands = next(iters[h])
            except StopIteration:
                remaining[h] = 0
                continue
            remaining[h] -= 1
            yield hands, 2 * h


# ---------------------------------------------------------------------------
# Step 2 — Walk avg-strategy → row generator
# ---------------------------------------------------------------------------

@dataclass
class _Row:
    deal_idx:      int
    state_player:  int
    cur_bid_idx:   int                # -1 = root (None)
    hand_p0:       list[int]
    hand_p1:       list[int]
    pool_size:     int
    target_call:   float              # always emitted
    target_bid:    np.ndarray | None  # None when residual mass is 0
    feasible_mask: np.ndarray | None  # bool, NUM_BIDS; aligned with target_bid
    infostate:     Infostate          # carried for trunk precompute (not serialized)


def walk_avg_strategy(
    sol: SubgameSolution,
    hands: tuple[list[int], list[int]],
    deal_idx: int,
) -> Iterator[_Row]:
    """BFS over visited decision states; emit one `_Row` per state.

    `_Row.target_call` is always present. `_Row.target_bid` is `None` when
    `1 - call_prob ≈ 0` (no probability mass on bidding actions); the
    consumer drops those from the bid shard but keeps the call shard row.

    HH-gated states are *not* decision states (the solver excludes them);
    we therefore never visit them here.
    """
    best_idx = pool_best_bid_idx(tuple(list(h) for h in hands))

    def hh_gate_at(b: int) -> bool:
        return b == best_idx

    seen: set[tuple[int | None, int]] = set()
    queue: deque[tuple[int | None, int]] = deque()
    root: tuple[int | None, int] = (None, 0)
    queue.append(root)
    seen.add(root)

    pool_size = len(hands[0]) + len(hands[1])

    while queue:
        s = queue.popleft()
        cur, cp = s

        if s not in sol.avg_call_prob:
            # Never visited under any iteration's reach — solver dropped it.
            continue

        # Build canonical Infostate at this decision point.
        match_state = build_canonical_match_state(hands, cur, cp)
        info = Infostate.from_match_state(match_state)

        call_p = float(sol.avg_call_prob[s])
        bid_vec = sol.avg_bid_dist.get(s)
        residual = 1.0 - call_p

        target_bid: np.ndarray | None = None
        feasible_mask_b: np.ndarray | None = None
        if bid_vec is not None and residual > 1e-6:
            # The solver's action set is "any bid > cur_bid_idx", which is
            # broader than the engine's hand-feasibility mask (the engine
            # also rules out bids no pool of this size can satisfy). Project
            # the solver's avg onto the engine's feasible set + renormalize
            # so the distillation target lives in the same space the
            # network's softmax outputs.
            mask_b = np.array(info.feasible_mask[:NUM_BIDS], dtype=np.bool_)
            tb = np.asarray(bid_vec, dtype=np.float32).copy()
            tb[~mask_b] = 0.0
            tb_sum = float(tb.sum())
            if tb_sum > 0.0:
                tb /= tb_sum
                target_bid = tb
                feasible_mask_b = mask_b

        yield _Row(
            deal_idx     = deal_idx,
            state_player = cp,
            cur_bid_idx  = -1 if cur is None else int(cur),
            hand_p0      = list(hands[0]),
            hand_p1      = list(hands[1]),
            pool_size    = pool_size,
            target_call  = call_p,
            target_bid   = target_bid,
            feasible_mask= feasible_mask_b,
            infostate    = info,
        )

        # Enqueue successors of any bid action with non-zero average prob.
        if bid_vec is None:
            continue
        for b in range(NUM_BIDS):
            if bid_vec[b] <= 0.0:
                continue
            if hh_gate_at(b):
                continue  # forced HH terminal
            succ = (b, 1 - cp)
            if succ in seen:
                continue
            seen.add(succ)
            queue.append(succ)


# ---------------------------------------------------------------------------
# Step 3 — Sharded `.npz` writer
# ---------------------------------------------------------------------------

def _pad_hand(hand: list[int]) -> list[int]:
    return list(hand) + [-1] * (_MAX_HAND - len(hand))


def write_shards(
    rows: list[_Row],
    run_id: str,
    *,
    shard_count: int = 64,
    data_root: str | None = None,
) -> str:
    """Group rows by `deal_idx % shard_count`; write call/bid `.npz` shards.

    Returns the directory the shards were written to.
    """
    if data_root is None:
        data_root = os.path.join(_SRC, "..", "data")
    out_dir = os.path.abspath(os.path.join(data_root, "runs", run_id, "cfr_deals"))
    os.makedirs(out_dir, exist_ok=True)

    call_buckets: dict[int, list[_Row]]  = {i: [] for i in range(shard_count)}
    bid_buckets:  dict[int, list[_Row]]  = {i: [] for i in range(shard_count)}

    for r in rows:
        sh = r.deal_idx % shard_count
        call_buckets[sh].append(r)
        if r.target_bid is not None:
            bid_buckets[sh].append(r)

    for sh, bucket in call_buckets.items():
        if not bucket:
            continue
        np.savez(
            os.path.join(out_dir, f"call_{sh:03d}.npz"),
            deal_idx     = np.array([r.deal_idx     for r in bucket], dtype=np.int32),
            state_player = np.array([r.state_player for r in bucket], dtype=np.int8),
            cur_bid_idx  = np.array([r.cur_bid_idx  for r in bucket], dtype=np.int16),
            hand_p0      = np.array([_pad_hand(r.hand_p0) for r in bucket], dtype=np.int8),
            hand_p1      = np.array([_pad_hand(r.hand_p1) for r in bucket], dtype=np.int8),
            pool_size    = np.array([r.pool_size    for r in bucket], dtype=np.int16),
            target_call  = np.array([r.target_call  for r in bucket], dtype=np.float32),
        )

    for sh, bucket in bid_buckets.items():
        if not bucket:
            continue
        np.savez(
            os.path.join(out_dir, f"bid_{sh:03d}.npz"),
            deal_idx      = np.array([r.deal_idx     for r in bucket], dtype=np.int32),
            state_player  = np.array([r.state_player for r in bucket], dtype=np.int8),
            cur_bid_idx   = np.array([r.cur_bid_idx  for r in bucket], dtype=np.int16),
            hand_p0       = np.array([_pad_hand(r.hand_p0) for r in bucket], dtype=np.int8),
            hand_p1       = np.array([_pad_hand(r.hand_p1) for r in bucket], dtype=np.int8),
            pool_size     = np.array([r.pool_size    for r in bucket], dtype=np.int16),
            target_bid    = np.stack([r.target_bid for r in bucket]).astype(np.float32),
            feasible_mask = np.stack([r.feasible_mask for r in bucket]).astype(np.bool_),
            target_call   = np.array([r.target_call for r in bucket], dtype=np.float32),
        )

    return out_dir


def write_split_json(
    deal_indices: list[int],
    run_id: str,
    *,
    data_root: str | None = None,
) -> str:
    """Hash-stratified 80/10/10 split by deal_idx. Written once per `run_id`."""
    if data_root is None:
        data_root = os.path.join(_SRC, "..", "data")
    out_dir = os.path.abspath(os.path.join(data_root, "runs", run_id, "cfr_deals"))
    os.makedirs(out_dir, exist_ok=True)

    train, val, test = [], [], []
    for d in deal_indices:
        h = int(hashlib.sha1(f"{run_id}:{d}".encode()).hexdigest()[:8], 16) % 10
        if h < 8:
            train.append(d)
        elif h == 8:
            val.append(d)
        else:
            test.append(d)

    path = os.path.join(out_dir, "split.json")
    with open(path, "w") as f:
        json.dump({"train": train, "val": val, "test": test}, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Step 3.5 — Shared shard iterator (Phase 7)
# ---------------------------------------------------------------------------

# Phase-7 deviation: trunk_<sh>.npz aligns 1:1 with bid_<sh>.npz (not all call
# rows). For Phase 7 the call head trains on the bid-eligible subset only —
# call-only rows (where target_call ≈ 1, no real call-vs-bid decision) are
# dropped. iter_shards therefore joins bid_<sh>.npz with trunk_<sh>.npz for
# both heads; the head argument only controls which target/feature keys are
# yielded.

def iter_shards(
    run_id:     str,
    split:      str,
    *,
    head:       Literal["call", "bid"],
    data_root:  str | None = None,
    batch_size: int | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield row dicts (or stacked batches) from a distillation run.

    Joins `bid_<sh>.npz` + `trunk_<sh>.npz` (1:1 aligned), filtered to
    `deal_idx` ∈ the requested split per `split.json`.

    Row dict keys when `head="call"`:
        trunk_repr, q, standing_bid, pool_size, target_call_prob, deal_idx
    Row dict keys when `head="bid"`:
        trunk_repr, q, pool_size, feasible_mask, target_bid, deal_idx
    """
    if head not in ("call", "bid"):
        raise ValueError(f"head must be 'call' or 'bid', got {head!r}")
    if data_root is None:
        data_root = os.path.join(_SRC, "..", "data")
    run_dir = os.path.abspath(os.path.join(data_root, "runs", run_id, "cfr_deals"))

    with open(os.path.join(run_dir, "split.json")) as f:
        splits = json.load(f)
    if split not in splits:
        raise KeyError(f"split {split!r} not in split.json (keys: {list(splits)})")
    deal_set = set(int(d) for d in splits[split])

    bid_files   = sorted(f for f in os.listdir(run_dir) if f.startswith("bid_") and f.endswith(".npz"))
    pending: list[dict[str, np.ndarray]] = []

    def _flush_batch(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        keys = rows[0].keys()
        return {k: np.stack([r[k] for r in rows]) for k in keys}

    for bf in bid_files:
        sh = bf[len("bid_"):-len(".npz")]
        tf = f"trunk_{sh}.npz"
        bpath = os.path.join(run_dir, bf)
        tpath = os.path.join(run_dir, tf)
        if not os.path.exists(tpath):
            continue
        b = np.load(bpath)
        t = np.load(tpath)
        n_rows = b["deal_idx"].shape[0]
        if t["trunk_repr"].shape[0] != n_rows:
            raise RuntimeError(f"shard {sh}: bid rows {n_rows} != trunk rows {t['trunk_repr'].shape[0]}")
        keep = np.array([int(d) in deal_set for d in b["deal_idx"]], dtype=np.bool_)
        if not keep.any():
            continue
        idx = np.nonzero(keep)[0]

        trunk_repr = t["trunk_repr"][idx]
        q          = t["q"][idx]
        pool_size  = b["pool_size"][idx]
        deal_idx   = b["deal_idx"][idx]

        if head == "call":
            standing_bid = b["cur_bid_idx"][idx]
            target_call  = b["target_call"][idx]
            for i in range(len(idx)):
                row = {
                    "trunk_repr":       trunk_repr[i],
                    "q":                q[i],
                    "standing_bid":     standing_bid[i],
                    "pool_size":        pool_size[i],
                    "target_call_prob": target_call[i],
                    "deal_idx":         deal_idx[i],
                }
                if batch_size is None:
                    yield row
                else:
                    pending.append(row)
                    if len(pending) >= batch_size:
                        yield _flush_batch(pending)
                        pending = []
        else:  # head == "bid"
            feasible_mask = b["feasible_mask"][idx]
            target_bid    = b["target_bid"][idx]
            for i in range(len(idx)):
                row = {
                    "trunk_repr":    trunk_repr[i],
                    "q":             q[i],
                    "pool_size":     pool_size[i],
                    "feasible_mask": feasible_mask[i],
                    "target_bid":    target_bid[i],
                    "deal_idx":      deal_idx[i],
                }
                if batch_size is None:
                    yield row
                else:
                    pending.append(row)
                    if len(pending) >= batch_size:
                        yield _flush_batch(pending)
                        pending = []

    if batch_size is not None and pending:
        yield _flush_batch(pending)


# ---------------------------------------------------------------------------
# Step 4 — Trunk activation precompute pass
# ---------------------------------------------------------------------------

def precompute_trunk(
    rows: list[_Row],
    run_id: str,
    trunk_ckpt: str,
    *,
    shard_count: int = 64,
    device: str = "cpu",
    data_root: str | None = None,
) -> str:
    """Write `trunk_<shard>.npz` aligned 1:1 with `bid_<shard>.npz`."""
    from agents.learned.callpolicy.network import _trunk_forward
    if data_root is None:
        data_root = os.path.join(_SRC, "..", "data")
    out_dir = os.path.abspath(os.path.join(data_root, "runs", run_id, "cfr_deals"))
    os.makedirs(out_dir, exist_ok=True)

    trunk = LearnedHandModel.from_checkpoint(trunk_ckpt, device=device)

    bid_rows = [r for r in rows if r.target_bid is not None]
    buckets: dict[int, list[_Row]] = {i: [] for i in range(shard_count)}
    for r in bid_rows:
        buckets[r.deal_idx % shard_count].append(r)

    import torch
    dev = torch.device(device)

    for sh, bucket in buckets.items():
        if not bucket:
            continue
        infos = [r.infostate for r in bucket]
        with torch.no_grad():
            trunk_repr = _trunk_forward(trunk, infos, device=dev).cpu().numpy().astype(np.float32)
        beliefs = trunk.belief_batch(infos)
        q       = np.stack([b.q for b in beliefs]).astype(np.float32)
        np.savez(
            os.path.join(out_dir, f"trunk_{sh:03d}.npz"),
            trunk_repr = trunk_repr,
            q          = q,
        )

    return out_dir


# ---------------------------------------------------------------------------
# Step 5 — Top-level entry point
# ---------------------------------------------------------------------------

def run_distillation(
    N:           int,
    seed:        int,
    run_id:      str,
    trunk_ckpt:  str,
    *,
    max_iters:   int = 500,
    mix:         dict[int, float] | None = None,
    shard_count: int = 64,
    device:      str = "cpu",
    data_root:   str | None = None,
) -> dict:
    """Sample → solve → walk → shard → precompute trunk. Returns summary dict."""
    solver = CFRPlusSubgameSolver(max_iters=max_iters)

    all_rows:  list[_Row] = []
    deal_idxs: list[int]  = []
    for deal_idx, (hands, _n) in enumerate(sample_deals_mixture(N, seed=seed, mix=mix)):
        deal_idxs.append(deal_idx)
        sol = solver.solve(hands)
        for row in walk_avg_strategy(sol, hands, deal_idx):
            all_rows.append(row)

    out_dir = write_shards(all_rows, run_id, shard_count=shard_count, data_root=data_root)
    write_split_json(deal_idxs, run_id, data_root=data_root)
    precompute_trunk(all_rows, run_id, trunk_ckpt,
                     shard_count=shard_count, device=device, data_root=data_root)

    return {
        "run_id":     run_id,
        "out_dir":    out_dir,
        "n_deals":    len(deal_idxs),
        "n_rows":     len(all_rows),
        "n_bid_rows": sum(1 for r in all_rows if r.target_bid is not None),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    p = argparse.ArgumentParser(description="AR-2 Phase 4 distillation pipeline")
    p.add_argument("--N",           type=int, required=True)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--run-id",      type=str, required=True)
    p.add_argument("--trunk-ckpt",  type=str, required=True)
    p.add_argument("--max-iters",   type=int, default=500)
    p.add_argument("--shard-count", type=int, default=64)
    p.add_argument("--device",      type=str, default="cpu")
    args = p.parse_args()

    summary = run_distillation(
        N=args.N, seed=args.seed, run_id=args.run_id, trunk_ckpt=args.trunk_ckpt,
        max_iters=args.max_iters, shard_count=args.shard_count, device=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()


__all__ = [
    "sample_deals_mixture",
    "walk_avg_strategy",
    "write_shards",
    "write_split_json",
    "iter_shards",
    "precompute_trunk",
    "run_distillation",
]
