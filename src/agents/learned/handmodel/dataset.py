"""Phase-A rollout collector + dataset loader for AR-1.

`collect_rollouts` plays 1v1 single-round games at fixed hand size
under a frozen agent and writes one JSONL row per decision:

    {"infostate": <Infostate JSON>, "label": <bid index>, "deal_id": int}

`HandModelDataset` is a tiny `torch.utils.data.Dataset` that streams
those rows and emits the encoded tensor batch for `LearnedHandModelNet`.
Encoding lives in `network.encode_infostate` so the trainer and the
inference wrapper agree byte-for-byte.

The split discipline is by deal: callers pass disjoint (start, count)
ranges per split so no decisions from the same deal cross splits.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

_HERE      = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_PROBS_DIR = os.path.join(_SRC_DIR, "training", "probs")
for _p in (_PROBS_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.contracts import Infostate  # noqa: E402
from agents.learned.handmodel.network import encode_infostate  # noqa: E402
from agents.registry import build_agent  # noqa: E402
from game.bids import Bid, NUM_BIDS, bid_to_index  # noqa: E402
from game.engine import MatchState, new_match  # noqa: E402
from game.feasibility import feasible_bid_mask  # noqa: E402


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def _set_hand_sizes(state: MatchState, n: int) -> None:
    """Force both seats to hand size `n` for a single-round Phase-A deal."""
    for s in range(state.num_players):
        state.hand_sizes[s] = n
        state.active[s] = True


def _pool_best_index(state: MatchState) -> int:
    """Return the bid index of the resolved pool's best 5-card hand.

    Reads the most recent `RoundResult.pool_best`. Caller must have
    just stepped the round to terminal.
    """
    if not state.round_history:
        raise RuntimeError("round_history empty — round did not resolve")
    pb = state.round_history[-1].pool_best
    return bid_to_index(Bid(pb.hand_type, pb.primary_rank))


@dataclass
class RolloutDecision:
    infostate: Infostate
    label:     int
    deal_id:   int


def collect_rollouts(
    *,
    agent_key: str,
    hand_size: int,
    n_decisions: int,
    seed: int,
    deal_id_offset: int = 0,
) -> Iterable[RolloutDecision]:
    """Yield decisions until `n_decisions` have been emitted.

    Plays single-round 1v1 deals at `hand_size` cards per seat under
    `agent_key` (both seats use the same agent). The pool best hand is
    determined by playing the round to resolution; we record an
    Infostate at every actor turn before that resolution.
    """
    if n_decisions <= 0:
        return

    deal_id = deal_id_offset
    yielded = 0
    rng = random.Random(seed)

    while yielded < n_decisions:
        # Fresh per-deal seed for full reproducibility.
        deal_seed = rng.randrange(1 << 31)
        agent_a = build_agent(agent_key)
        agent_b = build_agent(agent_key)
        agents = [agent_a, agent_b]

        state = new_match(
            num_players=2,
            seed=deal_seed,
            mode="countup",
            exact_rules=True,
            high_hand=True,
        )
        _set_hand_sizes(state, hand_size)
        state.start_next_round()

        # Capture infostates for every decision; resolve the round; then
        # emit (infostate, pool_best_index) for each captured decision.
        captured: list[Infostate] = []
        # Defensive cap: 1v1 rounds rarely exceed ~30 bids; 200 is generous.
        for _ in range(200):
            if state.round_state is None:
                break
            info = Infostate.from_match_state(state)
            captured.append(info)
            actor = state.current_player()
            agent = agents[actor]
            action = agent.choose_action(state)
            state.apply_action(action)

        if state.round_state is not None:
            # Round did not resolve within the cap — skip; do not pollute
            # the dataset with truncated rounds.
            continue

        try:
            label = _pool_best_index(state)
        except Exception:
            continue

        for info in captured:
            mask = feasible_bid_mask(info.pool_size)
            if not bool(mask[label]):
                # The label must always be feasible at the actor's pool size.
                # 1v1 single-round means pool_size == 2*hand_size for both
                # decisions in the deal, so this is a sanity guard.
                continue
            yield RolloutDecision(infostate=info, label=label, deal_id=deal_id)
            yielded += 1
            if yielded >= n_decisions:
                break
        deal_id += 1


# ---------------------------------------------------------------------------
# JSONL.gz I/O
# ---------------------------------------------------------------------------

def write_jsonl_gz(path: str, decisions: Iterable[RolloutDecision]) -> int:
    """Append-write decisions; return count written."""
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for d in decisions:
            row = {
                "infostate": d.infostate.to_json(),
                "label":     int(d.label),
                "deal_id":   int(d.deal_id),
            }
            f.write(json.dumps(row, separators=(",", ":")))
            f.write("\n")
            n += 1
    return n


def read_jsonl_gz(path: str) -> list[RolloutDecision]:
    out: list[RolloutDecision] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            info = Infostate.from_json(obj["infostate"])
            out.append(RolloutDecision(
                infostate=info,
                label=int(obj["label"]),
                deal_id=int(obj["deal_id"]),
            ))
    return out


# ---------------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------------

class HandModelDataset(Dataset):
    """In-memory dataset of pre-encoded rollouts.

    Holds a list of `RolloutDecision`. `__getitem__` runs the encoder
    inline; for ~1M rows this is dominated by the underlying encode
    work, not Python overhead, and keeps the dataset trivially picklable
    for `DataLoader` workers.
    """

    def __init__(
        self,
        decisions: list[RolloutDecision],
        *,
        bid_hist_len: int,
        num_seats: int,
    ) -> None:
        self._decisions = decisions
        self._bid_hist_len = bid_hist_len
        self._num_seats = num_seats

    def __len__(self) -> int:
        return len(self._decisions)

    def __getitem__(self, idx: int) -> dict:
        d = self._decisions[idx]
        enc = encode_infostate(d.infostate, self._bid_hist_len, self._num_seats)
        feasible = feasible_bid_mask(d.infostate.pool_size)
        out = {k: torch.from_numpy(v) for k, v in enc.items()}
        out["feasible"] = torch.from_numpy(feasible)
        out["label"]    = torch.tensor(int(d.label), dtype=torch.long)
        out["pool_size"] = torch.tensor(int(d.infostate.pool_size), dtype=torch.long)
        return out


def collate_batch(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    out: dict = {}
    for k in keys:
        if batch[0][k].ndim == 0:
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


# ---------------------------------------------------------------------------
# Convenience: build splits to a directory tree
# ---------------------------------------------------------------------------

def build_phase_a_dataset(
    *,
    out_dir: str,
    agent_key: str,
    hand_sizes: tuple[int, ...],
    train_per_n: int,
    val_per_n: int,
    test_per_n: int,
    seed: int,
) -> dict[str, dict[int, str]]:
    """Run rollouts for every (split, n) and write JSONL.gz files.

    Returns a {split: {n: path}} map for the trainer to load.
    """
    paths: dict[str, dict[int, str]] = {"train": {}, "val": {}, "test": {}}
    rng = random.Random(seed)
    deal_offset = 0

    for n in hand_sizes:
        for split, count in [("train", train_per_n), ("val", val_per_n), ("test", test_per_n)]:
            split_seed = rng.randrange(1 << 31)
            split_dir = Path(out_dir) / split
            split_dir.mkdir(parents=True, exist_ok=True)
            path = split_dir / f"n{n}.jsonl.gz"
            written = write_jsonl_gz(
                str(path),
                collect_rollouts(
                    agent_key=agent_key,
                    hand_size=n,
                    n_decisions=count,
                    seed=split_seed,
                    deal_id_offset=deal_offset,
                ),
            )
            deal_offset += written + 1  # +1 for safety; deal IDs are advisory
            paths[split][n] = str(path)
    return paths


def load_split(paths: dict[int, str]) -> list[RolloutDecision]:
    rows: list[RolloutDecision] = []
    for _, p in sorted(paths.items()):
        rows.extend(read_jsonl_gz(p))
    return rows


# ---------------------------------------------------------------------------
# Brier helper (used by trainer + tests)
# ---------------------------------------------------------------------------

def brier_per_n(
    qs: np.ndarray,         # (B, NUM_BIDS) float32 — masked posteriors
    labels: np.ndarray,     # (B,) int64
    pool_sizes: np.ndarray, # (B,) int64
) -> dict[int, float]:
    """Return {n: mean Brier} for each unique pool size in the batch."""
    out: dict[int, float] = {}
    for n in np.unique(pool_sizes):
        mask = pool_sizes == n
        q = qs[mask]
        y = labels[mask]
        one_hot = np.zeros_like(q)
        one_hot[np.arange(len(y)), y] = 1.0
        diff = q - one_hot
        out[int(n)] = float((diff * diff).sum(axis=1).mean())
    return out


__all__ = [
    "RolloutDecision",
    "collect_rollouts",
    "write_jsonl_gz",
    "read_jsonl_gz",
    "HandModelDataset",
    "collate_batch",
    "build_phase_a_dataset",
    "load_split",
    "brier_per_n",
]
