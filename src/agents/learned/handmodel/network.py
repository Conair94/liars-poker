"""LearnedHandModel — AR-1 network and `HandModel` Protocol wrapper.

Two-layer surface:

- `LearnedHandModelNet` (`torch.nn.Module`) — the trainable network.
  Trainer / dataset code talks to this directly.
- `LearnedHandModel` — a thin wrapper that implements the AR-0a
  `HandModel` Protocol (`belief` / `belief_batch` returning `HandBelief`).
  Benchmark / LBR / sweep harness talk to this.

Architecture is fixed at construction by `HandModelConfig`; see
`docs-internal/design/agent_redesign_ar1.md` §2.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Paths: src/training/probs is on sys.path for poker_math_exact (engine
# imports it transitively). src/ for the agents/game packages.
_HERE      = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_PROBS_DIR = os.path.join(_SRC_DIR, "training", "probs")
for _p in (_PROBS_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.checkpoints import load_component, save_component  # noqa: E402
from agents.contracts import HandBelief, Infostate  # noqa: E402
from agents.learned.handmodel.config import HandModelConfig  # noqa: E402
from game.bids import CALL_ACTION, NUM_BIDS  # noqa: E402
from game.feasibility import feasible_bid_mask  # noqa: E402

_CALL_TOKEN = NUM_BIDS          # 110
_PAD_TOKEN  = NUM_BIDS + 1      # 111
_BID_VOCAB  = NUM_BIDS + 2      # 112

_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Tokenisation helpers (stateless; pure)
# ---------------------------------------------------------------------------

def _bid_token(action: int) -> int:
    if action == CALL_ACTION:
        return _CALL_TOKEN
    if 0 <= action < NUM_BIDS:
        return action
    # HH_ACTION (or any other) collapses to PAD — HH ends the round so
    # the model never sees it as a prior bid in the sequence.
    return _PAD_TOKEN


def encode_infostate(
    info: Infostate,
    bid_hist_len: int,
    num_seats: int,
) -> dict:
    """Return a dict of tensor-friendly arrays for one infostate.

    Shapes (numpy):
      own_hand:     (5,)  int64, padded with -1
      hand_count:   ()    int64, number of valid cards in own_hand
      bid_tokens:   (L,)  int64
      seat_tokens:  (L,)  int64
      pos_tokens:   (L,)  int64
      hist_mask:    (L,)  bool — True where token is real (not pad)
      scalars:      (5,)  float32
    """
    L = bid_hist_len
    own_hand = list(info.own_hand)[:5]
    hand_padded = own_hand + [-1] * (5 - len(own_hand))

    # Right-aligned: most recent bid at index L-1; left-padded with PAD/0.
    hist = info.bid_history[-L:]
    n_pad = L - len(hist)
    bid_tokens  = [_PAD_TOKEN] * n_pad + [_bid_token(a) for _, a in hist]
    seat_tokens = [0] * n_pad + [
        (s - info.own_seat) % num_seats for s, _ in hist
    ]
    pos_tokens  = list(range(L))
    hist_mask   = [False] * n_pad + [True] * len(hist)

    n          = info.pool_size
    own_size   = len(own_hand)
    other_sizes = [info.hand_sizes[s] for s in range(len(info.hand_sizes))
                   if s != info.own_seat and info.hand_sizes[s] > 0]
    avg_opp = (sum(other_sizes) / len(other_sizes)) if other_sizes else 0.0
    is_first = 1.0 if info.standing_bid is None else 0.0
    round_pos = min(len(info.bid_history), 20) / 20.0

    scalars = [
        n / 25.0,
        own_size / 5.0,
        avg_opp / 5.0,
        is_first,
        round_pos,
    ]

    return {
        "own_hand":    np.asarray(hand_padded,  dtype=np.int64),
        "hand_count":  np.asarray(own_size,     dtype=np.int64),
        "bid_tokens":  np.asarray(bid_tokens,   dtype=np.int64),
        "seat_tokens": np.asarray(seat_tokens,  dtype=np.int64),
        "pos_tokens":  np.asarray(pos_tokens,   dtype=np.int64),
        "hist_mask":   np.asarray(hist_mask,    dtype=np.bool_),
        "scalars":     np.asarray(scalars,      dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class LearnedHandModelNet(nn.Module):
    """Card + bid-history → (NUM_BIDS,) masked logits."""

    SCALAR_DIM = 5

    def __init__(self, config: HandModelConfig) -> None:
        super().__init__()
        self.config = config

        # Card encoder (DeepSet sum over an Embedding(52)).
        self.card_emb = nn.Embedding(53, config.card_emb_dim, padding_idx=52)

        # Bid-history encoder (None if disabled).
        self._use_history = config.bid_hist_dim > 0
        if self._use_history:
            model_dim = config.bid_emb_dim
            self.bid_emb  = nn.Embedding(_BID_VOCAB,         model_dim, padding_idx=_PAD_TOKEN)
            self.seat_emb = nn.Embedding(max(config.num_seats, 1), model_dim)
            self.pos_emb  = nn.Embedding(config.bid_hist_len, model_dim)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=config.transformer_heads,
                dim_feedforward=config.transformer_ffn_dim,
                dropout=0.0,
                batch_first=True,
                activation="relu",
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                enc_layer, num_layers=config.transformer_layers,
            )
            # Project pooled output to the configured bid-history width.
            if model_dim != config.bid_hist_dim:
                self.history_proj = nn.Linear(model_dim, config.bid_hist_dim)
            else:
                self.history_proj = nn.Identity()

        # Trunk.
        in_dim = config.card_emb_dim + (config.bid_hist_dim if self._use_history else 0) + self.SCALAR_DIM
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(config.num_trunk_layers):
            layers += [
                nn.Linear(d, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
            ]
            d = config.hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Output head over the bid space.
        self.head = nn.Linear(config.hidden_dim, NUM_BIDS)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        own_hand:     torch.Tensor,   # (B, 5) int64; -1 = pad
        bid_tokens:   torch.Tensor,   # (B, L) int64
        seat_tokens:  torch.Tensor,   # (B, L) int64
        pos_tokens:   torch.Tensor,   # (B, L) int64
        hist_mask:    torch.Tensor,   # (B, L) bool — True for real tokens
        scalars:      torch.Tensor,   # (B, 5) float32
        feasible:     torch.Tensor,   # (B, NUM_BIDS) bool
    ) -> torch.Tensor:
        """Return masked logits (B, NUM_BIDS). Infeasible positions are -inf."""
        # Card DeepSet (mask out pad cards via the embedding's padding_idx,
        # but -1 is not a valid index — remap to padding_idx 52).
        card_idx = torch.where(own_hand >= 0, own_hand, torch.full_like(own_hand, 52))
        card_vec = self.card_emb(card_idx).sum(dim=1)  # (B, card_emb_dim)

        if self._use_history:
            B, L = bid_tokens.shape
            tok = self.bid_emb(bid_tokens) + self.seat_emb(seat_tokens) + self.pos_emb(pos_tokens)
            # Transformer expects key_padding_mask=True for *padded* positions.
            key_pad = ~hist_mask  # (B, L) — True where pad
            # An all-pad row (opener: no bids yet) makes every attention
            # weight -inf and the softmax produce NaN. Keep one position
            # unmasked for those rows; the mean-pool below uses the
            # original hist_mask so the pooled output is still zero.
            all_pad = key_pad.all(dim=1)
            if bool(all_pad.any()):
                key_pad = key_pad.clone()
                key_pad[all_pad, -1] = False
            enc = self.transformer(tok, src_key_padding_mask=key_pad)
            # Mean-pool over real positions; if none, use zeros.
            real = (~key_pad).unsqueeze(-1).to(enc.dtype)
            denom = real.sum(dim=1).clamp_min(1.0)
            pooled = (enc * real).sum(dim=1) / denom
            hist_vec = self.history_proj(pooled)
        else:
            hist_vec = card_vec.new_zeros((card_vec.shape[0], 0))

        x = torch.cat([card_vec, hist_vec, scalars], dim=-1)
        x = self.trunk(x)
        logits = self.head(x)  # (B, NUM_BIDS)

        # Hard mask before softmax — gradients respect feasibility.
        return logits.masked_fill(~feasible, _NEG_INF)


# ---------------------------------------------------------------------------
# `HandModel` Protocol wrapper
# ---------------------------------------------------------------------------

class LearnedHandModel:
    """Wrap a trained `LearnedHandModelNet` behind the AR-0a Protocol."""

    def __init__(self, net: LearnedHandModelNet, *, device: str | torch.device = "cpu") -> None:
        self.net = net.to(device)
        self.net.eval()
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    # `HandModel` protocol
    # ------------------------------------------------------------------

    def belief(self, info: Infostate) -> HandBelief:
        return self.belief_batch([info])[0]

    def belief_batch(self, infos: list[Infostate]) -> list[HandBelief]:
        if not infos:
            return []
        batch = self._collate(infos)
        with torch.no_grad():
            logits = self.net(**batch)             # (B, NUM_BIDS), masked
            probs  = torch.softmax(logits, dim=-1)
        logits_np = logits.cpu().numpy().astype(np.float32)
        probs_np  = probs.cpu().numpy().astype(np.float32)

        out: list[HandBelief] = []
        for i, info in enumerate(infos):
            mask = feasible_bid_mask(info.pool_size)
            q = probs_np[i].copy()
            # Clean numerical fuzz on masked positions before validation.
            q[~mask] = 0.0
            s = float(q.sum())
            if s > 0.0:
                q /= s
            out.append(HandBelief(
                q=q.astype(np.float32),
                q_logits=logits_np[i],
                feasible_mask=mask,
                n=info.pool_size,
            ))
        return out

    # ------------------------------------------------------------------
    # Encoding helpers (also used by the trainer's dataset loader)
    # ------------------------------------------------------------------

    def _collate(self, infos: list[Infostate]) -> dict:
        cfg = self.net.config
        rows = [encode_infostate(info, cfg.bid_hist_len, cfg.num_seats) for info in infos]

        own_hand    = torch.from_numpy(np.stack([r["own_hand"]    for r in rows])).to(self.device)
        bid_tokens  = torch.from_numpy(np.stack([r["bid_tokens"]  for r in rows])).to(self.device)
        seat_tokens = torch.from_numpy(np.stack([r["seat_tokens"] for r in rows])).to(self.device)
        pos_tokens  = torch.from_numpy(np.stack([r["pos_tokens"]  for r in rows])).to(self.device)
        hist_mask   = torch.from_numpy(np.stack([r["hist_mask"]   for r in rows])).to(self.device)
        scalars     = torch.from_numpy(np.stack([r["scalars"]     for r in rows])).to(self.device)

        feasible = torch.zeros((len(infos), NUM_BIDS), dtype=torch.bool, device=self.device)
        for i, info in enumerate(infos):
            feasible[i] = torch.from_numpy(feasible_bid_mask(info.pool_size)).to(self.device)

        return {
            "own_hand":    own_hand,
            "bid_tokens":  bid_tokens,
            "seat_tokens": seat_tokens,
            "pos_tokens":  pos_tokens,
            "hist_mask":   hist_mask,
            "scalars":     scalars,
            "feasible":    feasible,
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O via the AR-0a contract
    # ------------------------------------------------------------------

    def save(self, path: str, *, iter: int, parent_run: str | None = None) -> None:
        save_component(
            path,
            component="handmodel",
            config=self.net.config.to_dict(),
            state_dict={k: v.cpu() for k, v in self.net.state_dict().items()},
            iter=iter,
            parent_run=parent_run,
        )

    @classmethod
    def from_checkpoint(cls, path: str, *, device: str = "cpu") -> "LearnedHandModel":
        blob = load_component(path)
        if blob["component"] != "handmodel":
            raise ValueError(f"checkpoint at {path} is component={blob['component']!r}, expected 'handmodel'")
        cfg = HandModelConfig.from_dict(blob["config"])
        net = LearnedHandModelNet(cfg)
        net.load_state_dict(blob["state_dict"])
        return cls(net, device=device)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def make_lr_lambda(warmup_steps: int):
    """Linear warmup, constant after. `step` is the optimizer step index."""
    warmup = max(1, int(warmup_steps))

    def _lr(step: int) -> float:
        if step >= warmup:
            return 1.0
        return float(step) / float(warmup)

    return _lr


__all__ = [
    "LearnedHandModelNet",
    "LearnedHandModel",
    "encode_infostate",
    "make_lr_lambda",
]


# Silence unused-import warnings for re-exports.
_ = math
_ = F
