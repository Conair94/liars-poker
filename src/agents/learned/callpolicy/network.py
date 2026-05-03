"""CallPolicy — head only; trunk is loaded frozen from an AR-1 checkpoint.

Architecture (design §4.2 + feature spec §2):

    features = concat([trunk_repr(info)(256), q.q(110), one_hot(standing_bid, 110),
                       n/25(1), q.q[standing_bid](1)])  → 478
    head     = Linear(478→hidden) → LayerNorm → ReLU → Linear(hidden→1) → sigmoid

The byte layout is fixed in
`docs-internal/design/agent_redesign_ar2_feature_spec.md` §2 — feature
construction here is the single canonical implementation; tests pin offsets.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE    = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from agents.contracts import CallDecision, HandBelief, Infostate  # noqa: E402
from agents.learned.callpolicy.config import CallPolicyConfig  # noqa: E402
from agents.learned.handmodel.network import (  # noqa: E402
    LearnedHandModel,
    LearnedHandModelNet,
    encode_infostate,
)
from game.bids import NUM_BIDS  # noqa: E402


# ---------------------------------------------------------------------------
# Feature builder (numpy; trainer + inference share this)
# ---------------------------------------------------------------------------

def build_call_features(
    trunk_repr:   np.ndarray,    # (B, trunk_dim) float32
    q:            np.ndarray,    # (B, NUM_BIDS) float32 — already feasibility-masked simplex
    standing_bid: np.ndarray,    # (B,) int64; -1 sentinel for None
    pool_size:    np.ndarray,    # (B,) int64
) -> np.ndarray:
    """Pack the 478-d CallPolicy input per feature spec §2. Returns (B, input_dim) float32."""
    B = trunk_repr.shape[0]
    if q.shape != (B, NUM_BIDS):
        raise ValueError(f"q shape {q.shape} expected ({B}, {NUM_BIDS})")

    one_hot = np.zeros((B, NUM_BIDS), dtype=np.float32)
    has_bid = standing_bid >= 0
    one_hot[has_bid, standing_bid[has_bid]] = 1.0

    n_scalar = (pool_size.astype(np.float32) / 25.0)[:, None]

    q_at_bid = np.zeros((B, 1), dtype=np.float32)
    if has_bid.any():
        q_at_bid[has_bid, 0] = q[has_bid, standing_bid[has_bid]]

    return np.concatenate(
        [trunk_repr.astype(np.float32), q.astype(np.float32), one_hot, n_scalar, q_at_bid],
        axis=1,
    )


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class CallPolicyNet(nn.Module):
    """One-hidden-layer MLP head. Trunk is *not* part of this module."""

    def __init__(self, config: CallPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.input_dim, config.hidden)
        self.ln  = nn.LayerNorm(config.hidden)
        self.fc2 = nn.Linear(config.hidden, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.orthogonal_(self.fc1.weight, gain=0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=0.5)
        nn.init.zeros_(self.fc2.bias)

    def _raw_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-sigmoid logits, shape (B,). Used by the trainer for BCE-with-logits."""
        h = self.fc1(x)
        h = self.ln(h)
        h = torch.relu(h)
        return self.fc2(h).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return p_call ∈ (0, 1), shape (B,)."""
        return torch.sigmoid(self._raw_logits(x))


# ---------------------------------------------------------------------------
# `CallPolicy` Protocol wrapper
# ---------------------------------------------------------------------------

class DistilledCallPolicy:
    """Wrap a trained `CallPolicyNet` + frozen trunk behind the AR-0a Protocol.

    Phase 3: implements the Protocol with an uncached forward (re-runs the
    trunk for every call). Phase 4 distillation precomputes trunk activations
    once per (deal, infostate) and trains on cached features.
    """

    def __init__(
        self,
        net:        CallPolicyNet,
        trunk:      LearnedHandModel,
        *,
        device:     str | torch.device = "cpu",
    ) -> None:
        if net.config.trunk_dim != trunk.net.config.hidden_dim:
            raise ValueError(
                f"trunk_dim {net.config.trunk_dim} != HandModel hidden {trunk.net.config.hidden_dim}"
            )
        self.net    = net.to(device)
        self.trunk  = trunk
        self.device = torch.device(device)
        self.net.eval()
        # Belt-and-braces: caller may have forgotten to freeze.
        for p in self.trunk.net.parameters():
            p.requires_grad_(False)

    def call_prob(self, info: Infostate, q: HandBelief) -> CallDecision:
        feats = self._features([info], [q])
        with torch.no_grad():
            p = self.net(torch.from_numpy(feats).to(self.device)).cpu().numpy()
        p_call = float(np.clip(p[0], 0.0, 1.0))
        bi = info.standing_bid
        return CallDecision(
            p_call=p_call,
            inputs={
                "q_at_bid": float(q.q[bi]) if bi is not None else 0.0,
                "peak_q":   float(q.q.max()),
                "n":        float(info.pool_size),
            },
        )

    # ------------------------------------------------------------------
    # Internal: trunk activation + feature pack
    # ------------------------------------------------------------------

    def _features(self, infos: list[Infostate], beliefs: list[HandBelief]) -> np.ndarray:
        trunk_repr = _trunk_forward(self.trunk, infos, device=self.device).cpu().numpy()
        q          = np.stack([b.q for b in beliefs])
        standing   = np.array([(i.standing_bid if i.standing_bid is not None else -1) for i in infos], dtype=np.int64)
        pool_size  = np.array([i.pool_size for i in infos], dtype=np.int64)
        return build_call_features(trunk_repr, q, standing, pool_size)


# ---------------------------------------------------------------------------
# Helper: shared with bidpolicy
# ---------------------------------------------------------------------------

def _trunk_forward(
    trunk:  LearnedHandModel,
    infos:  list[Infostate],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Run the frozen trunk forward and return (B, hidden_dim) float32."""
    cfg  = trunk.net.config
    rows = [encode_infostate(info, cfg.bid_hist_len, cfg.num_seats) for info in infos]

    own_hand    = torch.from_numpy(np.stack([r["own_hand"]    for r in rows])).to(device)
    bid_tokens  = torch.from_numpy(np.stack([r["bid_tokens"]  for r in rows])).to(device)
    seat_tokens = torch.from_numpy(np.stack([r["seat_tokens"] for r in rows])).to(device)
    pos_tokens  = torch.from_numpy(np.stack([r["pos_tokens"]  for r in rows])).to(device)
    hist_mask   = torch.from_numpy(np.stack([r["hist_mask"]   for r in rows])).to(device)
    scalars     = torch.from_numpy(np.stack([r["scalars"]     for r in rows])).to(device)

    net: LearnedHandModelNet = trunk.net
    with torch.no_grad():
        return net.trunk_forward(
            own_hand, bid_tokens, seat_tokens, pos_tokens, hist_mask, scalars,
        )
