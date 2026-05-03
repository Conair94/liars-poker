"""BidPolicy — head only; trunk is loaded frozen from an AR-1 checkpoint.

Architecture (design §4.3 + feature spec §3):

    features = concat([trunk_repr(info)(256), q.q(110), n/25(1)])  → 367
    hidden   = Linear(367→128) → LN → ReLU → Linear(128→NUM_BIDS)
    logits   = hidden + log(q + 1e-12)              # warm-start
    masked   = logits.masked_fill(~bid_mask, -inf)  # bid_mask = info.feasible_mask[:NUM_BIDS]
    pi       = softmax(masked)

Final layer is initialized with a small orthogonal gain so that at init,
hidden ≈ 0 and pi ≈ softmax(log q) — the parent §4.3 prior.
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

from agents.contracts import BidDistribution, HandBelief, Infostate  # noqa: E402
from agents.learned.bidpolicy.config import BidPolicyConfig  # noqa: E402
from agents.learned.callpolicy.network import _trunk_forward  # noqa: E402  (shared helper)
from agents.learned.handmodel.network import LearnedHandModel  # noqa: E402
from game.bids import NUM_ACTIONS, NUM_BIDS  # noqa: E402

_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Feature builder (numpy; trainer + inference share this)
# ---------------------------------------------------------------------------

def build_bid_features(
    trunk_repr:  np.ndarray,    # (B, trunk_dim) float32
    q:           np.ndarray,    # (B, NUM_BIDS) float32
    pool_size:   np.ndarray,    # (B,) int64
) -> np.ndarray:
    """Pack the 367-d BidPolicy input per feature spec §3. Returns (B, input_dim) float32."""
    B = trunk_repr.shape[0]
    if q.shape != (B, NUM_BIDS):
        raise ValueError(f"q shape {q.shape} expected ({B}, {NUM_BIDS})")
    n_scalar = (pool_size.astype(np.float32) / 25.0)[:, None]
    return np.concatenate(
        [trunk_repr.astype(np.float32), q.astype(np.float32), n_scalar],
        axis=1,
    )


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class BidPolicyNet(nn.Module):
    """One-hidden-layer MLP head. Logits over NUM_BIDS (CALL/HH excluded)."""

    def __init__(self, config: BidPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.input_dim, config.hidden)
        self.ln  = nn.LayerNorm(config.hidden)
        self.fc2 = nn.Linear(config.hidden, NUM_BIDS)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.orthogonal_(self.fc1.weight, gain=0.5)
        nn.init.zeros_(self.fc1.bias)
        # Final layer: small orthogonal init so hidden ≈ 0 and the
        # log-q warm-start dominates at init.
        nn.init.orthogonal_(self.fc2.weight, gain=self.config.final_init_gain)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        x:        torch.Tensor,   # (B, input_dim)
        log_q:    torch.Tensor,   # (B, NUM_BIDS) — log(q + 1e-12)
        bid_mask: torch.Tensor,   # (B, NUM_BIDS) bool — info.feasible_mask[:NUM_BIDS]
    ) -> torch.Tensor:
        """Return masked logits over NUM_BIDS. Apply softmax at the call site."""
        h = self.fc1(x)
        h = self.ln(h)
        h = torch.relu(h)
        hidden = self.fc2(h)
        logits = hidden + log_q
        return logits.masked_fill(~bid_mask, _NEG_INF)


# ---------------------------------------------------------------------------
# `BidPolicy` Protocol wrapper
# ---------------------------------------------------------------------------

class DistilledBidPolicy:
    """Wrap a trained `BidPolicyNet` + frozen trunk behind the AR-0a Protocol.

    Phase 3: uncached forward. Phase 4 distillation precomputes trunk
    activations once per (deal, infostate) and trains on cached features.
    Inference-time entropy floor (design §5.3) is wired in Phase 4.
    """

    def __init__(
        self,
        net:    BidPolicyNet,
        trunk:  LearnedHandModel,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if net.config.trunk_dim != trunk.net.config.hidden_dim:
            raise ValueError(
                f"trunk_dim {net.config.trunk_dim} != HandModel hidden {trunk.net.config.hidden_dim}"
            )
        self.net    = net.to(device)
        self.trunk  = trunk
        self.device = torch.device(device)
        self.net.eval()
        for p in self.trunk.net.parameters():
            p.requires_grad_(False)

    def bid_dist(
        self,
        info:     Infostate,
        q:        HandBelief,
        *,
        hh_fired: bool,
    ) -> BidDistribution:
        legal_mask    = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        legal_mask[list(info.legal_actions)] = True
        feasible_mask = np.array(info.feasible_mask, dtype=np.bool_)

        if hh_fired:
            return BidDistribution.empty(legal_mask=legal_mask, feasible_mask=feasible_mask)

        feats = self._features([info], [q])
        log_q = np.log(q.q + 1e-12).astype(np.float32)[None, :]
        bid_mask_np = feasible_mask[:NUM_BIDS][None, :]

        with torch.no_grad():
            logits_b = self.net(
                torch.from_numpy(feats).to(self.device),
                torch.from_numpy(log_q).to(self.device),
                torch.from_numpy(bid_mask_np).to(self.device),
            ).cpu().numpy()[0]   # (NUM_BIDS,)

        pi_bids = _stable_softmax(logits_b)

        # Inference-time entropy floor (design §5.3). Applies only at
        # n ≤ 4 (i.e. when `n` is a key of `floor_frac`).
        feasible_bids = feasible_mask[:NUM_BIDS]
        pi_bids = apply_entropy_floor(
            pi_bids, feasible_bids, info.pool_size, self.net.config.floor_frac,
        )

        # Embed back into NUM_ACTIONS-shaped vectors expected by the contract.
        pi_actions     = np.zeros(NUM_ACTIONS, dtype=np.float32)
        logits_actions = np.full(NUM_ACTIONS, _NEG_INF, dtype=np.float32)
        pi_actions[:NUM_BIDS]     = pi_bids
        logits_actions[:NUM_BIDS] = logits_b

        support = int(np.count_nonzero(pi_bids > 0.0))
        entropy = float(-(pi_bids[pi_bids > 0.0] * np.log(pi_bids[pi_bids > 0.0])).sum())

        return BidDistribution(
            pi=pi_actions,
            pi_logits=logits_actions,
            legal_mask=legal_mask,
            feasible_mask=feasible_mask,
            entropy=entropy,
            support_size=support,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _features(self, infos: list[Infostate], beliefs: list[HandBelief]) -> np.ndarray:
        trunk_repr = _trunk_forward(self.trunk, infos, device=self.device).cpu().numpy()
        q         = np.stack([b.q for b in beliefs])
        pool_size = np.array([i.pool_size for i in infos], dtype=np.int64)
        return build_bid_features(trunk_repr, q, pool_size)


def _entropy(pi: np.ndarray) -> float:
    """Shannon entropy in nats; safe on zeros."""
    nz = pi > 0.0
    if not nz.any():
        return 0.0
    return float(-(pi[nz] * np.log(pi[nz])).sum())


def apply_entropy_floor(
    pi:            np.ndarray,    # (NUM_BIDS,) — sums to 1 over feasible
    feasible_mask: np.ndarray,    # (NUM_BIDS,) bool
    n:             int,
    floor_frac:    dict[int, float],
) -> np.ndarray:
    """Mix with uniform-over-feasible until `H(pi') ≥ floor_frac[n] · H(uniform)`.

    Returns the floored distribution, or `pi` unchanged if `n` is not in
    `floor_frac` (i.e. the floor doesn't apply at this pool size) or
    `pi` already meets the floor.

    Closed-form: `pi'(α) = (1 - α) · pi + α · u` is convex in `α ∈ [0, 1]`;
    `H(pi'(α))` is monotone non-decreasing on the same interval (since
    `u` maximizes entropy on the feasible support). Bisect for the smallest
    α achieving the floor; ~14 iters to 1e-4.
    """
    if n not in floor_frac:
        return pi
    feas_count = int(feasible_mask.sum())
    if feas_count <= 1:
        return pi

    u = feasible_mask.astype(np.float32) / float(feas_count)
    H_uniform = float(np.log(feas_count))
    H_target  = floor_frac[n] * H_uniform

    if _entropy(pi) >= H_target - 1e-6:
        return pi

    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        cand = (1.0 - mid) * pi + mid * u
        if _entropy(cand) < H_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            break
    alpha = hi
    out = ((1.0 - alpha) * pi + alpha * u).astype(np.float32)
    s = float(out.sum())
    if s > 0.0:
        out /= s
    return out


def _stable_softmax(x: np.ndarray) -> np.ndarray:
    """Softmax that returns all-zero on an all-`-inf` input."""
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    m = x[finite].max()
    e = np.where(finite, np.exp(x - m), 0.0)
    s = e.sum()
    if s <= 0.0:
        return np.zeros_like(x, dtype=np.float32)
    return (e / s).astype(np.float32)
