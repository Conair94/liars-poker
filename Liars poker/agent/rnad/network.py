"""
network.py — Policy / value network for R-NaD Liar's Poker agent.

Architecture (§5.4 of AGENT_DESIGN.md):
  • Private-card encoder  : Embedding(52, card_emb_dim), sum-pooled (DeepSet)
  • Warm-start features   : marginal_vec ∥ conditional_vec  (2×NUM_BIDS, no grad)
  • Bid/history encoder   : Embedding(NUM_BIDS+2, bid_emb_dim)
                            — one token per bid + CALL token + padding token
  • Scalar features       : pool_size, own_hand_size, avg_opp_hand_size,
                            is_first_bidder, round_position, round_index
  • Trunk                 : N × (Linear → LayerNorm → ReLU)
  • Policy head           : Linear → NUM_ACTIONS  (illegal-action mask applied)
  • Value head            : Linear → tanh → scalar ∈ (−1, 1)
  • Auxiliary head        : Linear → NUM_BIDS  (optional; predicts conditional
                            pool-hand distribution for the auxiliary loss)

The warm-start lookup is held as a non-trainable attribute so the same
checkpoint loads cleanly with or without it.

Usage
-----
net = LiarsPokerNet(config)
obs = net.encode_obs(info_state_dict)          # → Tensor (trunk_input_dim,)
logits, value = net(obs)                        # → Tensor (NUM_ACTIONS,), Tensor (1,)
action, log_p, val = net.act(info, legal)       # sample + score in one call
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_RNAD_DIR  = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.abspath(os.path.join(_RNAD_DIR, ".."))
_PAPER_DIR = os.path.abspath(os.path.join(_AGENT_DIR, ".."))
for _p in (_PAPER_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.game.bids import (   # noqa: E402
    NUM_BIDS, NUM_ACTIONS, CALL_ACTION, Bid, bid_to_index,
)
from agent.rnad.config import RNaDConfig   # noqa: E402

# Warm-start lookup is loaded lazily at first use (expensive IO).
_WARM_START_CACHE = None


def _get_warm_start():
    global _WARM_START_CACHE
    if _WARM_START_CACHE is None:
        from agent.rnad.warm_start import WarmStartLookup
        _WARM_START_CACHE = WarmStartLookup()
    return _WARM_START_CACHE


# ---------------------------------------------------------------------------
# Token indices for the bid embedding table
# ---------------------------------------------------------------------------
# 0 .. NUM_BIDS-1  → individual bid tokens
# NUM_BIDS         → CALL token  (same numeric value as CALL_ACTION = 110)
# NUM_BIDS + 1     → padding / "no action yet" token
_CALL_TOKEN    = NUM_BIDS          # 110
_PAD_TOKEN     = NUM_BIDS + 1      # 111
_BID_VOCAB     = NUM_BIDS + 2      # 112

# Number of scalar features (must match _scalar_features())
_SCALAR_DIM = 6


# ---------------------------------------------------------------------------
# Observation encoding helpers (pure functions)
# ---------------------------------------------------------------------------

def _scalar_features(info: dict) -> List[float]:
    """
    Six normalised scalar features extracted from an info_state dict.

    info is the dict returned by MatchState.info_state(seat).
    """
    active      = info["active"]
    hand_sizes  = info["hand_sizes"]
    seat        = info["seat"]

    n = sum(hand_sizes[s] for s, a in enumerate(active) if a)

    opp_sizes = [hand_sizes[s] for s, a in enumerate(active)
                 if s != seat and a]
    avg_opp = sum(opp_sizes) / max(len(opp_sizes), 1)

    is_first_bidder = 1.0 if info["current_bid"] is None else 0.0
    round_position  = len(info["bid_history"]) / 20.0   # normalised depth
    round_index     = info["round_index"] / 20.0

    return [
        n / 25.0,
        hand_sizes[seat] / 5.0,
        avg_opp / 5.0,
        is_first_bidder,
        round_position,
        round_index,
    ]


def _current_bid_token(info: dict) -> int:
    """Index into the bid embedding table for the standing bid (or PAD if none)."""
    cb = info["current_bid"]
    if cb is None:
        return _PAD_TOKEN
    return bid_to_index(Bid(cb[0], cb[1]))


def _bid_history_tokens(info: dict, hist_len: int) -> List[int]:
    """Last `hist_len` actions in this round, padded on the left with _PAD_TOKEN."""
    history = info["bid_history"]   # [(seat, action_idx), ...]
    tokens = []
    for _, act in history[-hist_len:]:
        tokens.append(_CALL_TOKEN if act == CALL_ACTION else act)
    while len(tokens) < hist_len:
        tokens.insert(0, _PAD_TOKEN)
    return tokens


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class LiarsPokerNet(nn.Module):
    """
    Policy / value network for card-based Liar's Poker.

    Parameters
    ----------
    config : RNaDConfig
        Hyperparameter object.  The network reads card_emb_dim, bid_emb_dim,
        bid_hist_len, hidden_dim, num_trunk_layers, use_warm_start, use_aux_loss.
    """

    def __init__(self, config: RNaDConfig) -> None:
        super().__init__()
        self.config = config

        # --- encoders ---------------------------------------------------
        # Card embedding: 52 cards → card_emb_dim each, sum-pooled (DeepSet)
        self.card_emb = nn.Embedding(52, config.card_emb_dim)

        # Bid / history embedding: _BID_VOCAB tokens → bid_emb_dim each
        self.bid_emb = nn.Embedding(_BID_VOCAB, config.bid_emb_dim)

        # --- compute trunk input dimension ------------------------------
        ws_dim       = 2 * NUM_BIDS if config.use_warm_start else 0
        bid_hist_dim = config.bid_emb_dim * config.bid_hist_len
        trunk_in_dim = (
            config.card_emb_dim    # card summary
            + ws_dim               # marginal ∥ conditional
            + config.bid_emb_dim   # current bid
            + bid_hist_dim         # bid history
            + _SCALAR_DIM          # scalars
        )
        self._trunk_in_dim = trunk_in_dim

        # --- trunk MLP --------------------------------------------------
        layers: List[nn.Module] = []
        in_d = trunk_in_dim
        for _ in range(config.num_trunk_layers):
            layers += [
                nn.Linear(in_d, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
            ]
            in_d = config.hidden_dim
        self.trunk = nn.Sequential(*layers)

        # --- heads ------------------------------------------------------
        self.policy_head = nn.Linear(config.hidden_dim, NUM_ACTIONS)

        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Tanh(),
        )

        if config.use_aux_loss:
            self.aux_head: Optional[nn.Module] = nn.Linear(
                config.hidden_dim, NUM_BIDS
            )
        else:
            self.aux_head = None

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
    # Observation encoding
    # ------------------------------------------------------------------

    def encode_obs(self, info: dict) -> torch.Tensor:
        """
        Convert an info_state dict (from MatchState.info_state) to a flat
        float32 Tensor of shape (trunk_in_dim,).

        Gradients flow through the card_emb and bid_emb layers.
        Warm-start features are detached (they come from a static lookup).
        """
        own_hand = info["own_hand"]
        device   = self.card_emb.weight.device

        # 1. Card summary: sum of card embeddings (DeepSet)
        if own_hand:
            card_idx_t = torch.tensor(own_hand, dtype=torch.long, device=device)
            card_sum   = self.card_emb(card_idx_t).sum(dim=0)   # (card_emb_dim,)
        else:
            card_sum = torch.zeros(self.config.card_emb_dim, device=device)

        # 2. Warm-start features (no gradient)
        if self.config.use_warm_start:
            active     = info["active"]
            hand_sizes = info["hand_sizes"]
            n = sum(hand_sizes[s] for s, a in enumerate(active) if a)
            lookup     = _get_warm_start()
            m_vec, c_vec, _ = lookup.get_features(own_hand, n)
            ws_t = torch.cat([
                torch.from_numpy(m_vec).to(device),
                torch.from_numpy(c_vec).to(device),
            ]).detach()   # static, no gradient
        else:
            ws_t = torch.zeros(0, device=device)

        # 3. Current bid embedding
        cb_tok  = torch.tensor(_current_bid_token(info), dtype=torch.long, device=device)
        cb_emb  = self.bid_emb(cb_tok)                           # (bid_emb_dim,)

        # 4. Bid history embeddings (concatenated, padded)
        hist_tokens = torch.tensor(
            _bid_history_tokens(info, self.config.bid_hist_len),
            dtype=torch.long, device=device
        )
        hist_emb = self.bid_emb(hist_tokens).flatten()           # (bid_emb_dim * K,)

        # 5. Scalar features
        scalars = torch.tensor(
            _scalar_features(info), dtype=torch.float32, device=device
        )

        return torch.cat([card_sum, ws_t, cb_emb, hist_emb, scalars])

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        obs : Tensor shape (trunk_in_dim,) or (B, trunk_in_dim)

        Returns
        -------
        policy_logits : Tensor shape matching obs batch dim, size NUM_ACTIONS
                        *before* illegal-action masking — mask externally.
        value         : Tensor shape (..., 1)
        """
        x             = self.trunk(obs)
        policy_logits = self.policy_head(x)
        value         = self.value_head(x)
        return policy_logits, value

    def forward_with_aux(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Like forward() but also returns aux_logits when aux_head is present.

        aux_logits : Tensor (NUM_BIDS,) or None — unnormalised prediction of
                     the conditional pool-hand distribution (auxiliary loss target).
        """
        x             = self.trunk(obs)
        policy_logits = self.policy_head(x)
        value         = self.value_head(x)
        aux_logits    = self.aux_head(x) if self.aux_head is not None else None
        return policy_logits, value, aux_logits

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    def act(
        self,
        info: dict,
        legal_actions: List[int],
        greedy: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Sample (or greedily select) an action given an info_state dict.

        Parameters
        ----------
        info          : info_state dict from MatchState.info_state(seat)
        legal_actions : list of legal action indices
        greedy        : if True, take argmax instead of sampling

        Returns
        -------
        action   : int
        log_prob : float  (log π(action | obs))
        value    : float  (V(obs) ∈ (−1, 1))
        """
        with torch.no_grad():
            obs     = self.encode_obs(info)
            logits, value_t = self.forward(obs)

        logits = _mask_logits(logits, legal_actions)
        dist   = Categorical(logits=logits)

        if greedy:
            action = int(logits.argmax().item())
        else:
            action = int(dist.sample().item())

        log_prob = float(dist.log_prob(torch.tensor(action)).item())
        return action, log_prob, float(value_t.item())

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def trunk_input_dim(self) -> int:
        return self._trunk_in_dim

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Action masking helper
# ---------------------------------------------------------------------------

def _mask_logits(
    logits: torch.Tensor,
    legal_actions: List[int],
) -> torch.Tensor:
    """
    Set logits for illegal actions to −∞ so they get zero probability.
    Returns a new tensor (does not modify in place).
    """
    mask   = torch.full_like(logits, float("-inf"))
    for a in legal_actions:
        mask[a] = 0.0
    return logits + mask
