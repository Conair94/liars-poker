"""
RNaDConfig — hyperparameters for the R-NaD trainer.

All defaults are for Stage A (N=2, fixed hand size, round-level episodes).
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RNaDConfig:
    # ------------------------------------------------------------------ #
    # Curriculum stage
    # ------------------------------------------------------------------ #
    # "A" : fixed hand size, one round per episode
    # "B" : full match with elimination (not yet implemented)
    stage: str = "A"

    # For Stage A: hand size to freeze both seats at (1..5).
    stage_a_hand_size: int = 1

    # ------------------------------------------------------------------ #
    # Game
    # ------------------------------------------------------------------ #
    num_players: int = 2

    # ------------------------------------------------------------------ #
    # Network
    # ------------------------------------------------------------------ #
    card_emb_dim: int   = 32    # dim of each card embedding (learned, DeepSet sum)
    bid_emb_dim: int    = 32    # dim of each bid / CALL / padding token embedding
    bid_hist_len: int   = 6     # number of past bid tokens in history window
    hidden_dim: int     = 256   # width of trunk MLP layers
    num_trunk_layers: int = 4   # depth of trunk MLP

    use_warm_start: bool  = True   # prepend 220-dim marginal+conditional vectors
    use_aux_loss: bool    = True   # auxiliary head predicting conditional distribution

    # ------------------------------------------------------------------ #
    # R-NaD algorithm
    # ------------------------------------------------------------------ #
    eta: float   = 0.2    # KL regularisation weight r̃ = r − η(log π − log π_reg)
    gamma: float = 1.0    # discount (1.0 is correct for short, finite rounds)

    anchor_update_freq: int = 200  # outer iterations between π_reg ← π snapshots

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    episodes_per_update: int = 128   # episodes collected before each gradient step
    num_iterations: int      = 20_000
    lr: float                = 3e-4
    grad_clip: float         = 1.0
    value_loss_coef: float   = 0.5
    aux_loss_coef: float     = 0.1   # weight of auxiliary conditional-prediction loss
    entropy_coef: float      = 0.01  # small entropy bonus for exploration

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    eval_freq: int      = 500    # outer iterations between evaluations
    eval_episodes: int  = 500    # games per evaluation

    # ------------------------------------------------------------------ #
    # Stage B — full match with elimination
    # ------------------------------------------------------------------ #
    # Maximum rounds per match episode (safety cap; real matches end much
    # sooner; with N=2, expected match length is O(hand_size²) rounds).
    max_match_rounds: int = 200

    # ------------------------------------------------------------------ #
    # Checkpointing & logging
    # ------------------------------------------------------------------ #
    checkpoint_dir: str  = "agent/checkpoints"
    checkpoint_freq: int = 2_000
    log_freq: int        = 100    # iterations between console log lines
