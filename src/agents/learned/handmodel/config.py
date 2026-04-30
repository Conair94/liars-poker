"""HandModelConfig — AR-1 architecture and training hyperparameters.

Defaults pin the design-doc choices (§2 of agent_redesign_ar1.md):

- card_emb_dim 32, summed DeepSet over own_hand
- bid-history: 1-layer Transformer, 4 heads, model dim 32, len 16
- trunk: 2 × Linear-LN-ReLU, hidden 128
- head: Linear(hidden, NUM_BIDS), masked
- 5 raw scalar features

The architectural sweep varies `hidden_dim`, `num_trunk_layers`,
`bid_hist_dim` (∈ {0, 64} — 0 disables the bid-history encoder).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HandModelConfig:
    # --- Encoders ---------------------------------------------------------
    card_emb_dim:    int = 32
    bid_emb_dim:     int = 32
    seat_emb_dim:    int = 8
    pos_emb_dim:     int = 32
    bid_hist_len:    int = 16
    # bid_hist_dim == 0 disables the bid-history encoder entirely (the
    # AR-1 ablation cell). When > 0 the Transformer pools to this width.
    bid_hist_dim:    int = 64
    # Transformer over the bid history. Only used when bid_hist_dim > 0.
    transformer_heads:    int = 4
    transformer_ffn_dim:  int = 64
    transformer_layers:   int = 1

    # --- Trunk + head -----------------------------------------------------
    hidden_dim:       int = 128
    num_trunk_layers: int = 2

    # --- Game / dataset constraints ---------------------------------------
    num_seats: int = 5  # max seats handled by the seat-offset embedding (5p ready)

    # --- Optim ------------------------------------------------------------
    lr:           float = 1e-3
    weight_decay: float = 0.0
    batch_size:   int   = 512
    warmup_steps: int   = 1000
    max_epochs:   int   = 30
    min_epochs:   int   = 5
    early_stop_patience: int   = 3
    early_stop_rel_tol:  float = 0.005   # 0.5% relative improvement

    # --- Dataset (Phase A pretrain) --------------------------------------
    rollout_agent: str = "exact_adaptive"
    hand_sizes:    tuple[int, ...] = (2, 3, 4, 5)
    train_decisions_per_n: int = 250_000   # 4 × 250k = 1.0M
    val_decisions_per_n:   int =  25_000
    test_decisions_per_n:  int =  25_000

    # --- Misc -------------------------------------------------------------
    seed:   int = 0
    device: str = "cpu"

    def to_dict(self) -> dict:
        # Cheap, stable dict for checkpoints / configs.
        return {
            "card_emb_dim": self.card_emb_dim,
            "bid_emb_dim":  self.bid_emb_dim,
            "seat_emb_dim": self.seat_emb_dim,
            "pos_emb_dim":  self.pos_emb_dim,
            "bid_hist_len": self.bid_hist_len,
            "bid_hist_dim": self.bid_hist_dim,
            "transformer_heads":   self.transformer_heads,
            "transformer_ffn_dim": self.transformer_ffn_dim,
            "transformer_layers":  self.transformer_layers,
            "hidden_dim":          self.hidden_dim,
            "num_trunk_layers":    self.num_trunk_layers,
            "num_seats":           self.num_seats,
            "lr":                  self.lr,
            "weight_decay":        self.weight_decay,
            "batch_size":          self.batch_size,
            "warmup_steps":        self.warmup_steps,
            "max_epochs":          self.max_epochs,
            "min_epochs":          self.min_epochs,
            "early_stop_patience": self.early_stop_patience,
            "early_stop_rel_tol":  self.early_stop_rel_tol,
            "rollout_agent":       self.rollout_agent,
            "hand_sizes":          list(self.hand_sizes),
            "train_decisions_per_n": self.train_decisions_per_n,
            "val_decisions_per_n":   self.val_decisions_per_n,
            "test_decisions_per_n":  self.test_decisions_per_n,
            "seed":   self.seed,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, obj: dict) -> "HandModelConfig":
        kwargs = dict(obj)
        if "hand_sizes" in kwargs:
            kwargs["hand_sizes"] = tuple(kwargs["hand_sizes"])
        return cls(**kwargs)
