"""CallPolicyConfig — AR-2 §4.2 architecture and Phase 4 training hyperparameters.

Architecture is fixed by the design: 478-d input → Linear(478→64) → LN → ReLU
→ Linear(64→1) → sigmoid. Trunk is loaded frozen via `load_trunk`. Loss/optim
fields are populated for Phase 4; Phase 3 just stubs the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CallPolicyConfig:
    # --- Architecture (design §4.2) ---------------------------------------
    trunk_dim:    int = 256   # AR-1 winner b64-h256-n2
    q_dim:        int = 110   # NUM_BIDS
    hidden:       int = 64

    # --- Trunk loading ----------------------------------------------------
    load_trunk:   str | None = None   # path to handmodel checkpoint

    # --- Optim (Phase 4) --------------------------------------------------
    lr:           float = 1e-3
    weight_decay: float = 0.0
    batch_size:   int   = 512
    max_epochs:   int   = 20

    # --- Misc -------------------------------------------------------------
    seed:   int = 0
    device: str = "cpu"

    @property
    def input_dim(self) -> int:
        # 256 trunk + 110 q + 110 one-hot(standing_bid) + 2 scalars
        return self.trunk_dim + self.q_dim + self.q_dim + 2

    def to_dict(self) -> dict:
        return {
            "trunk_dim":    self.trunk_dim,
            "q_dim":        self.q_dim,
            "hidden":       self.hidden,
            "load_trunk":   self.load_trunk,
            "lr":           self.lr,
            "weight_decay": self.weight_decay,
            "batch_size":   self.batch_size,
            "max_epochs":   self.max_epochs,
            "seed":         self.seed,
            "device":       self.device,
        }

    @classmethod
    def from_dict(cls, obj: dict) -> "CallPolicyConfig":
        return cls(**obj)
