"""BidPolicyConfig — AR-2 §4.3 architecture and Phase 4 training hyperparameters.

Architecture is fixed by design §4.3: 367-d input → Linear(367→128) → LN → ReLU
→ Linear(128→NUM_BIDS); add `log(q + 1e-12)` warm-start before masking + softmax.
The entropy schedule fields (`beta_max`, `floor_frac`) are declared here so the
dataclass is stable for Phase 4 wiring; the regularizer + inference floor
themselves live in trainer / inference code added in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BidPolicyConfig:
    # --- Architecture (design §4.3) ---------------------------------------
    trunk_dim:    int = 256
    q_dim:        int = 110   # NUM_BIDS
    hidden:       int = 128
    final_init_gain: float = 0.01   # orthogonal gain on final layer

    # --- Trunk loading ----------------------------------------------------
    load_trunk:   str | None = None

    # --- Entropy schedule (design §5.3; consumed in Phase 4) --------------
    beta_max:     float = 0.05
    # floor_frac[n] = inference-time minimum entropy as fraction of log(support)
    # for hand-pool size n. Empty for n ≥ 5.
    floor_frac:   dict[int, float] = field(
        default_factory=lambda: {2: 0.6, 3: 0.5, 4: 0.4}
    )

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
        # 256 trunk + 110 q + 1 scalar
        return self.trunk_dim + self.q_dim + 1

    def to_dict(self) -> dict:
        return {
            "trunk_dim":       self.trunk_dim,
            "q_dim":           self.q_dim,
            "hidden":          self.hidden,
            "final_init_gain": self.final_init_gain,
            "load_trunk":      self.load_trunk,
            "beta_max":        self.beta_max,
            "floor_frac":      dict(self.floor_frac),
            "lr":              self.lr,
            "weight_decay":    self.weight_decay,
            "batch_size":      self.batch_size,
            "max_epochs":      self.max_epochs,
            "seed":            self.seed,
            "device":          self.device,
        }

    @classmethod
    def from_dict(cls, obj: dict) -> "BidPolicyConfig":
        kwargs = dict(obj)
        if "floor_frac" in kwargs:
            kwargs["floor_frac"] = {int(k): float(v) for k, v in kwargs["floor_frac"].items()}
        return cls(**kwargs)
