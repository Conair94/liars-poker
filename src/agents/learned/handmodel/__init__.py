"""Modular HandModel package (AR-1).

Public surface:

- `HandModelConfig` — hyperparameters.
- `LearnedHandModel` — `HandModel` Protocol implementer wrapping the
  trained network; loadable via `LearnedHandModel.from_checkpoint(path)`.
- `AnalyticHandModel`, `HeuristicHandModel` — baseline adapters.

See `docs-internal/design/agent_redesign_ar1.md` for the architecture
and training plan.
"""

from agents.learned.handmodel.baselines import AnalyticHandModel, HeuristicHandModel
from agents.learned.handmodel.config import HandModelConfig
from agents.learned.handmodel.network import LearnedHandModel, LearnedHandModelNet

__all__ = [
    "HandModelConfig",
    "LearnedHandModel",
    "LearnedHandModelNet",
    "AnalyticHandModel",
    "HeuristicHandModel",
]
