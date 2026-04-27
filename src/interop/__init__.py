"""Interop layer: adapters bridging the project engine to external frameworks.

Importing this package registers the project's games with OpenSpiel:
  - ``python_liars_poker_kuhn``  — small-game oracle (3-card deck, 1 card each)
  - ``python_liars_poker_exact`` — single-round adapter for the 52-card,
    exact-rules variant at configurable hand size (Q2 target = hand_size 5)
"""

from . import openspiel_adapter  # noqa: F401  (registers games on import)
