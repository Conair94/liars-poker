"""Regression guard on the OpenSpiel CFR solver applied to our Kuhn variant.

Exit-criterion (TRAINING_PIPELINE_PLAN.md P4): CFR+ on the Kuhn-sized variant
converges to < 1e-3 exploitability within a fixed iteration budget.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from training.metrics.exploitability import (  # noqa: E402
    compute_exploitability,
    kuhn_cfr_plus_solve,
)


def test_kuhn_cfr_converges_under_threshold():
    game, policy = kuhn_cfr_plus_solve(iterations=1000)
    expl = compute_exploitability(game, policy)
    assert expl < 1e-3, f"expected exploitability < 1e-3, got {expl:.6f}"


def test_kuhn_cfr_monotone_improvement():
    """Sanity: more iterations ⇒ lower (or equal) exploitability on Kuhn."""
    _, p_low = kuhn_cfr_plus_solve(iterations=50)
    _, p_high = kuhn_cfr_plus_solve(iterations=500)
    import pyspiel
    g = pyspiel.load_game("python_liars_poker_kuhn")
    e_low = compute_exploitability(g, p_low)
    e_high = compute_exploitability(g, p_high)
    assert e_high < e_low, f"expected {e_high} < {e_low}"
