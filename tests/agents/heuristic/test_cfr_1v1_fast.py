"""
Equivalence + correctness tests for the vectorized CFRSolverFast.

Verifies the fast solver against the reference CFRSolver on a small-tree
configuration (max_bids=3) where the reference runs fast enough to compare.

Fast solver floors regrets once per iteration (canonical CFR+), while the
reference floors per-rank-pair within an iteration; bit-identical match is
not expected, but average strategies and exploitability should converge to
the same Nash equilibrium. Tolerances reflect that reality.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.heuristic.cfr_1v1 import CFRSolver
from agents.heuristic.cfr_1v1_fast import CFRSolverFast

# ---------------------------------------------------------------------------
# Tree construction tests
# ---------------------------------------------------------------------------

def test_tree_size_max_bids_3_hh():
    """Precomputed tree dimensions for max_bids=3, HC+Pair, HH enabled."""
    solver = CFRSolverFast(max_bids=3, include_hh=True)
    assert solver._num_internal > 0
    # Root exists and is P0
    assert int(solver._node_player[solver._topo_order[0]]) == 0
    # Root legal actions = full bid space (26 HC+Pair bids)
    root_id = int(solver._topo_order[0])
    assert len(solver._node_legal[root_id]) == 26


def test_tree_depth_bounded():
    """No node exceeds max_bids depth."""
    solver = CFRSolverFast(max_bids=3, include_hh=True)
    assert solver._node_depth.max() <= 3


# ---------------------------------------------------------------------------
# Algorithmic correctness — strategy matching
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_opening_mix_converges_to_reference():
    """Both solvers produce similar opening mixes after enough iterations.

    Uses max_bids=3 to keep the reference solver tractable (2000 iters take
    ~2 minutes).  Tolerance reflects the per-iteration vs per-rank-pair floor
    divergence, which washes out at convergence.
    """
    N = 500
    max_bids = 3

    slow = CFRSolver(max_bids=max_bids, include_hh=True)
    slow.run(N)

    fast = CFRSolverFast(max_bids=max_bids, include_hh=True)
    fast.run(N)

    slow_mix = slow.opening_mix_by_rank()
    fast_mix = fast.opening_mix_by_rank()

    # Compare the top-1 opening bid at each rank — it should agree.
    disagreements = 0
    for r in range(13):
        slow_top = max(slow_mix[r].items(), key=lambda kv: kv[1])[0] if slow_mix[r] else None
        fast_top = max(fast_mix[r].items(), key=lambda kv: kv[1])[0] if fast_mix[r] else None
        if slow_top != fast_top:
            disagreements += 1
    # Allow up to 2 disagreements at 500 iter (rare mixed-strategy ties flip).
    assert disagreements <= 2, \
        f"Opening mixes disagree at top choice for {disagreements}/13 ranks"


@pytest.mark.slow
def test_exploitability_converges():
    """Exploitability under fast solver trends downward and matches reference
    within a reasonable tolerance at moderate iteration counts."""
    N = 500
    slow = CFRSolver(max_bids=3, include_hh=True)
    slow.run(N)
    fast = CFRSolverFast(max_bids=3, include_hh=True)
    fast.run(N)

    # Both should be well below the baseline (random play exploitability ~= 1.0)
    exp_slow = slow.exploitability()
    exp_fast = fast.exploitability()
    assert exp_slow < 1.0
    assert exp_fast < 1.0
    # And reasonably close to each other (both approach Nash at similar rate)
    assert abs(exp_slow - exp_fast) < 0.5, (
        f"Exploitability gap too large: slow={exp_slow}, fast={exp_fast}"
    )


# ---------------------------------------------------------------------------
# Speed sanity check — don't regress past the profiled baseline
# ---------------------------------------------------------------------------

def test_runs_one_iter_quickly():
    """A single iteration at max_bids=3 finishes in < 1s (regression guard)."""
    import time
    solver = CFRSolverFast(max_bids=3, include_hh=True)
    t0 = time.time()
    solver.iterate()
    assert time.time() - t0 < 1.0


# ---------------------------------------------------------------------------
# Exploitability passes under average strategy at root
# ---------------------------------------------------------------------------

def test_game_value_bounded():
    """P0 expected utility under avg strategy is in [-1, +1]."""
    solver = CFRSolverFast(max_bids=3, include_hh=True)
    solver.run(50)
    gv = solver.game_value()
    assert -1.0 <= gv <= 1.0


def test_avg_strategy_sums_to_one():
    """At every info-set, the average strategy sums to 1 per rank row."""
    solver = CFRSolverFast(max_bids=3, include_hh=True)
    solver.run(20)
    for node_id in range(solver._num_internal):
        avg = solver._avg_strategy_for(node_id)
        sums = avg.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------

def test_to_from_dict_roundtrip():
    solver = CFRSolverFast(max_bids=3, include_hh=True)
    solver.run(20)
    d = solver.to_dict()
    reloaded = CFRSolverFast.from_dict(d)
    assert reloaded._iterations == solver._iterations
    # Regret and strategy arrays preserved
    for a, b in zip(solver._regret_sum, reloaded._regret_sum):
        assert np.allclose(a, b, atol=1e-5)
    for a, b in zip(solver._strategy_sum, reloaded._strategy_sum):
        assert np.allclose(a, b, atol=1e-5)
    # Game value preserved
    assert abs(solver.game_value() - reloaded.game_value()) < 1e-5
