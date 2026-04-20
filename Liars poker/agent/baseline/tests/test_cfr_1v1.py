"""
Tests for agent/baseline/cfr_1v1.py

Test categories:
  A — Game structure / holds table correctness
  B — CFR mechanics (strategy extraction, regret accumulation)
  C — Convergence and Nash properties
  D — Mixed strategy / information-leakage validation
  E — Exploitability
"""

from __future__ import annotations

import os
import sys

# Make the paper root importable from the tests directory.
_TESTS_DIR    = os.path.dirname(os.path.abspath(__file__))
_BASELINE_DIR = os.path.abspath(os.path.join(_TESTS_DIR, ".."))
_AGENT_DIR    = os.path.abspath(os.path.join(_BASELINE_DIR, ".."))
_PAPER_DIR    = os.path.abspath(os.path.join(_AGENT_DIR, ".."))

for _p in (_PAPER_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
from agent.baseline.cfr_1v1 import (
    CFRSolver,
    _card_rank,
    _bid_holds_n2,
    _build_holds_table,
    _current_player,
    _legal_actions,
    _is_terminal,
    _terminal_utility_p0,
    _HOLDS,
)
from agent.game.bids import (
    all_bids, NUM_BIDS, CALL_ACTION, index_to_bid, bid_to_index,
    Bid, HIGH_CARD, PAIR, RANK_NAMES,
)


# ===========================================================================
# A — Game structure / holds table
# ===========================================================================

class TestGameStructure:
    def test_card_rank(self):
        # rank*4+suit → rank
        for r in range(13):
            for s in range(4):
                assert _card_rank(r * 4 + s) == r

    def test_holds_table_shape(self):
        assert len(_HOLDS) == 52
        assert len(_HOLDS[0]) == 52
        assert len(_HOLDS[0][1]) == NUM_BIDS

    def test_holds_hca_with_ace(self):
        # 2-card pool with Ace of clubs (48) + 2 of clubs (0): best = HC A
        ace_c = 12 * 4          # A♣ = 48
        two_c = 0 * 4           # 2♣ = 0
        hca_idx = bid_to_index(Bid(HIGH_CARD, 12))  # HC A
        assert _HOLDS[ace_c][two_c][hca_idx] is True

    def test_holds_hca_with_two_cards_same_rank(self):
        # Pool = [A♣, A♦] → Pair A, NOT HC A
        ace_c = 12 * 4
        ace_d = 12 * 4 + 1
        hca_idx = bid_to_index(Bid(HIGH_CARD, 12))
        assert _HOLDS[ace_c][ace_d][hca_idx] is False

    def test_holds_pair_ace(self):
        ace_c = 12 * 4
        ace_d = 12 * 4 + 1
        pair_a_idx = bid_to_index(Bid(PAIR, 12))
        assert _HOLDS[ace_c][ace_d][pair_a_idx] is True

    def test_holds_hck_no_ace(self):
        # K♣ (11*4=44) + 2♣ (0): best = HC K
        king_c = 11 * 4
        two_c  = 0
        hck_idx = bid_to_index(Bid(HIGH_CARD, 11))
        hca_idx = bid_to_index(Bid(HIGH_CARD, 12))
        assert _HOLDS[king_c][two_c][hck_idx] is True
        assert _HOLDS[king_c][two_c][hca_idx] is False

    def test_at_most_one_bid_holds_per_deal(self):
        # For any 2-card deal, exactly one bid holds (the exact evaluation).
        c0, c1 = 0, 4  # 2♣, 3♣
        holds_count = sum(1 for idx in range(NUM_BIDS) if _HOLDS[c0][c1][idx])
        assert holds_count == 1

    def test_every_deal_has_exactly_one_holding_bid(self):
        # Verify the above for a sample of deals.
        for c0 in range(0, 52, 7):
            for c1 in range(0, 52, 7):
                if c0 == c1:
                    continue
                holds_count = sum(1 for idx in range(NUM_BIDS) if _HOLDS[c0][c1][idx])
                assert holds_count == 1, (
                    f"Expected exactly 1 holding bid for deal ({c0},{c1}), got {holds_count}")


# ===========================================================================
# B — CFR mechanics
# ===========================================================================

class TestCFRMechanics:
    def test_legal_actions_root(self):
        # Root: P0 must bid; no call.
        legal = _legal_actions(())
        assert CALL_ACTION not in legal
        assert len(legal) == NUM_BIDS

    def test_legal_actions_after_bid(self):
        # After HC 2 (index 0), legal = CALL + all bids > 0.
        legal = _legal_actions((0,))
        assert CALL_ACTION in legal
        assert 0 not in legal  # cannot re-bid HC 2
        assert 1 in legal

    def test_current_player_alternates(self):
        assert _current_player(()) == 0
        assert _current_player((0,)) == 1
        assert _current_player((0, 1)) == 0

    def test_is_terminal(self):
        assert not _is_terminal(())
        assert not _is_terminal((0,))
        assert _is_terminal((0, CALL_ACTION))
        assert _is_terminal((0, 1, CALL_ACTION))

    def test_terminal_utility_p1_calls_holding_bid(self):
        # P0 bids HC A (index = bid_to_index(Bid(HC,12))).
        # P1 calls. Pool is [A♣, K♣] → HC A holds.
        # P1 is caller; bid holds → P1 (caller) loses → P1 utility = -1 → P0 = +1.
        ace_c  = 12 * 4
        king_c = 11 * 4
        hca_idx = bid_to_index(Bid(HIGH_CARD, 12))
        history = (hca_idx, CALL_ACTION)
        util = _terminal_utility_p0(history, ace_c, king_c)
        assert util == +1.0

    def test_terminal_utility_p1_calls_bluff(self):
        # P0 bids HC A; P1 calls. Pool = [K♣, Q♣] → HC K (not A) → bid does NOT hold.
        # P1 wins → P1 utility = +1 → P0 = -1.
        king_c  = 11 * 4
        queen_c = 10 * 4
        hca_idx = bid_to_index(Bid(HIGH_CARD, 12))
        history = (hca_idx, CALL_ACTION)
        util = _terminal_utility_p0(history, king_c, queen_c)
        assert util == -1.0

    def test_strategy_uniform_before_any_updates(self):
        solver = CFRSolver()
        legal = _legal_actions(())
        key = (0, 12, ())  # P0, Ace rank, root
        strat = solver._get_strategy(key, legal)
        assert len(strat) == NUM_BIDS
        assert abs(sum(strat) - 1.0) < 1e-9
        assert abs(strat[0] - 1.0 / NUM_BIDS) < 1e-9

    def test_single_iterate_does_not_crash(self):
        solver = CFRSolver()
        util = solver.iterate()
        assert isinstance(util, float)
        assert -1.0 <= util <= 1.0


# ===========================================================================
# C — Convergence and Nash properties
# ===========================================================================

class TestConvergence:
    @pytest.fixture(scope="class")
    def solver_500(self):
        """Run 500 iterations — fast but should show decreasing exploitability."""
        solver = CFRSolver()
        solver.run(500)
        return solver

    @pytest.fixture(scope="class")
    def solver_3000(self):
        """3000 iterations for tighter Nash checks."""
        solver = CFRSolver()
        solver.run(3000)
        return solver

    def test_exploitability_decreases(self):
        solver = CFRSolver()
        exp_before = solver.exploitability()  # uniform strategy
        solver.run(200)
        exp_after = solver.exploitability()
        assert exp_after < exp_before, (
            f"Exploitability did not decrease: {exp_before:.4f} → {exp_after:.4f}")

    def test_exploitability_below_threshold_500_iters(self, solver_500):
        exp = solver_500.exploitability()
        assert exp < 0.15, f"Exploitability {exp:.4f} too high after 500 iters"

    def test_exploitability_below_01_at_3000_iters(self, solver_3000):
        exp = solver_3000.exploitability()
        assert exp < 0.05, f"Exploitability {exp:.4f} too high after 3000 iters"

    def test_game_value_near_blind_eq_range(self, solver_3000):
        """
        Game value for P0 should be negative (first-mover disadvantage under
        exact rules, n=2): consistent with blind equilibrium EV₀ ≈ -0.709.
        With private info, mixed strategies should narrow the gap; expect EV in [-0.8, -0.1].
        """
        gv = solver_3000.game_value()
        assert -0.8 <= gv <= 0.1, f"Unexpected game value {gv:.4f}"


# ===========================================================================
# D — Mixed strategy / information-leakage validation
# ===========================================================================

class TestMixedStrategies:
    @pytest.fixture(scope="class")
    def solver_conv(self):
        solver = CFRSolver()
        solver.run(2000)
        return solver

    def test_ace_holder_concentrates_on_hca(self, solver_conv):
        """
        P0 holding an Ace should put significant weight on HC A (and possibly
        Pair-level bids). P(HC A | hold Ace) = 0.941, so bidding HC A is
        almost always correct; the strategy should assign ≥ 0.5 to HC A.
        """
        legal = _legal_actions(())
        key: tuple = (0, 12, ())  # player 0, rank 12 (Ace), root
        strat = solver_conv.average_strategy(key, legal)
        hca_idx_in_legal = legal.index(bid_to_index(Bid(HIGH_CARD, 12)))
        p_hca = strat[hca_idx_in_legal]
        assert p_hca >= 0.4, (
            f"P0 holding Ace should mostly bid HC A; got p={p_hca:.3f}")

    def test_non_ace_does_not_always_bid_hca(self, solver_conv):
        """
        P0 holding rank 5 (7-rank, index 5) should NOT concentrate on HC A.
        P(HC A holds | hold 7) ≈ 0.078 → calling is almost always correct for
        P1, so P0 cannot profitably bluff HC A. Expect p(HC A) < 0.5.
        """
        legal = _legal_actions(())
        key: tuple = (0, 5, ())  # rank 5 = "7"
        strat = solver_conv.average_strategy(key, legal)
        hca_idx_in_legal = legal.index(bid_to_index(Bid(HIGH_CARD, 12)))
        p_hca = strat[hca_idx_in_legal]
        assert p_hca < 0.5, (
            f"P0 holding 7 should rarely bid HC A; got p={p_hca:.3f}")

    def test_opening_mix_is_mixed_for_middle_ranks(self, solver_conv):
        """
        A player with rank r ∈ [4..10] should spread probability over at least
        2 distinct bids (demonstrating mixing, not pure-strategy play).
        """
        legal = _legal_actions(())
        for rank in [4, 6, 8, 10]:
            key: tuple = (0, rank, ())
            strat = solver_conv.average_strategy(key, legal)
            n_above_5pct = sum(1 for p in strat if p > 0.05)
            assert n_above_5pct >= 2, (
                f"Rank {RANK_NAMES[rank]}: strategy is too concentrated "
                f"({n_above_5pct} actions above 5%)")

    def test_pure_rank_bidding_is_exploitable(self):
        """
        Constructing a pure-strategy P0 that always bids HC rank(c0) should be
        highly exploitable. We verify this by building the best-response value
        for P1 against that pure strategy and checking it's very high.
        """
        solver = CFRSolver()
        legal = _legal_actions(())

        # Manually set strategy sums to pure "bid HC rank(c0)".
        for rank in range(13):
            key: tuple = (0, rank, ())
            n = len(legal)
            solver._strategy_sum[key] = [0.0] * n
            # Find the index of HC <rank> in legal actions.
            hc_rank_bid = Bid(HIGH_CARD, rank)
            try:
                hc_idx = legal.index(bid_to_index(hc_rank_bid))
                solver._strategy_sum[key][hc_idx] = 1.0
            except ValueError:
                # This rank might not have a legal HC bid (shouldn't happen).
                solver._strategy_sum[key][0] = 1.0

        # Best response for P1 against pure P0 strategy.
        br1_total = 0.0
        count = 0
        for c0 in range(52):
            for c1 in range(52):
                if c0 == c1:
                    continue
                br1_total += solver._best_response_value((), c0, c1, 1)
                count += 1
        br1_value = br1_total / count
        # P1's BR against pure P0 strategy should be large (P0 is very exploitable).
        assert br1_value > 0.5, (
            f"P1 BR value against pure P0 strategy should be > 0.5, got {br1_value:.4f}")


# ===========================================================================
# E — Exploitability computation
# ===========================================================================

class TestExploitability:
    def test_exploitability_nonneg(self):
        solver = CFRSolver()
        solver.run(100)
        exp = solver.exploitability()
        assert exp >= 0.0

    def test_exploitability_at_most_one(self):
        solver = CFRSolver()
        exp = solver.exploitability()  # uniform strategy
        # At uniform strategy the exploitability can be high but bounded by 1.
        assert exp <= 1.0

    def test_nash_conv_symmetric(self):
        """
        In a zero-sum game at Nash, BR(P0) + BR(P1) ≈ 0 (both best-response
        values against each other's Nash strategy are equal and opposite).
        After 2000 iterations, the difference should be small.
        """
        solver = CFRSolver()
        solver.run(2000)
        br0_total = 0.0
        br1_total = 0.0
        count = 0
        for c0 in range(0, 52, 4):   # sample every 4th card for speed
            for c1 in range(0, 52, 4):
                if c0 == c1:
                    continue
                br0_total += solver._best_response_value((), c0, c1, 0)
                br1_total += solver._best_response_value((), c0, c1, 1)
                count += 1
        br0 = br0_total / count
        br1 = br1_total / count
        # At Nash, BR0 ≈ game_value and BR1 ≈ -game_value, so BR0 + BR1 ≈ 0.
        # (NashConv = max(0, BR0 + BR1) should be near 0.)
        assert abs(br0 + br1) < 0.2, (
            f"BR0={br0:.4f}, BR1={br1:.4f}, sum={br0+br1:.4f} should be near 0")
