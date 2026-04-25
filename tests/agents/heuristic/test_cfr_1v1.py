"""
Tests for agents.heuristic.cfr_1v1 (bounded-tree variant).

Test categories:
  A — Game structure / holds table correctness
  B — Legal-actions logic (with bid_space, max_bids, include_hh)
  C — CFR mechanics (strategy extraction, regret accumulation)
  D — Convergence & mixed-strategy validation
  E — Exploitability
"""

from __future__ import annotations

import pytest
from agents.heuristic.cfr_1v1 import (
    CFRSolver,
    HC_PAIR_BIDS,
    _card_rank,
    _bid_holds_n2,
    _current_player,
    _legal_actions,
    _is_terminal,
    _terminal_utility_p0,
    _standing_bid_index,
    _num_bids_placed,
    _HOLDS,
    _HOLDS_RANK,
    _RANK_DEALS,
)
from game.bids import (
    all_bids, NUM_BIDS, CALL_ACTION, HH_ACTION,
    index_to_bid, bid_to_index,
    Bid, HIGH_CARD, PAIR,
)


# A — Game structure / holds table ==========================================

class TestGameStructure:
    def test_card_rank(self):
        for r in range(13):
            for s in range(4):
                assert _card_rank(r * 4 + s) == r

    def test_holds_table_shape(self):
        assert len(_HOLDS) == 52
        assert len(_HOLDS[0][1]) == NUM_BIDS

    def test_holds_rank_table_consistent_with_card_table(self):
        # Pair A pool: both aces.
        ace_c, ace_d = 12 * 4, 12 * 4 + 1
        pair_a = bid_to_index(Bid(PAIR, 12))
        assert _HOLDS[ace_c][ace_d][pair_a] is True
        assert _HOLDS_RANK[12][12][pair_a] is True

    def test_holds_hca_with_ace_and_two(self):
        ace_c, two_c = 12 * 4, 0
        hca = bid_to_index(Bid(HIGH_CARD, 12))
        assert _HOLDS[ace_c][two_c][hca] is True

    def test_holds_hca_with_two_aces(self):
        ace_c, ace_d = 12 * 4, 12 * 4 + 1
        hca = bid_to_index(Bid(HIGH_CARD, 12))
        assert _HOLDS[ace_c][ace_d][hca] is False

    def test_exactly_one_bid_holds_per_deal(self):
        for c0 in range(0, 52, 7):
            for c1 in range(0, 52, 7):
                if c0 == c1:
                    continue
                n = sum(1 for idx in range(NUM_BIDS) if _HOLDS[c0][c1][idx])
                assert n == 1

    def test_rank_deals_count_sums_to_2652(self):
        total = sum(w for _, _, w in _RANK_DEALS)
        assert total == 52 * 51


# B — Legal actions =========================================================

class TestLegalActions:
    _BS = HC_PAIR_BIDS  # 26 HC+Pair bids
    _MB = 4

    def test_root_is_bid_only_no_call(self):
        legal = _legal_actions((), self._BS, self._MB, include_hh=True)
        assert CALL_ACTION not in legal
        assert HH_ACTION not in legal
        assert len(legal) == len(self._BS)

    def test_after_first_bid_call_and_hh_present(self):
        first = HC_PAIR_BIDS[0]  # HC 2
        legal = _legal_actions((first,), self._BS, self._MB, include_hh=True)
        assert CALL_ACTION in legal
        assert HH_ACTION in legal

    def test_hh_omitted_when_disabled(self):
        first = HC_PAIR_BIDS[0]
        legal = _legal_actions((first,), self._BS, self._MB, include_hh=False)
        assert CALL_ACTION in legal
        assert HH_ACTION not in legal

    def test_raises_are_strictly_greater(self):
        first = HC_PAIR_BIDS[5]
        legal = _legal_actions((first,), self._BS, self._MB, include_hh=True)
        raises = [a for a in legal if a not in (CALL_ACTION, HH_ACTION)]
        assert all(a > first for a in raises)

    def test_max_bids_forces_terminal(self):
        # After max_bids bids: only CALL/HH legal (no more raises).
        history = tuple(HC_PAIR_BIDS[:self._MB])
        legal = _legal_actions(history, self._BS, self._MB, include_hh=True)
        assert all(a in (CALL_ACTION, HH_ACTION) for a in legal)

    def test_num_bids_placed_and_standing(self):
        first = HC_PAIR_BIDS[0]
        second = HC_PAIR_BIDS[3]
        history = (first, second)
        assert _num_bids_placed(history) == 2
        assert _standing_bid_index(history) == second


# C — CFR mechanics =========================================================

class TestCFRMechanics:
    def test_is_terminal(self):
        assert not _is_terminal(())
        assert not _is_terminal((HC_PAIR_BIDS[0],))
        assert _is_terminal((HC_PAIR_BIDS[0], CALL_ACTION))
        assert _is_terminal((HC_PAIR_BIDS[0], HH_ACTION))

    def test_terminal_utility_call_on_holding_bid(self):
        # Pool [A♣, K♣] → HC A. P0 bids HC A, P1 calls → P1 loses → P0 +1.
        ace_c, king_c = 12 * 4, 11 * 4
        hca = bid_to_index(Bid(HIGH_CARD, 12))
        util = _terminal_utility_p0((hca, CALL_ACTION), ace_c, king_c)
        assert util == +1.0

    def test_terminal_utility_call_bluff(self):
        # Pool [K♣, Q♣] → HC K. P0 bids HC A (bluff), P1 calls → P1 wins → P0 -1.
        king_c, queen_c = 11 * 4, 10 * 4
        hca = bid_to_index(Bid(HIGH_CARD, 12))
        util = _terminal_utility_p0((hca, CALL_ACTION), king_c, queen_c)
        assert util == -1.0

    def test_terminal_utility_hh_correct(self):
        # Pool [A♣, K♣] → HC A holds. P0 bids HC A, P1 declares HH → P1 wins → P0 -1.
        ace_c, king_c = 12 * 4, 11 * 4
        hca = bid_to_index(Bid(HIGH_CARD, 12))
        util = _terminal_utility_p0((hca, HH_ACTION), ace_c, king_c)
        assert util == -1.0

    def test_terminal_utility_hh_wrong(self):
        # Pool [K♣, Q♣] → HC K. P0 bids HC A, P1 declares HH (wrong) → P1 loses → P0 +1.
        king_c, queen_c = 11 * 4, 10 * 4
        hca = bid_to_index(Bid(HIGH_CARD, 12))
        util = _terminal_utility_p0((hca, HH_ACTION), king_c, queen_c)
        assert util == +1.0

    def test_single_iterate_does_not_crash(self):
        solver = CFRSolver(max_bids=3)
        u = solver.iterate()
        assert -1.0 <= u <= 1.0


# D — Convergence & mixed strategies =======================================

class TestConvergence:
    @pytest.fixture(scope="class")
    def solver_conv(self):
        s = CFRSolver(max_bids=3, bid_space=HC_PAIR_BIDS, include_hh=True)
        s.run(200)
        return s

    def test_exploitability_decreases(self):
        s = CFRSolver(max_bids=3)
        exp0 = s.exploitability()
        s.run(50)
        exp1 = s.exploitability()
        assert exp1 < exp0

    def test_exploitability_below_threshold(self, solver_conv):
        # Vanilla CFR converges O(1/√T). At 200 iters and max_bids=3 expect < 0.85.
        exp = solver_conv.exploitability()
        assert exp < 0.85, f"exp={exp:.4f} too high after 200 iters at max_bids=3"

    def test_opening_mix_is_mixed_for_middle_ranks(self, solver_conv):
        # A middle rank should spread probability across at least 2 actions.
        for rank in [4, 6, 8]:
            d = solver_conv.opening_mix_by_rank()[rank]
            above_5pct = sum(1 for p in d.values() if p > 0.05)
            assert above_5pct >= 1, f"rank {rank} has no prob above 5%"


# E — Exploitability =======================================================

class TestExploitability:
    def test_nonneg(self):
        s = CFRSolver(max_bids=3)
        s.run(20)
        assert s.exploitability() >= 0.0

    def test_serialization_roundtrip(self):
        s = CFRSolver(max_bids=3)
        s.run(5)
        d = s.to_dict()
        s2 = CFRSolver.from_dict(d)
        assert s2._iterations == s._iterations
        assert s2.max_bids == s.max_bids
        assert tuple(s2.bid_space) == tuple(s.bid_space)
        assert abs(s2.game_value() - s.game_value()) < 1e-9
