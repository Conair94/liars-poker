"""
Unit tests for agent.game.engine. Pure Python, no pytest dep. Run via:

    cd "papers/Liars poker/"
    python -m agent.game.tests.test_engine
"""

from agent.game import (
    new_match, MatchState, Bid, CALL_ACTION, NUM_BIDS,
    bid_to_index, index_to_bid, enumerate_bids,
)
from agent.game.bids import (
    HIGH_CARD, PAIR, STRAIGHT_FLUSH, normalize_hand_type, ROYAL_FLUSH,
)
from poker_math_exact import _evaluate_ranked


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _force_round(state: MatchState, hands):
    """
    Overwrite the current round's dealt hands with hands specified by the
    test. hands: list of list of card indices per seat (use [] for inactive
    seats). Preserves first-bidder and other round metadata.
    """
    assert state.round_state is not None
    state.round_state.hands = [list(h) for h in hands]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_new_match_initial_state():
    s = new_match(3, seed=1)
    assert s.num_players == 3
    assert s.hand_sizes == [1, 1, 1]
    assert s.active == [True, True, True]
    assert not s.terminal
    assert s.round_state is None
    assert s.first_bidder_next == 0


def test_player_count_bounds():
    for n in (2, 3, 4, 5):
        new_match(n, seed=0)
    for bad in (0, 1, 6, 10):
        try:
            new_match(bad)
        except ValueError:
            continue
        raise AssertionError(f"new_match({bad}) should have raised")


def test_start_round_deals_correct_sizes():
    s = new_match(4, seed=7)
    s.hand_sizes = [1, 2, 3, 4]
    s.start_next_round()
    rs = s.round_state
    assert rs is not None
    assert len(rs.hands[0]) == 1
    assert len(rs.hands[1]) == 2
    assert len(rs.hands[2]) == 3
    assert len(rs.hands[3]) == 4
    # All cards unique across all hands
    all_cards = [c for h in rs.hands for c in h]
    assert len(set(all_cards)) == len(all_cards)
    # Current player = first_bidder_next
    assert rs.current_player == s.first_bidder_next


def test_legal_actions_first_bid_cannot_call():
    s = new_match(2, seed=0)
    s.start_next_round()
    legal = s.legal_actions()
    assert CALL_ACTION not in legal
    assert len(legal) == NUM_BIDS


def test_legal_actions_after_bid():
    s = new_match(2, seed=0)
    s.start_next_round()
    # Bid the weakest thing (HC 2)
    first_bid_idx = 0
    s.apply_action(first_bid_idx)
    legal = s.legal_actions()
    assert CALL_ACTION in legal
    # Can raise to any strictly stronger bid
    assert first_bid_idx not in legal
    assert all(a > first_bid_idx or a == CALL_ACTION for a in legal)


def test_bid_then_call_round_resolves():
    s = new_match(2, seed=0)
    s.start_next_round()
    # Force hands: seat 0 has 2 of clubs, seat 1 has 3 of clubs.
    # So pool is just {2c, 3c}; best 5-card hand from 2 cards is a High Card 3.
    _force_round(s, [[0], [4]])  # card 0 = rank 0 (2) suit 0 (C); card 4 = rank 1 (3) suit 0 (C)

    # Seat 0 (first bidder) bids Pair 2 (a lie — pool is just high card).
    pair_2_idx = bid_to_index(Bid(PAIR, 0))
    result = s.apply_action(pair_2_idx)
    assert result is None  # not terminal yet

    # Seat 1 calls the bluff.
    result = s.apply_action(CALL_ACTION)
    assert result is not None
    assert result.call_succeeded is True   # bid was a lie
    assert result.loser_seat == 0          # bidder loses
    assert result.winner_seat == 1
    # Loser's hand_size should have incremented
    assert s.hand_sizes[0] == 2
    assert s.hand_sizes[1] == 1
    # Round cleared; between rounds now
    assert s.round_state is None
    # Loser leads next round
    assert s.first_bidder_next == 0


def test_bid_true_caller_loses():
    s = new_match(2, seed=0)
    s.start_next_round()
    # Both seats dealt pocket aces (all four aces in play).
    # Pool = {A♣, A♦, A♥, A♠} → four of a kind aces.
    s.hand_sizes = [2, 2]
    s.round_state.hands = [[48, 49], [50, 51]]  # ranks 12, suits 0..3

    # First bidder bids Pair A (true: pool even has four of a kind).
    bid = Bid(PAIR, 12)
    s.apply_action(bid_to_index(bid))
    # Opponent calls → caller loses
    result = s.apply_action(CALL_ACTION)
    assert result is not None
    assert result.call_succeeded is False
    assert result.loser_seat == 1
    assert result.winner_seat == 0
    assert s.hand_sizes == [2, 3]


def test_elimination_at_five_cards():
    s = new_match(2, seed=0)
    # Seat 0 is already at 5 cards. If they lose this round they must be eliminated.
    s.hand_sizes = [5, 1]
    s.start_next_round()
    # Force a pool that supports no bids. Six cards, all rank-disjoint, off-suit.
    # Hand sizes are 5 + 1 = 6. Give seat 0 five distinct low cards off-suit,
    # seat 1 one disjoint card. Best hand will be a high card.
    s.round_state.hands = [[0, 4, 8, 13, 17], [22]]
    # ^ 2c 3c 4c 2d 3d 4d  -> wait that's duplicates. Rebuild:
    # rank*4+suit:
    #   0  = 2C   (rank 0, suit 0)
    #   5  = 3D   (rank 1, suit 1)
    #   10 = 4H   (rank 2, suit 2)
    #   15 = 5S   (rank 3, suit 3)
    #   16 = 6C   (rank 4, suit 0)
    #   21 = 7D   (rank 5, suit 1)
    s.round_state.hands = [[0, 5, 10, 15, 16], [21]]

    # Seat 0 makes an outrageous bid: Straight Flush Ace.
    s.apply_action(bid_to_index(Bid(STRAIGHT_FLUSH, 12)))
    # Seat 1 calls — pool definitely doesn't contain a straight flush.
    result = s.apply_action(CALL_ACTION)
    assert result is not None
    assert result.loser_seat == 0
    # Seat 0 eliminated
    assert s.active == [False, True]
    assert s.hand_sizes[0] == 0
    # Match should now be terminal with seat 1 winning
    assert s.terminal
    assert s.winner == 1
    r = s.returns()
    assert r[0] == -1.0
    assert r[1] == 1.0


def test_clone_is_independent():
    s = new_match(2, seed=5)
    s.start_next_round()
    s.apply_action(0)  # weakest bid
    c = s.clone()
    # Advance c further; s should be unaffected
    c.apply_action(CALL_ACTION)
    assert c.round_state is None
    assert s.round_state is not None


def test_info_state_hides_opponents_private_cards():
    s = new_match(2, seed=42)
    s.hand_sizes = [3, 3]
    s.start_next_round()
    # Seat 0 sees their own hand, NOT seat 1's.
    info0 = s.info_state(0)
    info1 = s.info_state(1)
    assert info0["own_hand"] == sorted(s.round_state.hands[0])
    assert info1["own_hand"] == sorted(s.round_state.hands[1])
    # Opponent hand is never leaked.
    assert s.round_state.hands[1] != info0["own_hand"]
    # Public features agree across seats
    assert info0["hand_sizes"] == info1["hand_sizes"]
    assert info0["active"] == info1["active"]


def test_royal_flush_evaluator_output_compares_equal_to_sf_ace_bid():
    """
    If the pool contains A-K-Q-J-10 of one suit, the evaluator returns
    ROYAL_FLUSH=9, primary=12. The engine must normalize this to
    (STRAIGHT_FLUSH, 12) so a bid of "Straight Flush A" holds.
    """
    # Royal flush of spades: cards with suit=3 and ranks {12, 11, 10, 9, 8}
    rf_cards = [12 * 4 + 3, 11 * 4 + 3, 10 * 4 + 3, 9 * 4 + 3, 8 * 4 + 3]
    raw = _evaluate_ranked(rf_cards)
    assert raw == (ROYAL_FLUSH, 12)  # evaluator still returns RF
    assert normalize_hand_type(*raw) == (STRAIGHT_FLUSH, 12)

    s = new_match(2, seed=0)
    s.hand_sizes = [3, 2]
    s.start_next_round()
    s.round_state.hands = [rf_cards[:3], rf_cards[3:]]

    # Seat 0 bids Straight Flush A (truthful — pool contains it)
    s.apply_action(bid_to_index(Bid(STRAIGHT_FLUSH, 12)))
    # Seat 1 calls. Pool actually is a Royal Flush. Bid holds → caller loses.
    result = s.apply_action(CALL_ACTION)
    assert result is not None
    assert result.call_succeeded is False, "RF pool must make SF+A bid hold"
    assert result.loser_seat == 1


def test_three_player_round_rotation():
    s = new_match(3, seed=9)
    s.hand_sizes = [2, 2, 2]
    s.start_next_round()
    # first player bids
    s.apply_action(0)
    assert s.round_state.current_player == 1
    s.apply_action(5)  # some stronger bid
    assert s.round_state.current_player == 2
    # Seat 2 calls
    result = s.apply_action(CALL_ACTION)
    assert result is not None
    assert result.caller_seat == 2


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    run_all()
