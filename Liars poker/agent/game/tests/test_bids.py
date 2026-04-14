"""
Unit tests for agent.game.bids — no external deps. Runnable as:

    cd "papers/Liars poker/"
    python -m agent.game.tests.test_bids
"""

from agent.game.bids import (
    Bid, NUM_BIDS, CALL_ACTION, NUM_ACTIONS,
    enumerate_bids, bid_to_index, index_to_bid, normalize_hand_type,
    HIGH_CARD, PAIR, TWO_PAIR, THREE_OF_A_KIND, STRAIGHT, FLUSH,
    FULL_HOUSE, FOUR_OF_A_KIND, STRAIGHT_FLUSH, ROYAL_FLUSH,
)


def test_bid_count():
    # HC 13 + Pair 13 + 2P 12 + 3K 13 + St 10 + Fl 13 + FH 13 + 4K 13 + SF 10 = 110
    expected = 13 + 13 + 12 + 13 + 10 + 13 + 13 + 13 + 10
    assert NUM_BIDS == expected, f"NUM_BIDS={NUM_BIDS} expected={expected}"
    assert NUM_ACTIONS == NUM_BIDS + 1
    assert CALL_ACTION == NUM_BIDS


def test_bid_ordering_is_total_and_monotonic():
    bids = enumerate_bids()
    # Strictly ascending under the natural tuple comparison
    for i in range(len(bids) - 1):
        a, b = bids[i], bids[i + 1]
        assert (a.hand_type, a.primary_rank) < (b.hand_type, b.primary_rank), \
            f"non-monotonic at {i}: {a} !< {b}"


def test_no_royal_flush_in_bid_space():
    for b in enumerate_bids():
        assert b.hand_type != ROYAL_FLUSH, f"Royal Flush leaked into bid space: {b}"


def test_strongest_bid_is_sf_ace():
    bids = enumerate_bids()
    top = bids[-1]
    assert top.hand_type == STRAIGHT_FLUSH
    assert top.primary_rank == 12  # Ace


def test_index_roundtrip():
    for i in range(NUM_BIDS):
        b = index_to_bid(i)
        assert bid_to_index(b) == i


def test_two_pair_lower_bound():
    # No "Two Pair 2" — the higher pair must be strictly above some lower pair
    two_pair_ranks = sorted(b.primary_rank for b in enumerate_bids() if b.hand_type == TWO_PAIR)
    assert two_pair_ranks == list(range(1, 13))


def test_straight_range():
    st_ranks = sorted(b.primary_rank for b in enumerate_bids() if b.hand_type == STRAIGHT)
    assert st_ranks == list(range(3, 13))  # 5-high (idx 3) .. A-high (idx 12)


def test_straight_flush_includes_ace_high():
    sf_ranks = sorted(b.primary_rank for b in enumerate_bids() if b.hand_type == STRAIGHT_FLUSH)
    assert sf_ranks == list(range(3, 13)), f"SF range wrong: {sf_ranks}"


def test_normalize_rf_maps_to_sf_ace():
    assert normalize_hand_type(ROYAL_FLUSH, 12) == (STRAIGHT_FLUSH, 12)
    # Non-RF inputs pass through
    assert normalize_hand_type(STRAIGHT_FLUSH, 7) == (STRAIGHT_FLUSH, 7)
    assert normalize_hand_type(PAIR, 0) == (PAIR, 0)


def test_bid_string_formatting():
    # Sanity check on printing
    assert str(Bid(PAIR, 0)) == "Pair 2"
    assert str(Bid(STRAIGHT_FLUSH, 12)) == "Straight Flush A"


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    run_all()
