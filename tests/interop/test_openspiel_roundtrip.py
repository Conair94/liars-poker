"""Round-trip parity tests: OpenSpiel adapter ⇄ project engine.

Each random rollout is replayed simultaneously through both engines.  At every
non-chance step we assert that:
  - the OpenSpiel state and the project engine agree on whose turn it is,
  - the sets of legal action indices are identical,
  - terminal flag and final returns match (project win/lose semantics).

The Kuhn-sized variant is exhaustively comparable to a hand-rolled reference
implementation; the full 52-card adapter is checked for parity against
``game.engine.MatchState`` configured as a single-round, exact-rules,
hand-size-5 game.
"""

from __future__ import annotations

import os
import random
import sys

# Path setup so this test file works under `pytest` from repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pyspiel  # noqa: E402

import interop  # noqa: E402,F401  (registers games)
from game.bids import CALL_ACTION, HH_ACTION, NUM_BIDS  # noqa: E402
from game.engine import new_match  # noqa: E402


def _drive_full_roundtrip(seed: int, high_hand: bool = False) -> None:
    rng = random.Random(seed)

    # Project engine: single round, hand_size 5, 2 players, exact rules.
    match = new_match(num_players=2, seed=seed, exact_rules=True, high_hand=high_hand)
    # Force everyone to start at hand_size=5 so this is single-round.
    match.hand_sizes = [5, 5]
    match.start_next_round()

    # OpenSpiel state.
    g = pyspiel.load_game("python_liars_poker_exact", {"num_players": 2, "hand_size": 5})
    s = g.new_initial_state()

    # Mirror the engine's deal into the OpenSpiel state via chance actions.
    rs = match.round_state
    for p in range(2):
        for c in rs.hands[p]:
            assert s.is_chance_node()
            s.apply_action(c)
    assert not s.is_chance_node()

    # Now drive bidding randomly until terminal in the OpenSpiel state, mirroring
    # the same actions in the engine.
    while not s.is_terminal():
        legal_os = sorted(s.legal_actions())
        legal_eng = sorted(match.legal_actions())
        # Adapter always exposes HH (canonical going forward). When the engine
        # has high_hand=False, drop HH from the OpenSpiel side for parity.
        if not high_hand:
            legal_os = [a for a in legal_os if a != HH_ACTION]
        assert legal_os == legal_eng, (
            f"legal mismatch:\n  os={legal_os[:10]}...{legal_os[-3:]}\n"
            f"  eng={legal_eng[:10]}...{legal_eng[-3:]}"
        )
        cur_os = s.current_player()
        cur_eng = match.current_player()
        assert cur_os == cur_eng, f"player mismatch {cur_os} vs {cur_eng}"
        a = rng.choice(legal_os)
        s.apply_action(a)
        match.apply_action(a)

    # Match's round resolved — read result from history.
    result = match.round_history[-1]
    eng_returns = [0.0, 0.0]
    eng_returns[result.winner_seat] = 1.0
    eng_returns[result.loser_seat] = -1.0
    assert s.returns() == eng_returns, (
        f"return mismatch: os={s.returns()} eng={eng_returns}"
    )


def test_full_adapter_roundtrip_1000_games():
    """Exit-criterion: round-trip game-play test across 1000 random games."""
    for seed in range(1000):
        _drive_full_roundtrip(seed)


def test_full_adapter_roundtrip_1000_games_high_hand():
    """Same parity test with HH enabled — exercises the HH action path."""
    for seed in range(1000):
        _drive_full_roundtrip(seed, high_hand=True)


def _hh_test_pool():
    """Returns (P0 cards, P1 cards) where pool best = FOUR_OF_A_KIND 2s.

    Encoding: card = rank*4 + suit, suits 0=c,1=d,2=h,3=s.
    P0: 2c,2d,2h,2s,3c -> ranks 0,0,0,0,1
    P1: 4d,7h,9c,Js,Kd -> ranks 2,5,7,9,11 (no pairs/straights/flushes with P0)
    Pool best is FOUR_OF_A_KIND, primary=0.
    """
    p0 = [0, 1, 2, 3, 4]
    p1 = [9, 22, 28, 39, 45]
    return p0, p1


def test_full_adapter_hh_resolution_incorrect_declaration():
    """HH declaration that doesn't match pool best → bidder wins."""
    g = pyspiel.load_game("python_liars_poker_exact", {"num_players": 2, "hand_size": 5})
    s = g.new_initial_state()
    p0, p1 = _hh_test_pool()
    for c in p0 + p1:
        s.apply_action(c)
    s.apply_action(0)  # P0 bids HC 2 (does not match pool best = FK 2s)
    legal = s.legal_actions()
    assert HH_ACTION in legal
    s.apply_action(HH_ACTION)  # P1 declares HH — incorrect
    assert s.is_terminal()
    assert s.returns() == [1.0, -1.0]


def test_full_adapter_hh_resolution_correct_when_bid_matches_best():
    """HH declaration that matches pool best exactly → declarer wins."""
    from game.bids import FOUR_OF_A_KIND, Bid, bid_to_index
    g = pyspiel.load_game("python_liars_poker_exact", {"num_players": 2, "hand_size": 5})
    s = g.new_initial_state()
    p0, p1 = _hh_test_pool()
    for c in p0 + p1:
        s.apply_action(c)
    fk2 = bid_to_index(Bid(FOUR_OF_A_KIND, 0))
    s.apply_action(fk2)
    legal = s.legal_actions()
    assert HH_ACTION in legal
    s.apply_action(HH_ACTION)
    assert s.is_terminal()
    assert s.returns() == [-1.0, 1.0]


def test_kuhn_load_and_actions():
    g = pyspiel.load_game("python_liars_poker_kuhn")
    assert g.num_distinct_actions() == 4
    assert g.num_players() == 2
    s = g.new_initial_state()
    assert s.is_chance_node()
    outs = s.chance_outcomes()
    assert {o for o, _ in outs} == {0, 1, 2}
    s.apply_action(0)  # P0 = Q
    s.apply_action(2)  # P1 = A
    # Opening bid: only HC bids (no CALL legal yet).
    assert sorted(s.legal_actions()) == [0, 1, 2]
    s.apply_action(2)  # bid HC A — true!
    # P1 can only call (no stronger bid available).
    assert sorted(s.legal_actions()) == [3]
    s.apply_action(3)  # call
    assert s.is_terminal()
    # Bid HC A holds (pool best = HC A, exact rules) → bidder (P0) wins.
    assert s.returns() == [1.0, -1.0]


def test_kuhn_lie_resolves_correctly():
    """If bidder lies, caller wins."""
    g = pyspiel.load_game("python_liars_poker_kuhn")
    s = g.new_initial_state()
    s.apply_action(0)  # P0 = Q
    s.apply_action(1)  # P1 = K  (pool best = HC K)
    s.apply_action(2)  # P0 bids HC A (lie)
    s.apply_action(3)  # P1 calls
    assert s.is_terminal()
    assert s.returns() == [-1.0, 1.0]


def test_full_adapter_action_space_size():
    g = pyspiel.load_game("python_liars_poker_exact")
    assert g.num_distinct_actions() == NUM_BIDS + 2
    # Sanity: CALL and HH are legal once a bid stands.
    s = g.new_initial_state()
    for c in range(10):
        s.apply_action(c)
    assert not s.is_chance_node()
    s.apply_action(0)  # bid HC 2
    legal = s.legal_actions()
    assert CALL_ACTION in legal
    assert HH_ACTION in legal
