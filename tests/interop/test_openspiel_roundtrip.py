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
from game.bids import CALL_ACTION, NUM_BIDS  # noqa: E402
from game.engine import new_match  # noqa: E402


def _drive_full_roundtrip(seed: int) -> None:
    rng = random.Random(seed)

    # Project engine: single round, hand_size 5, 2 players, exact rules.
    match = new_match(num_players=2, seed=seed, exact_rules=True, high_hand=False)
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
    assert g.num_distinct_actions() == NUM_BIDS + 1
    # Sanity: CALL action index is the last one.
    s = g.new_initial_state()
    # deal 10 cards
    for c in range(10):
        s.apply_action(c)
    assert not s.is_chance_node()
    s.apply_action(0)  # bid HC 2
    assert CALL_ACTION in s.legal_actions()
