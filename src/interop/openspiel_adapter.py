"""OpenSpiel adapter for card-based Liar's Poker.

Two games are registered:

  ``python_liars_poker_kuhn`` — small-game oracle.
    * 2 players, 1 private card each, 3-card deck (ranks Q=0, K=1, A=2).
    * Pool size = 2 ⇒ best hand is always High Card of max rank.
    * Bid space = {HC Q, HC K, HC A} (3 bids) + CALL (1 action) = 4 actions.
    * Exact-rules: call resolves true iff pool's best hand exactly equals bid.
    * Single round; terminal on CALL. ±1 reward to bidder / caller.
    * Tractable for tabular CFR+ → reference policy + exploitability oracle.

  ``python_liars_poker_exact`` — single-round adapter for the canonical game.
    * Configurable ``num_players`` (default 2) and ``hand_size`` (default 5).
    * Full 52-card deck, exact-rules call resolution.
    * Action space = NUM_ACTIONS = 112 (110 bids + CALL + HH).
    * Used for round-trip parity tests with the project engine and for
      OpenSpiel ecosystem interop. NOT tractable for tabular exploitability.

Encoding decisions are pinned in ADR-005.
"""

from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np
import pyspiel

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from game.bids import (  # noqa: E402
    CALL_ACTION,
    HIGH_CARD,
    NUM_BIDS,
    Bid,
    bid_to_index,
    index_to_bid,
    normalize_hand_type,
)

_PROBS = os.path.abspath(os.path.join(_SRC, "training", "probs"))
if _PROBS not in sys.path:
    sys.path.insert(0, _PROBS)
from poker_math_exact import _evaluate_ranked  # noqa: E402

# ---------------------------------------------------------------------------
# Small-game (Kuhn-like) variant
# ---------------------------------------------------------------------------

# Deck ranks (using the project's 0..12 = 2..A encoding). We pick high cards so
# pool-best is unambiguous and bids are HC-only.
_KUHN_DECK_RANKS = (10, 11, 12)  # Q, K, A
_KUHN_NUM_PLAYERS = 2

# Action layout: 0..len(deck)-1 = HC bids in ascending rank order, last = CALL.
_KUHN_NUM_BIDS = len(_KUHN_DECK_RANKS)
_KUHN_CALL = _KUHN_NUM_BIDS                   # = 3
_KUHN_NUM_ACTIONS = _KUHN_NUM_BIDS + 1        # = 4


def _kuhn_action_to_bid_rank(action: int) -> int:
    """Map a bid-action index to the underlying rank (0..12 in project encoding)."""
    if not (0 <= action < _KUHN_NUM_BIDS):
        raise IndexError(action)
    return _KUHN_DECK_RANKS[action]


_KUHN_GAME_TYPE = pyspiel.GameType(
    short_name="python_liars_poker_kuhn",
    long_name="Python Liars Poker (Kuhn-sized oracle)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_KUHN_NUM_PLAYERS,
    min_num_players=_KUHN_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=False,
)
# max_game_length: deck-size bids + CALL = 4 in worst case (HC Q, HC K, HC A, CALL).
_KUHN_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_KUHN_NUM_ACTIONS,
    max_chance_outcomes=len(_KUHN_DECK_RANKS),
    num_players=_KUHN_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_KUHN_NUM_BIDS + 1,
)


class LiarsPokerKuhnGame(pyspiel.Game):
    """Kuhn-sized Liar's Poker for use as an exact-solve oracle."""

    def __init__(self, params=None):
        super().__init__(_KUHN_GAME_TYPE, _KUHN_GAME_INFO, params or dict())

    def new_initial_state(self):
        return LiarsPokerKuhnState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return LiarsPokerKuhnObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params,
        )


class LiarsPokerKuhnState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self.cards: list[int] = []           # rank held by player 0, then 1
        self.bets: list[int] = []            # action history after dealing
        self.current_bid_idx: int | None = None  # action index of standing bid
        self.last_bidder: int | None = None
        self._game_over = False
        self._returns = [0.0, 0.0]
        self._next_player = 0

    def current_player(self):
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        if len(self.cards) < _KUHN_NUM_PLAYERS:
            return pyspiel.PlayerId.CHANCE
        return self._next_player

    def _legal_actions(self, player):
        assert player >= 0
        if self.current_bid_idx is None:
            # Opening bid: any HC bid (no call legal).
            return list(range(_KUHN_NUM_BIDS))
        # Otherwise: any strictly-stronger bid, plus CALL.
        return list(range(self.current_bid_idx + 1, _KUHN_NUM_BIDS)) + [_KUHN_CALL]

    def chance_outcomes(self):
        assert self.is_chance_node()
        remaining = sorted(set(range(len(_KUHN_DECK_RANKS))) - set(self.cards))
        p = 1.0 / len(remaining)
        return [(o, p) for o in remaining]

    def _apply_action(self, action):
        if self.is_chance_node():
            self.cards.append(action)
            return
        self.bets.append(action)
        if action == _KUHN_CALL:
            self._resolve_call()
        else:
            self.current_bid_idx = action
            self.last_bidder = self._next_player
            self._next_player = 1 - self._next_player

    def _resolve_call(self):
        # Pool best = HC of max private rank.
        pool_ranks = [_KUHN_DECK_RANKS[c] for c in self.cards]
        pool_best_rank = max(pool_ranks)
        bid_rank = _KUHN_DECK_RANKS[self.current_bid_idx]
        bid_holds = (pool_best_rank == bid_rank)  # exact rules
        caller = self._next_player
        bidder = self.last_bidder
        if bid_holds:
            winner, loser = bidder, caller
        else:
            winner, loser = caller, bidder
        self._returns = [0.0, 0.0]
        self._returns[winner] = 1.0
        self._returns[loser] = -1.0
        self._game_over = True

    def _action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            rank = _KUHN_DECK_RANKS[action]
            return f"Deal:{_RANK_LETTER[rank]}"
        if action == _KUHN_CALL:
            return "Call"
        return f"HC {_RANK_LETTER[_KUHN_DECK_RANKS[action]]}"

    def is_terminal(self):
        return self._game_over

    def returns(self):
        return list(self._returns)

    def __str__(self):
        deal = "".join(_RANK_LETTER[_KUHN_DECK_RANKS[c]] for c in self.cards)
        history = []
        for a in self.bets:
            history.append("c" if a == _KUHN_CALL else _RANK_LETTER[_KUHN_DECK_RANKS[a]])
        return deal + ("|" if history else "") + "".join(history)


_RANK_LETTER = {
    0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8", 7: "9",
    8: "T", 9: "J", 10: "Q", 11: "K", 12: "A",
}


class LiarsPokerKuhnObserver:
    """Single-tensor observer: own-card one-hot + bid history per turn."""

    def __init__(self, iig_obs_type, params):
        if params:
            raise ValueError(f"observation parameters not supported; got {params}")
        # Pieces: player one-hot (2), private card one-hot (3), per-turn bid history.
        # History layout: max_game_length × num_distinct_actions one-hot.
        self._private = (
            iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER
        )
        self._public = iig_obs_type.public_info
        size = 2
        if self._private:
            size += _KUHN_NUM_BIDS
        if self._public:
            size += (_KUHN_NUM_BIDS + 1) * _KUHN_NUM_ACTIONS
        self.tensor = np.zeros(size, np.float32)
        self.dict: dict[str, np.ndarray] = {}
        idx = 0
        self.dict["player"] = self.tensor[idx:idx + 2].reshape((2,)); idx += 2
        if self._private:
            self.dict["private_card"] = self.tensor[idx:idx + _KUHN_NUM_BIDS]
            idx += _KUHN_NUM_BIDS
        if self._public:
            self.dict["history"] = self.tensor[
                idx:idx + (_KUHN_NUM_BIDS + 1) * _KUHN_NUM_ACTIONS
            ].reshape((_KUHN_NUM_BIDS + 1, _KUHN_NUM_ACTIONS))

    def set_from(self, state, player):
        self.tensor.fill(0)
        self.dict["player"][player] = 1
        if self._private and len(state.cards) > player:
            self.dict["private_card"][state.cards[player]] = 1
        if self._public:
            for turn, a in enumerate(state.bets[: _KUHN_NUM_BIDS + 1]):
                self.dict["history"][turn, a] = 1

    def string_from(self, state, player):
        parts = [f"p{player}"]
        if self._private and len(state.cards) > player:
            parts.append(f"card:{_RANK_LETTER[_KUHN_DECK_RANKS[state.cards[player]]]}")
        if self._public and state.bets:
            parts.append(
                "".join(
                    "c" if a == _KUHN_CALL else _RANK_LETTER[_KUHN_DECK_RANKS[a]]
                    for a in state.bets
                )
            )
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Full single-round adapter (52-card, exact rules, hand_size configurable)
# ---------------------------------------------------------------------------

_FULL_DECK = 52
_FULL_NUM_PLAYERS_DEFAULT = 2
_FULL_HAND_SIZE_DEFAULT = 5
_FULL_NUM_ACTIONS = NUM_BIDS + 2   # bids 0..109, CALL = 110, HH = 111.
_FULL_CALL = NUM_BIDS              # = 110
_FULL_HH = NUM_BIDS + 1            # = 111  (matches engine HH_ACTION)

_FULL_GAME_TYPE = pyspiel.GameType(
    short_name="python_liars_poker_exact",
    long_name="Python Liars Poker (single round, exact rules, 52-card)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=5,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    provides_factored_observation_string=False,
    parameter_specification={
        "num_players": 2,
        "hand_size": 5,
    },
)
# Worst case: every bid (NUM_BIDS) plus CALL.
_FULL_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_FULL_NUM_ACTIONS,
    max_chance_outcomes=_FULL_DECK,
    num_players=_FULL_NUM_PLAYERS_DEFAULT,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=NUM_BIDS + 1,
)


def _has_exact_hand(pool: list[int], bid: Bid) -> bool:
    """Return True if some 5-card subset of pool evaluates to EXACTLY bid."""
    if len(pool) < 5:
        raw_t, raw_p = _evaluate_ranked(pool)
        t, p = normalize_hand_type(raw_t, raw_p)
        return t == bid.hand_type and p == bid.primary_rank
    for combo in combinations(pool, 5):
        raw_t, raw_p = _evaluate_ranked(list(combo))
        t, p = normalize_hand_type(raw_t, raw_p)
        if t == bid.hand_type and p == bid.primary_rank:
            return True
    return False


class LiarsPokerExactGame(pyspiel.Game):
    """Single-round, exact-rules, 52-card Liar's Poker."""

    def __init__(self, params=None):
        params = params or {}
        self._num_players = int(params.get("num_players", _FULL_NUM_PLAYERS_DEFAULT))
        self._hand_size = int(params.get("hand_size", _FULL_HAND_SIZE_DEFAULT))
        if not (2 <= self._num_players <= 5):
            raise ValueError(f"num_players must be in [2,5]; got {self._num_players}")
        if not (1 <= self._hand_size <= 5):
            raise ValueError(f"hand_size must be in [1,5]; got {self._hand_size}")
        super().__init__(_FULL_GAME_TYPE, _FULL_GAME_INFO, params)

    @property
    def num_players_(self):
        return self._num_players

    @property
    def hand_size(self):
        return self._hand_size

    def new_initial_state(self):
        return LiarsPokerExactState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return _NullObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False)
        )


class LiarsPokerExactState(pyspiel.State):
    def __init__(self, game: LiarsPokerExactGame):
        super().__init__(game)
        self._np = game.num_players_
        self._hs = game.hand_size
        self._total_cards = self._np * self._hs
        self._dealt: list[int] = []          # order: P0's cards, then P1's, ...
        self._bets: list[int] = []
        self._current_bid: Bid | None = None
        self._last_bidder: int | None = None
        self._next_player = 0
        self._game_over = False
        self._returns = [0.0] * self._np

    def _hand(self, p: int) -> list[int]:
        s = p * self._hs
        return self._dealt[s:s + self._hs]

    def current_player(self):
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        if len(self._dealt) < self._total_cards:
            return pyspiel.PlayerId.CHANCE
        return self._next_player

    def _legal_actions(self, player):
        assert player >= 0
        if self._current_bid is None:
            return list(range(NUM_BIDS))
        cur_idx = bid_to_index(self._current_bid)
        return list(range(cur_idx + 1, NUM_BIDS)) + [_FULL_CALL, _FULL_HH]

    def chance_outcomes(self):
        assert self.is_chance_node()
        remaining = sorted(set(range(_FULL_DECK)) - set(self._dealt))
        p = 1.0 / len(remaining)
        return [(o, p) for o in remaining]

    def _apply_action(self, action):
        if self.is_chance_node():
            self._dealt.append(action)
            return
        self._bets.append(action)
        if action == _FULL_CALL:
            self._resolve_call()
        elif action == _FULL_HH:
            self._resolve_high_hand()
        else:
            self._current_bid = index_to_bid(action)
            self._last_bidder = self._next_player
            self._next_player = (self._next_player + 1) % self._np

    def _resolve_call(self):
        caller = self._next_player
        bidder = self._last_bidder
        pool: list[int] = list(self._dealt)  # all active hands at single round
        bid_holds = _has_exact_hand(pool, self._current_bid)
        winner = bidder if bid_holds else caller
        loser = caller if bid_holds else bidder
        self._returns = [0.0] * self._np
        self._returns[winner] = 1.0
        self._returns[loser] = -1.0
        self._game_over = True

    def _resolve_high_hand(self):
        # HH: declarer claims the pool's BEST hand exactly equals the standing
        # bid. Mirrors MatchState._resolve_high_hand projected to single-round
        # (no card-count penalty; ±1 zero-sum reward).
        declarer = self._next_player
        bidder = self._last_bidder
        pool: list[int] = list(self._dealt)
        raw_t, raw_p = _evaluate_ranked(pool)
        pool_type, pool_primary = normalize_hand_type(raw_t, raw_p)
        bid = self._current_bid
        correct = (pool_type == bid.hand_type and pool_primary == bid.primary_rank)
        winner = declarer if correct else bidder
        loser = bidder if correct else declarer
        self._returns = [0.0] * self._np
        self._returns[winner] = 1.0
        self._returns[loser] = -1.0
        self._game_over = True

    def _action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            return f"Deal:{action}"
        if action == _FULL_CALL:
            return "Call"
        if action == _FULL_HH:
            return "HH"
        b = index_to_bid(action)
        return str(b)

    def is_terminal(self):
        return self._game_over

    def returns(self):
        return list(self._returns)

    def __str__(self):
        return f"dealt={self._dealt} bets={self._bets}"


class _NullObserver:
    """Minimal observer (string only); tensor not implemented for full game."""

    def __init__(self, iig_obs_type):
        self._private = (
            iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER
        )
        self._public = iig_obs_type.public_info
        self.tensor = np.zeros(0, np.float32)
        self.dict: dict[str, np.ndarray] = {}

    def set_from(self, state, player):
        pass  # tensor unused

    def string_from(self, state, player):
        parts = [f"p{player}"]
        if self._private:
            parts.append(f"hand:{sorted(state._hand(player))}")
        if self._public:
            parts.append(f"bets:{state._bets}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

pyspiel.register_game(_KUHN_GAME_TYPE, LiarsPokerKuhnGame)
pyspiel.register_game(_FULL_GAME_TYPE, LiarsPokerExactGame)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def kuhn_pool_best_rank_action(state: LiarsPokerKuhnState) -> int:
    """Return the bid-action index whose rank equals the pool's true best rank."""
    pool_ranks = [_KUHN_DECK_RANKS[c] for c in state.cards]
    best = max(pool_ranks)
    for a, rank in enumerate(_KUHN_DECK_RANKS):
        if rank == best:
            return a
    raise RuntimeError("unreachable")


def kuhn_full_bid_index(action: int) -> int:
    """Map a Kuhn-game bid-action index to the project's full-game NUM_BIDS index."""
    rank = _kuhn_action_to_bid_rank(action)
    return bid_to_index(Bid(HIGH_CARD, rank))
