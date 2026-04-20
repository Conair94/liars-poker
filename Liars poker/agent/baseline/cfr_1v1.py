"""
CFR (Counterfactual Regret Minimization) solver for the 1v1 n=2 private-info
exact-rules Liar's Poker game.

Game summary
------------
- Each player is dealt exactly 1 card from a 52-card deck (pool size n=2).
- Exact-rules resolution: bid holds iff some 5-card subset of the 2-card pool
  evaluates to exactly (hand_type, primary_rank). For n=2, no 5-card subsets
  exist, so the evaluator falls back to evaluating all 2 cards directly.
- Player 0 bids first; players alternate; first legal action on any turn is to
  call (except before any bid exists — P0 must bid on turn 0).
- On a call: caller wins (+1) if the standing bid does NOT hold; loses (-1) if
  the bid holds exactly.

Why mixed strategies are required
----------------------------------
If P0 always bids "HC rank(c0)", P1 can infer P0's rank from the bid:
  - P1 rank > P0 rank → HC (P0 rank) never holds → P1 always calls (wins).
  - P1 rank < P0 rank → HC (P0 rank) holds     → P1 never calls (loses).
  - P1 rank == P0 rank → pool = Pair, not HC   → P1 always calls (wins).
P1 has perfect certainty; pure bidding eliminates all strategic value of P0's
private card. Nash equilibrium requires P0 to mix over bids so that P1 cannot
infer P0's rank from the opening bid.

Algorithm: Vanilla CFR
-----------------------
- Information sets (infostates): (player, card_rank, history_tuple)
  card_rank = rank of player's single private card (0..12, Aces=12).
- Full game tree enumeration: iterate over all 52×51=2652 ordered deals per
  outer CFR iteration. Each deal is a deterministic path through the tree;
  regrets are accumulated in proportion to the opponent's reach probability.
- Average strategy: maintained across all iterations via cumulative sum.
- Exploitability: computed by best-response traversal (exact, not sampling).

Usage
-----
    python -m agent.baseline.cfr_1v1 [--iters 50000] [--compare]

Results are cached to agent/data/cfr_1v1.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

_BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR    = os.path.abspath(os.path.join(_BASELINE_DIR, ".."))
_PAPER_DIR    = os.path.abspath(os.path.join(_AGENT_DIR,    ".."))

for _p in (_PAPER_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from poker_math_exact import _evaluate_ranked          # noqa: E402
from agent.game.bids import (                          # noqa: E402
    Bid, all_bids, NUM_BIDS, CALL_ACTION,
    bid_to_index, index_to_bid, normalize_hand_type,
    HIGH_CARD,
)

_DATA_DIR   = os.path.join(_AGENT_DIR, "data")
_CACHE_FILE = os.path.join(_DATA_DIR, "cfr_1v1.json")

# ---------------------------------------------------------------------------
# Card / pool helpers
# ---------------------------------------------------------------------------

def _card_rank(card: int) -> int:
    """Return rank 0..12 from card index 0..51."""
    return card // 4


def _bid_holds_n2(card0: int, card1: int, bid: Bid) -> bool:
    """True if the 2-card pool [card0, card1] evaluates to exactly bid."""
    raw_t, raw_p = _evaluate_ranked([card0, card1])
    t, p = normalize_hand_type(raw_t, raw_p)
    return t == bid.hand_type and p == bid.primary_rank


# Precompute bid-holds table for all ordered 2-card deals × all bids.
# Shape: _HOLDS[card0][card1][bid_idx] → bool (stored as int 0/1)
def _build_holds_table() -> List[List[List[bool]]]:
    bids = all_bids()
    table: List[List[List[bool]]] = [
        [
            [_bid_holds_n2(c0, c1, b) for b in bids]
            for c1 in range(52)
        ]
        for c0 in range(52)
    ]
    return table

_HOLDS = _build_holds_table()


# ---------------------------------------------------------------------------
# History encoding
# ---------------------------------------------------------------------------

# A history is a tuple of action indices (bid indices or CALL_ACTION).
# Turn 0: P0 must bid (CALL_ACTION illegal). Turn t≥1: either player may call.
# The game terminates when CALL_ACTION is played.

def _current_player(history: Tuple[int, ...]) -> int:
    return len(history) % 2


def _legal_actions(history: Tuple[int, ...]) -> List[int]:
    """Legal actions at the given history."""
    if len(history) == 0:
        # P0 must bid; no call allowed before any bid exists.
        return list(range(NUM_BIDS))
    standing_bid_idx = max(a for a in history if a != CALL_ACTION)
    # Can call or raise above the standing bid.
    actions = [CALL_ACTION] + list(range(standing_bid_idx + 1, NUM_BIDS))
    return actions


def _is_terminal(history: Tuple[int, ...]) -> bool:
    return len(history) > 0 and history[-1] == CALL_ACTION


def _terminal_utility_p0(history: Tuple[int, ...], card0: int, card1: int) -> float:
    """
    Return P0's utility at a terminal node.
    The last action is CALL_ACTION by the current player.
    The standing bid is the last non-CALL action.
    """
    # Who called?
    caller = _current_player(history)  # player who just took the CALL action
    # Actually: caller took the last action (CALL), but current_player AFTER
    # the action would be the other player. We want the player who CALLED.
    # len(history) after appending CALL: we check before appending.
    # history already contains the CALL as the last element.
    caller = (len(history) - 1) % 2  # player at turn len(history)-1

    standing_bid_idx = max(a for a in history if a != CALL_ACTION)
    bid = index_to_bid(standing_bid_idx)
    holds = _HOLDS[card0][card1][standing_bid_idx]

    # Caller wins (+1 for caller) if bid does NOT hold.
    # Caller loses (-1 for caller) if bid holds.
    if holds:
        caller_utility = -1.0
    else:
        caller_utility = +1.0

    # Return from P0's perspective.
    if caller == 0:
        return caller_utility
    else:
        return -caller_utility


# ---------------------------------------------------------------------------
# CFR state
# ---------------------------------------------------------------------------

# infostate key: (player, card_rank, history)
InfoKey = Tuple[int, int, Tuple[int, ...]]


class CFRSolver:
    """Vanilla CFR for the 1v1 n=2 Liar's Poker game."""

    def __init__(self) -> None:
        # Cumulative regrets and cumulative strategy sum, keyed by infostate.
        self._regret_sum:   Dict[InfoKey, List[float]] = {}
        self._strategy_sum: Dict[InfoKey, List[float]] = {}
        self._iterations = 0

    # ------------------------------------------------------------------
    # Core CFR traversal
    # ------------------------------------------------------------------

    def _get_strategy(self, key: InfoKey, legal: List[int]) -> List[float]:
        """Current strategy via regret matching."""
        n = len(legal)
        if key not in self._regret_sum:
            self._regret_sum[key]   = [0.0] * n
            self._strategy_sum[key] = [0.0] * n

        regrets = self._regret_sum[key]
        pos = [max(r, 0.0) for r in regrets]
        total = sum(pos)
        if total > 0:
            return [p / total for p in pos]
        return [1.0 / n] * n

    def _cfr(
        self,
        history: Tuple[int, ...],
        card0: int,
        card1: int,
        reach0: float,
        reach1: float,
    ) -> float:
        """
        Returns expected utility for P0 from this node.
        reach0, reach1 = reach probabilities for P0 and P1 respectively.
        """
        if _is_terminal(history):
            return _terminal_utility_p0(history, card0, card1)

        player = _current_player(history)
        card   = card0 if player == 0 else card1
        rank   = _card_rank(card)
        legal  = _legal_actions(history)
        key    = (player, rank, history)

        strategy = self._get_strategy(key, legal)
        n = len(legal)

        action_utils = [0.0] * n
        node_util    = 0.0

        for i, a in enumerate(legal):
            new_history = history + (a,)
            if player == 0:
                action_utils[i] = self._cfr(new_history, card0, card1,
                                            reach0 * strategy[i], reach1)
            else:
                action_utils[i] = self._cfr(new_history, card0, card1,
                                            reach0, reach1 * strategy[i])
            node_util += strategy[i] * action_utils[i]

        # Accumulate regrets and strategy sum.
        opponent_reach = reach1 if player == 0 else reach0
        my_reach       = reach0 if player == 0 else reach1
        sign           = 1.0 if player == 0 else -1.0

        for i in range(n):
            regret = sign * (action_utils[i] - node_util)
            self._regret_sum[key][i]   += opponent_reach * regret
            self._strategy_sum[key][i] += my_reach * strategy[i]

        return node_util

    # ------------------------------------------------------------------
    # One full iteration over all 2652 ordered deals
    # ------------------------------------------------------------------

    def iterate(self) -> float:
        """Run one CFR iteration. Returns the average P0 utility over all deals."""
        total_util = 0.0
        count = 0
        for c0 in range(52):
            for c1 in range(52):
                if c0 == c1:
                    continue
                # Uniform prior over deals: each deal has equal weight.
                total_util += self._cfr((), c0, c1, 1.0, 1.0)
                count += 1
        self._iterations += 1
        return total_util / count

    def run(self, n_iterations: int = 10_000, verbose: bool = False) -> None:
        for i in range(n_iterations):
            util = self.iterate()
            if verbose and (i + 1) % 1000 == 0:
                exp = self.exploitability()
                print(f"  iter {i+1:6d}: game_value={util:+.4f}  exploitability={exp:.6f}")

    # ------------------------------------------------------------------
    # Average strategy extraction
    # ------------------------------------------------------------------

    def average_strategy(self, key: InfoKey, legal: List[int]) -> List[float]:
        if key not in self._strategy_sum:
            n = len(legal)
            return [1.0 / n] * n
        ss = self._strategy_sum[key]
        total = sum(ss)
        if total > 0:
            return [s / total for s in ss]
        n = len(legal)
        return [1.0 / n] * n

    # ------------------------------------------------------------------
    # Exploitability (exact best-response traversal)
    # ------------------------------------------------------------------

    def _best_response_value(
        self,
        history: Tuple[int, ...],
        card0: int,
        card1: int,
        br_player: int,
    ) -> float:
        """
        Compute the best-response value for br_player against the average
        strategy of the other player.
        """
        if _is_terminal(history):
            u = _terminal_utility_p0(history, card0, card1)
            return u if br_player == 0 else -u

        player = _current_player(history)
        card   = card0 if player == 0 else card1
        rank   = _card_rank(card)
        legal  = _legal_actions(history)
        key    = (player, rank, history)

        if player == br_player:
            # Best response: take the max.
            best = float("-inf")
            for a in legal:
                v = self._best_response_value(history + (a,), card0, card1, br_player)
                if v > best:
                    best = v
            return best
        else:
            # Opponent plays average strategy.
            strat = self.average_strategy(key, legal)
            val = 0.0
            for i, a in enumerate(legal):
                val += strat[i] * self._best_response_value(
                    history + (a,), card0, card1, br_player)
            return val

    def exploitability(self) -> float:
        """
        Nash convergence measure (NashConv / 2 = per-player exploitability).
        Sum over all deals of the best-response values for each player, then
        normalize by deal count. Returns a value ≥ 0; 0 = Nash equilibrium.
        """
        br0_total = 0.0  # P0's BR value against P1's avg strategy
        br1_total = 0.0  # P1's BR value against P0's avg strategy
        count = 0
        for c0 in range(52):
            for c1 in range(52):
                if c0 == c1:
                    continue
                br0_total += self._best_response_value((), c0, c1, 0)
                br1_total += self._best_response_value((), c0, c1, 1)
                count += 1
        # NashConv = E[BR0] + E[BR1].  Per-player exploitability = NashConv/2.
        nashconv = (br0_total + br1_total) / count
        return nashconv / 2.0

    # ------------------------------------------------------------------
    # Mixed opening frequencies by rank
    # ------------------------------------------------------------------

    def opening_mix_by_rank(self) -> Dict[int, Dict[str, float]]:
        """
        For each rank r (0..12), return P0's average-strategy opening bid
        distribution at the root (empty history).

        Returns: {rank: {bid_str: probability, ...}, ...}
        Bids with probability < 0.001 are omitted for readability.
        """
        legal = _legal_actions(())  # all bids; no call at root
        result = {}
        for rank in range(13):
            # Use any card of this rank (e.g. rank*4 = clubs suit).
            card = rank * 4
            key: InfoKey = (0, rank, ())
            strat = self.average_strategy(key, legal)
            dist = {}
            for i, a in enumerate(legal):
                if strat[i] >= 0.001:
                    bid_str = str(index_to_bid(a))
                    dist[bid_str] = round(strat[i], 5)
            result[rank] = dist
        return result

    # ------------------------------------------------------------------
    # Game value under average strategies
    # ------------------------------------------------------------------

    def _avg_value_traverse(
        self,
        history: Tuple[int, ...],
        card0: int,
        card1: int,
    ) -> float:
        """P0 utility when both players use their average strategies."""
        if _is_terminal(history):
            return _terminal_utility_p0(history, card0, card1)
        player = _current_player(history)
        card   = card0 if player == 0 else card1
        rank   = _card_rank(card)
        legal  = _legal_actions(history)
        key    = (player, rank, history)
        strat  = self.average_strategy(key, legal)
        val = 0.0
        for i, a in enumerate(legal):
            val += strat[i] * self._avg_value_traverse(history + (a,), card0, card1)
        return val

    def game_value(self) -> float:
        """P0's expected utility under the average strategy profile."""
        total = 0.0
        count = 0
        for c0 in range(52):
            for c1 in range(52):
                if c0 == c1:
                    continue
                total += self._avg_value_traverse((), c0, c1)
                count += 1
        return total / count

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize solver state for JSON caching."""
        return {
            "iterations": self._iterations,
            "regret_sum": {
                str(k): v for k, v in self._regret_sum.items()
            },
            "strategy_sum": {
                str(k): v for k, v in self._strategy_sum.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CFRSolver":
        import ast
        solver = cls()
        solver._iterations = d["iterations"]
        solver._regret_sum = {
            ast.literal_eval(k): v for k, v in d["regret_sum"].items()
        }
        solver._strategy_sum = {
            ast.literal_eval(k): v for k, v in d["strategy_sum"].items()
        }
        return solver


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_cfr_nash(
    n_iterations: int = 10_000,
    verbose: bool = False,
    force_recompute: bool = False,
) -> dict:
    """
    Return the CFR Nash solution for the 1v1 n=2 exact-rules game.

    Loads from JSON cache if available and n_iterations matches; otherwise
    runs CFR and saves.

    Returned dict keys:
        iterations      — int
        exploitability  — float
        game_value      — float (P0's equilibrium EV, should be near 0 for
                          the symmetric 1-card game — actually negative due to
                          first-mover disadvantage under exact rules)
        opening_mix     — dict[rank_str → dict[bid_str → prob]]
    """
    cache = _load_cache()
    key   = str(n_iterations)

    if not force_recompute and key in cache:
        return cache[key]

    print(f"  [cfr_1v1] running {n_iterations:,} CFR iterations "
          f"(enumerating all 2652 ordered deals per iteration)...")
    solver = CFRSolver()
    solver.run(n_iterations, verbose=verbose)

    exp        = solver.exploitability()
    mix        = solver.opening_mix_by_rank()
    game_value = solver.game_value()

    entry = {
        "iterations":     n_iterations,
        "exploitability": exp,
        "game_value":     game_value,
        "opening_mix":    {str(r): mix[r] for r in range(13)},
    }

    cache[key] = entry
    _save_cache(cache)
    print(f"  [cfr_1v1] done. exploitability={exp:.6f}  game_value={game_value:+.4f}")
    return entry


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    if os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_opening_mix(mix: dict) -> None:
    from agent.game.bids import RANK_NAMES
    print("\nOpening bid mix by P0's card rank:")
    print(f"  {'Rank':<6}  {'Top bids (avg strategy)':}")
    print("  " + "-" * 70)
    for r in range(12, -1, -1):
        rank_name = RANK_NAMES[r]
        d = mix.get(str(r), {})
        top = sorted(d.items(), key=lambda kv: -kv[1])[:4]
        top_str = "  ".join(f"{bid}:{prob:.3f}" for bid, prob in top)
        print(f"  {rank_name:<6}  {top_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CFR solver for 1v1 n=2 exact-rules Liar's Poker.")
    parser.add_argument("--iters",   type=int, default=10_000,
                        help="CFR iterations (default 10,000).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print exploitability every 1000 iterations.")
    parser.add_argument("--force",   action="store_true",
                        help="Force recompute even if cached.")
    args = parser.parse_args()

    result = get_cfr_nash(
        n_iterations=args.iters,
        verbose=args.verbose,
        force_recompute=args.force,
    )

    print(f"\n=== CFR 1v1 Nash (n=2, exact rules) ===")
    print(f"  Iterations:     {result['iterations']:,}")
    print(f"  Exploitability: {result['exploitability']:.6f}")
    print(f"  Game value (P0): {result['game_value']:+.4f}")
    _print_opening_mix(result["opening_mix"])
