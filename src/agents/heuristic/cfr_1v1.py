"""
CFR+ (Counterfactual Regret Minimization Plus) solver for the 1v1 n=2 private-info
exact-rules Liar's Poker game, bounded tree variant.

Upgraded from vanilla CFR to CFR+ (Tammelin 2014):
  - Regret matching+: regret sums floored at 0 after each update (no negative carry-forward)
  - Linear strategy averaging: strategy_sum weighted by iteration number
  Expected convergence: O(1/T) in practice vs O(1/√T) for vanilla CFR.

Game summary
------------
- Each player is dealt 1 card from a 52-card deck (pool size n=2).
- Exact-rules resolution: the pool's best hand (evaluated via
  `_evaluate_ranked`) is compared to the standing bid. At n=2 no 5-card hand
  exists; the pool is either Pair r (same rank) or High Card r (distinct).
- Player 0 bids first; players alternate.
- Three terminal actions may be available to the non-bidding player:
    CALL — claims the bid does NOT hold; caller wins (+1) iff pool != bid.
    HH   — claims the bid holds EXACTLY; declarer wins (+1) iff pool == bid.
    BID  — raise to a strictly stronger bid (continues the round).
- The tree is bounded in two ways to keep CFR tractable:
    max_bids    forces CALL after N bids have been made.
    bid_space   restricts the bid action set to a fixed subset of indices
                (default: the 26 HC+Pair bids — the only bids whose "holds"
                event has nonzero probability at n=2).

Why mixed strategies are required
----------------------------------
If P0 always bids "HC rank(c0)", P1 can infer P0's rank exactly:
  - P1 rank > P0 rank → HC (P0 rank) never holds → P1 calls, wins.
  - P1 rank < P0 rank → HC (P0 rank) holds       → P1 declares HH, wins.
  - P1 rank == P0 rank → pool = Pair, not HC     → P1 calls, wins.
Pure bidding leaks everything; Nash requires P0 to mix.

Algorithm: Vanilla CFR with full deal enumeration
-------------------------------------------------
- Information sets: (player, card_rank, history_tuple). Suit is irrelevant at
  n=2 since HC/Pair are rank-only; we collapse suits by enumerating all
  52*51=2652 ordered deals but keying infosets by rank only.
- Each outer iteration enumerates every ordered deal and performs a full tree
  traversal, accumulating regrets weighted by opponent reach.
- Average strategy: cumulative sum across iterations; extracted via
  `average_strategy()`.
- Exploitability: exact best-response traversal (no sampling).

Usage
-----
    python -m agent.baseline.cfr_1v1 --iters 2000 --max-bids 6 [--include-hh]

For long runs, use the overnight driver:

    python -m agent.baseline.cfr_1v1_overnight ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE      = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.abspath(os.path.join(_HERE, "..", ".."))
_PROBS_DIR = os.path.join(_SRC_DIR, "training", "probs")
_REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))

for _p in (_PROBS_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from poker_math_exact import _evaluate_ranked  # noqa: E402

from game.bids import (  # noqa: E402
    CALL_ACTION,
    HH_ACTION,
    HIGH_CARD,
    NUM_BIDS,
    PAIR,
    Bid,
    all_bids,
    bid_to_index,
    index_to_bid,
    normalize_hand_type,
)

_DATA_DIR   = os.path.join(_REPO_ROOT, "data", "probs")
_CACHE_FILE = os.path.join(_DATA_DIR, "cfr_1v1.json")

# ---------------------------------------------------------------------------
# Bid-space presets
# ---------------------------------------------------------------------------

def _hc_pair_bid_indices() -> tuple[int, ...]:
    """The 26 bid indices corresponding to HC 2..A and Pair 2..A."""
    idxs: list[int] = []
    for r in range(13):
        idxs.append(bid_to_index(Bid(HIGH_CARD, r)))
    for r in range(13):
        idxs.append(bid_to_index(Bid(PAIR, r)))
    return tuple(sorted(idxs))


HC_PAIR_BIDS: tuple[int, ...] = _hc_pair_bid_indices()


def _all_bid_indices() -> tuple[int, ...]:
    return tuple(range(NUM_BIDS))


# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------

def _card_rank(card: int) -> int:
    """Rank 0..12 from card index 0..51."""
    return card // 4


def _bid_holds_n2(card0: int, card1: int, bid: Bid) -> bool:
    """True iff the 2-card pool evaluates to exactly `bid`."""
    raw_t, raw_p = _evaluate_ranked([card0, card1])
    t, p = normalize_hand_type(raw_t, raw_p)
    return t == bid.hand_type and p == bid.primary_rank


def _build_holds_table() -> list[list[list[bool]]]:
    """Precompute holds[card0][card1][bid_idx] for all 52*52*NUM_BIDS combos."""
    bids = all_bids()
    return [
        [
            [_bid_holds_n2(c0, c1, b) for b in bids]
            for c1 in range(52)
        ]
        for c0 in range(52)
    ]


_HOLDS: list[list[list[bool]]] = _build_holds_table()


# ---------------------------------------------------------------------------
# Rank-level abstraction
# ---------------------------------------------------------------------------
# At n=2 the pool has only 2 cards — no 5-card hand exists — so the bid-holds
# relation depends only on card RANKS, not suits. We therefore iterate deals
# at rank granularity (13x13 = 169 ordered pairs) weighted by card counts.

def _rank_pair_weight(r0: int, r1: int) -> int:
    """Number of (c0, c1) ordered card pairs with rank(c0)=r0, rank(c1)=r1."""
    if r0 == r1:
        return 4 * 3   # pick c0 from 4 suits, c1 from remaining 3 of same rank
    return 4 * 4       # 4 suits × 4 suits


_TOTAL_DEAL_COUNT = 52 * 51  # 2652

_RANK_DEALS: list[tuple[int, int, int]] = [
    (r0, r1, _rank_pair_weight(r0, r1))
    for r0 in range(13) for r1 in range(13)
]

# Build a rank-level holds table: _HOLDS_RANK[r0][r1][bid_idx].
# Uses card (r*4) as a canonical representative per rank (suit 0).

def _build_holds_rank_table() -> list[list[list[bool]]]:
    table: list[list[list[bool]]] = []
    for r0 in range(13):
        row: list[list[bool]] = []
        for r1 in range(13):
            c0 = r0 * 4
            c1 = r1 * 4 + (1 if r1 == r0 else 0)  # avoid same card at r0==r1
            row.append(_HOLDS[c0][c1])
        table.append(row)
    return table


_HOLDS_RANK: list[list[list[bool]]] = _build_holds_rank_table()


# ---------------------------------------------------------------------------
# History / legal actions
# ---------------------------------------------------------------------------
# A history is a tuple of action ints:
#   bid_idx ∈ [0, NUM_BIDS)   → a raise
#   CALL_ACTION (110)          → the round ends: caller wins if bid doesn't hold
#   HH_ACTION   (111)          → the round ends: declarer wins if bid holds

_TERMINAL_ACTIONS: tuple[int, ...] = (CALL_ACTION, HH_ACTION)


def _current_player(history: tuple[int, ...]) -> int:
    return len(history) % 2


def _standing_bid_index(history: tuple[int, ...]) -> int | None:
    for a in reversed(history):
        if a not in _TERMINAL_ACTIONS:
            return a
    return None


def _num_bids_placed(history: tuple[int, ...]) -> int:
    return sum(1 for a in history if a not in _TERMINAL_ACTIONS)


def _is_terminal(history: tuple[int, ...]) -> bool:
    return len(history) > 0 and history[-1] in _TERMINAL_ACTIONS


def _legal_actions(
    history: tuple[int, ...],
    bid_space: tuple[int, ...],
    max_bids: int,
    include_hh: bool,
) -> list[int]:
    """Legal actions at `history` given tree constraints."""
    standing = _standing_bid_index(history)
    bids_placed = _num_bids_placed(history)

    if standing is None:
        # Root turn: P0 must bid; no call/HH legal.
        return [b for b in bid_space]

    # If we've hit the bid cap, only terminal resolutions remain.
    raise_legal = bids_placed < max_bids
    raises = [b for b in bid_space if b > standing] if raise_legal else []

    actions: list[int] = [CALL_ACTION]
    if include_hh:
        actions.append(HH_ACTION)
    actions.extend(raises)
    return actions


def _terminal_utility_p0(history: tuple[int, ...], card0: int, card1: int) -> float:
    """P0 utility at a terminal node (by card indices). `history` ends in CALL or HH."""
    return _terminal_utility_p0_rank(history, _card_rank(card0), _card_rank(card1))


def _terminal_utility_p0_rank(history: tuple[int, ...], r0: int, r1: int) -> float:
    """P0 utility at a terminal node, keyed by card ranks."""
    last = history[-1]
    resolver = (len(history) - 1) % 2
    standing = _standing_bid_index(history)
    assert standing is not None, "resolver action requires a standing bid"
    holds = _HOLDS_RANK[r0][r1][standing]
    if last == CALL_ACTION:
        resolver_util = +1.0 if not holds else -1.0
    else:  # HH_ACTION
        resolver_util = +1.0 if holds else -1.0
    return resolver_util if resolver == 0 else -resolver_util


# ---------------------------------------------------------------------------
# CFR state
# ---------------------------------------------------------------------------

# key = (player, card_rank, history_tuple)
InfoKey = tuple[int, int, tuple[int, ...]]


class CFRSolver:
    """
    CFR+ solver for the 1v1 n=2 Liar's Poker game with a bounded tree.

    Upgrades from vanilla CFR:
      - Regret matching+: regret sums are floored at 0 after each update,
        so negative regret is never carried forward. This eliminates the
        slow cancellation effect that keeps vanilla CFR at O(1/√T).
      - Linear strategy averaging: strategy_sum is weighted by the current
        iteration number rather than reach probability alone. Later iterates
        (which are closer to Nash) receive proportionally more weight,
        giving O(1/T) convergence in practice — 10–100× faster than vanilla.

    Both changes are backward-compatible with the existing serialization format.
    """

    def __init__(
        self,
        max_bids: int = 6,
        bid_space: tuple[int, ...] = HC_PAIR_BIDS,
        include_hh: bool = True,
    ) -> None:
        self.max_bids   = int(max_bids)
        self.bid_space  = tuple(sorted(bid_space))
        self.include_hh = bool(include_hh)

        self._regret_sum:   dict[InfoKey, list[float]] = {}
        self._strategy_sum: dict[InfoKey, list[float]] = {}
        self._iterations   = 0

    # ------------------------------------------------------------------

    def _get_strategy(self, key: InfoKey, legal: list[int]) -> list[float]:
        n = len(legal)
        if key not in self._regret_sum:
            self._regret_sum[key]   = [0.0] * n
            self._strategy_sum[key] = [0.0] * n
        # Regret matching+ uses floored regrets (same as vanilla — floor already applied
        # in the update step below, so values here are already >= 0).
        regrets = self._regret_sum[key]
        pos = [r if r > 0.0 else 0.0 for r in regrets]
        total = sum(pos)
        if total > 0:
            return [p / total for p in pos]
        return [1.0 / n] * n

    def _cfr(
        self,
        history: tuple[int, ...],
        r0: int,
        r1: int,
        reach0: float,
        reach1: float,
        chance_reach: float,
    ) -> float:
        if _is_terminal(history):
            return _terminal_utility_p0_rank(history, r0, r1)

        player = _current_player(history)
        rank   = r0 if player == 0 else r1
        legal  = _legal_actions(history, self.bid_space, self.max_bids, self.include_hh)
        key    = (player, rank, history)

        strategy = self._get_strategy(key, legal)
        n = len(legal)

        action_utils = [0.0] * n
        node_util    = 0.0

        for i, a in enumerate(legal):
            new_history = history + (a,)
            if player == 0:
                action_utils[i] = self._cfr(new_history, r0, r1,
                                            reach0 * strategy[i], reach1,
                                            chance_reach)
            else:
                action_utils[i] = self._cfr(new_history, r0, r1,
                                            reach0, reach1 * strategy[i],
                                            chance_reach)
            node_util += strategy[i] * action_utils[i]

        opponent_reach = reach1 if player == 0 else reach0
        my_reach       = reach0 if player == 0 else reach1
        sign           = 1.0 if player == 0 else -1.0
        cf_opp         = chance_reach * opponent_reach

        regret_sum   = self._regret_sum[key]
        strategy_sum = self._strategy_sum[key]
        # Linear weight for strategy averaging: later iterations count more.
        lin_weight = float(self._iterations + 1)
        for i in range(n):
            regret = sign * (action_utils[i] - node_util)
            # CFR+: floor regret sum at 0 — never carry negative regret forward.
            regret_sum[i] = max(0.0, regret_sum[i] + cf_opp * regret)
            # Linear averaging: weight current strategy by iteration number.
            strategy_sum[i] += lin_weight * my_reach * strategy[i]

        return node_util

    # ------------------------------------------------------------------

    def iterate(self) -> float:
        """One CFR iteration over all 169 rank-pair deals (weighted)."""
        total_util = 0.0
        total_w    = 0
        for r0, r1, w in _RANK_DEALS:
            chance = w / _TOTAL_DEAL_COUNT
            util   = self._cfr((), r0, r1, 1.0, 1.0, chance)
            total_util += util * w
            total_w    += w
        self._iterations += 1
        return total_util / total_w

    def run(self, n_iterations: int = 1000, verbose: bool = False) -> None:
        for i in range(n_iterations):
            util = self.iterate()
            if verbose and (i + 1) % 100 == 0:
                exp = self.exploitability()
                print(f"  iter {i+1:6d}: game_value={util:+.4f}  exploitability={exp:.6f}")

    # ------------------------------------------------------------------

    def average_strategy(self, key: InfoKey, legal: list[int]) -> list[float]:
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

    def _best_response_value(
        self,
        history: tuple[int, ...],
        r0: int,
        r1: int,
        br_player: int,
    ) -> float:
        if _is_terminal(history):
            u = _terminal_utility_p0_rank(history, r0, r1)
            return u if br_player == 0 else -u

        player = _current_player(history)
        rank   = r0 if player == 0 else r1
        legal  = _legal_actions(history, self.bid_space, self.max_bids, self.include_hh)
        key    = (player, rank, history)

        if player == br_player:
            best = float("-inf")
            for a in legal:
                v = self._best_response_value(history + (a,), r0, r1, br_player)
                if v > best:
                    best = v
            return best
        strat = self.average_strategy(key, legal)
        val = 0.0
        for i, a in enumerate(legal):
            val += strat[i] * self._best_response_value(
                history + (a,), r0, r1, br_player)
        return val

    def exploitability(self) -> float:
        """Per-player exploitability = (E[BR0] + E[BR1]) / 2. Nash → 0."""
        br0_total = 0.0
        br1_total = 0.0
        total_w   = 0
        for r0, r1, w in _RANK_DEALS:
            br0_total += w * self._best_response_value((), r0, r1, 0)
            br1_total += w * self._best_response_value((), r0, r1, 1)
            total_w   += w
        nashconv = (br0_total + br1_total) / total_w
        return nashconv / 2.0

    # ------------------------------------------------------------------

    def opening_mix_by_rank(self) -> dict[int, dict[str, float]]:
        """P0's root opening-bid distribution, keyed by card rank."""
        legal = _legal_actions((), self.bid_space, self.max_bids, self.include_hh)
        result: dict[int, dict[str, float]] = {}
        for rank in range(13):
            key: InfoKey = (0, rank, ())
            strat = self.average_strategy(key, legal)
            dist: dict[str, float] = {}
            for i, a in enumerate(legal):
                if strat[i] >= 0.001:
                    dist[str(index_to_bid(a))] = round(strat[i], 5)
            result[rank] = dist
        return result

    def response_mix_by_rank(
        self,
        opening_bid_idx: int,
    ) -> dict[int, dict[str, float]]:
        """P1's response distribution after a given opening bid, keyed by P1 rank."""
        history = (opening_bid_idx,)
        legal = _legal_actions(history, self.bid_space, self.max_bids, self.include_hh)
        result: dict[int, dict[str, float]] = {}
        for rank in range(13):
            key: InfoKey = (1, rank, history)
            strat = self.average_strategy(key, legal)
            dist: dict[str, float] = {}
            for i, a in enumerate(legal):
                if strat[i] < 0.001:
                    continue
                if a == CALL_ACTION:
                    name = "CALL"
                elif a == HH_ACTION:
                    name = "HH"
                else:
                    name = str(index_to_bid(a))
                dist[name] = round(strat[i], 5)
            result[rank] = dist
        return result

    # ------------------------------------------------------------------

    def _avg_value_traverse(
        self,
        history: tuple[int, ...],
        r0: int,
        r1: int,
    ) -> float:
        if _is_terminal(history):
            return _terminal_utility_p0_rank(history, r0, r1)
        player = _current_player(history)
        rank   = r0 if player == 0 else r1
        legal  = _legal_actions(history, self.bid_space, self.max_bids, self.include_hh)
        key    = (player, rank, history)
        strat  = self.average_strategy(key, legal)
        val = 0.0
        for i, a in enumerate(legal):
            val += strat[i] * self._avg_value_traverse(history + (a,), r0, r1)
        return val

    def game_value(self) -> float:
        """P0 expected utility under the average-strategy profile."""
        total   = 0.0
        total_w = 0
        for r0, r1, w in _RANK_DEALS:
            total   += w * self._avg_value_traverse((), r0, r1)
            total_w += w
        return total / total_w

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "iterations":   self._iterations,
            "max_bids":     self.max_bids,
            "bid_space":    list(self.bid_space),
            "include_hh":   self.include_hh,
            "regret_sum":   {repr(k): v for k, v in self._regret_sum.items()},
            "strategy_sum": {repr(k): v for k, v in self._strategy_sum.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> CFRSolver:
        import ast
        solver = cls(
            max_bids=int(d.get("max_bids", 6)),
            bid_space=tuple(d.get("bid_space", HC_PAIR_BIDS)),
            include_hh=bool(d.get("include_hh", True)),
        )
        solver._iterations = int(d["iterations"])
        solver._regret_sum = {
            ast.literal_eval(k): list(v) for k, v in d["regret_sum"].items()
        }
        solver._strategy_sum = {
            ast.literal_eval(k): list(v) for k, v in d["strategy_sum"].items()
        }
        return solver


# ---------------------------------------------------------------------------
# Public entry point / cache
# ---------------------------------------------------------------------------

def get_cfr_nash(
    n_iterations: int = 1000,
    max_bids: int = 6,
    bid_space: tuple[int, ...] = HC_PAIR_BIDS,
    include_hh: bool = True,
    verbose: bool = False,
    force_recompute: bool = False,
) -> dict:
    """Return the CFR Nash solution (or load from cache)."""
    cache = _load_cache()
    key = _cache_key(n_iterations, max_bids, bid_space, include_hh)

    if not force_recompute and key in cache:
        return cache[key]

    print(f"  [cfr_1v1] running {n_iterations:,} CFR iterations  "
          f"max_bids={max_bids}  |bids|={len(bid_space)}  include_hh={include_hh}")
    solver = CFRSolver(max_bids=max_bids, bid_space=bid_space, include_hh=include_hh)
    solver.run(n_iterations, verbose=verbose)

    entry = _solver_summary(solver)
    cache[key] = entry
    _save_cache(cache)
    print(f"  [cfr_1v1] done. exploitability={entry['exploitability']:.6f}  "
          f"game_value={entry['game_value']:+.4f}")
    return entry


def _solver_summary(solver: CFRSolver) -> dict:
    mix        = solver.opening_mix_by_rank()
    game_value = solver.game_value()
    exp        = solver.exploitability()
    return {
        "iterations":     solver._iterations,
        "max_bids":       solver.max_bids,
        "bid_space_size": len(solver.bid_space),
        "include_hh":     solver.include_hh,
        "exploitability": exp,
        "game_value":     game_value,
        "opening_mix":    {str(r): mix[r] for r in range(13)},
    }


def _cache_key(n_iterations: int, max_bids: int,
               bid_space: tuple[int, ...], include_hh: bool) -> str:
    bid_tag = "hcpair" if tuple(bid_space) == HC_PAIR_BIDS else f"bs{len(bid_space)}"
    hh_tag  = "hh" if include_hh else "nohh"
    return f"iters{n_iterations}_mb{max_bids}_{bid_tag}_{hh_tag}"


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
    from game.bids import RANK_NAMES
    print("\nOpening bid mix by P0's card rank:")
    print(f"  {'Rank':<6}  Top bids (avg strategy)")
    print("  " + "-" * 70)
    for r in range(12, -1, -1):
        rank_name = RANK_NAMES[r]
        d = mix.get(str(r), {}) if str(r) in mix else mix.get(r, {})
        top = sorted(d.items(), key=lambda kv: -kv[1])[:4]
        top_str = "  ".join(f"{bid}:{prob:.3f}" for bid, prob in top)
        print(f"  {rank_name:<6}  {top_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CFR solver for 1v1 n=2 exact-rules Liar's Poker (bounded).")
    parser.add_argument("--iters",      type=int, default=1000)
    parser.add_argument("--max-bids",   type=int, default=6,
                        help="Force CALL after this many bids (default 6).")
    parser.add_argument("--no-hh",      action="store_true",
                        help="Disable the High Hand declaration action.")
    parser.add_argument("--full-bids",  action="store_true",
                        help="Use all 110 bids (NOT recommended — tree explodes).")
    parser.add_argument("--verbose",    action="store_true")
    parser.add_argument("--force",      action="store_true")
    args = parser.parse_args()

    bid_space = _all_bid_indices() if args.full_bids else HC_PAIR_BIDS

    result = get_cfr_nash(
        n_iterations   = args.iters,
        max_bids       = args.max_bids,
        bid_space      = bid_space,
        include_hh     = not args.no_hh,
        verbose        = args.verbose,
        force_recompute= args.force,
    )

    print("\n=== CFR 1v1 Nash (n=2, exact rules, bounded) ===")
    print(f"  Iterations:     {result['iterations']:,}")
    print(f"  max_bids:       {result['max_bids']}")
    print(f"  |bid_space|:    {result['bid_space_size']}")
    print(f"  include_hh:     {result['include_hh']}")
    print(f"  Exploitability: {result['exploitability']:.6f}")
    print(f"  Game value(P0): {result['game_value']:+.4f}")
    _print_opening_mix(result["opening_mix"])
