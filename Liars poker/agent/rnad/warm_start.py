"""
warm_start.py — Warm-start feature lookup for the R-NaD network.

Implements §4.2 and §5.4 of AGENT_DESIGN.md: instead of learning pool-hand
priors from scratch via self-play, the network receives pre-computed
probability tables as fixed input features.

At each forward pass the network receives two additional vectors:
    marginal_vec   — P(pool hand == bid | pool size n), no private info
    conditional_vec — P(pool hand == bid | pool size n, my condition C)

Both are shape (NUM_BIDS,), aligned with the all_bids() ordering.

If no condition matches the private hand, conditional_vec == marginal_vec.

Condition matching priority (most specific first):
    1. trips       — 3+ cards of the same rank
    2. pair        — 2 cards of the same rank (highest-ranked pair used)
    3. 3suited     — 3+ cards of the same suit
    4. suited      — 2 cards of the same suit (highest card used)
    5. 3range      — 3+ cards with max_rank − min_rank ≤ 4 (lowest card used)
    6. adjacent    — 2 cards with consecutive ranks (lower card used)
    7. None        — fall back to marginal

Data sources (all in agent/data/):
    extended_conditional_probs_ranked.json  — pre-computed (from prior session)
    hand_rank_probs_matrix.json             — marginal ranked probs, built here
                                              if missing (runs MC once per n)

Usage:
    lookup = WarmStartLookup()          # loads / builds caches at startup
    m, c, key = lookup.get_features(own_hand=[card_idx, ...], n=10)
    # m, c are numpy float32 arrays of shape (NUM_BIDS,)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_RNAD_DIR   = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR  = os.path.abspath(os.path.join(_RNAD_DIR, ".."))
_PAPER_DIR  = os.path.abspath(os.path.join(_AGENT_DIR, ".."))
_DATA_DIR   = os.path.join(_AGENT_DIR, "data")

for _p in (_PAPER_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from poker_math_exact import get_hand_rank_counts, ROYAL_FLUSH, STRAIGHT_FLUSH  # noqa: E402
from agent.game.bids import all_bids, NUM_BIDS, Bid  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
_RANK_OF    = {name: i for i, name in enumerate(_RANK_NAMES)}

_RANKED_COND_CACHE  = os.path.join(_DATA_DIR, "extended_conditional_probs_ranked.json")
_MARGINAL_CACHE     = os.path.join(_DATA_DIR, "hand_rank_probs_matrix.json")
_EXACT_RULES_CACHE  = os.path.join(_DATA_DIR, "exact_rules_probs.json")
_FIVE_KINGS_CACHE   = os.path.join(_DATA_DIR, "five_kings_probs.json")

_BIDS = all_bids()  # stable reference, length NUM_BIDS


# ---------------------------------------------------------------------------
# Cache: marginal per-n ranked probabilities
# ---------------------------------------------------------------------------

def _build_marginal_cache(n_samples: int = 3_000_000, seed: int = 42) -> None:
    """
    Run get_hand_rank_counts for n=5..25 and save a 10×13 count matrix per n
    to hand_rank_probs_matrix.json.  Runs once; subsequent calls load cache.
    """
    print("[warm_start] Building marginal rank-level cache (runs once)...")
    data: dict = {"n_samples": n_samples}

    for n in range(5, 26):
        print(f"  n={n}...", end=" ", flush=True)
        counts, total = get_hand_rank_counts(n, n_samples=n_samples, seed=seed)

        # Build 10×13 matrix; accumulate RF into SF+A slot.
        matrix = [[0] * 13 for _ in range(10)]
        for (ht, pr), cnt in counts.items():
            if ht == ROYAL_FLUSH:
                ht, pr = STRAIGHT_FLUSH, 12
            matrix[ht][pr] += cnt
        data[str(n)] = matrix
        print("done")

    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_MARGINAL_CACHE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[warm_start] Saved marginal cache → {_MARGINAL_CACHE}")


def _load_or_build_marginal() -> Dict[int, List[List[int]]]:
    """Return {n: 10×13 count matrix} for n=5..25, building the cache if needed."""
    if not os.path.exists(_MARGINAL_CACHE):
        _build_marginal_cache()

    with open(_MARGINAL_CACHE) as f:
        raw = json.load(f)

    return {int(k): v for k, v in raw.items() if k != "n_samples"}


# ---------------------------------------------------------------------------
# Helper: convert a 10×13 count matrix to a NUM_BIDS probability vector
# ---------------------------------------------------------------------------

def _matrix_to_prob_vec(matrix: List[List[int]]) -> np.ndarray:
    """
    Flatten a 10×13 count matrix into a (NUM_BIDS,) float32 probability vector
    aligned with all_bids() ordering.

    RF (type 9) is already merged into SF+A (type 8, rank 12) before this is
    called (done at cache-build time), so the only types present are 0..8.
    """
    total = sum(matrix[ht][pr] for ht in range(10) for pr in range(13))
    if total == 0:
        return np.zeros(NUM_BIDS, dtype=np.float32)

    vec = np.zeros(NUM_BIDS, dtype=np.float32)
    for i, bid in enumerate(_BIDS):
        vec[i] = matrix[bid.hand_type][bid.primary_rank]
    vec /= total
    return vec


# ---------------------------------------------------------------------------
# Condition matching
# ---------------------------------------------------------------------------

def match_condition(hand: List[int]) -> Optional[str]:
    """
    Return the most specific condition key for the given private hand
    (list of card indices 0-51), or None if no condition matches.

    Card encoding: card_idx = rank * 4 + suit
        rank 0=2, 1=3, ..., 12=A
        suit 0=C, 1=D, 2=H, 3=S
    """
    if not hand:
        return None

    ranks = [c >> 2 for c in hand]
    suits = [c & 3  for c in hand]

    # --- rank counter ---
    from collections import Counter
    rank_cnt = Counter(ranks)
    suit_cnt = Counter(suits)

    # 1. Trips (3+ of same rank)
    trips_ranks = [r for r, cnt in rank_cnt.items() if cnt >= 3]
    if trips_ranks:
        r = max(trips_ranks)
        return f"trips_{_RANK_NAMES[r]}"

    # 2. Pair (2 of same rank) — use highest-ranked pair
    pair_ranks = [r for r, cnt in rank_cnt.items() if cnt == 2]
    if pair_ranks:
        r = max(pair_ranks)
        return f"pair_{_RANK_NAMES[r]}"

    # 3. 3suited (3+ cards of same suit)
    flush_suits = [s for s, cnt in suit_cnt.items() if cnt >= 3]
    if flush_suits:
        s = flush_suits[0]  # at most one suit can have 3+ cards here
        suited_cards = [c >> 2 for c in hand if (c & 3) == s]
        high = max(suited_cards)
        key = f"3suited_high_{_RANK_NAMES[high]}"
        # Fall through if this exact key wasn't computed (e.g. 3suited_high_2/3)
        if _key_exists(key):
            return key

    # 4. Suited (2 cards of same suit) — use highest card in the suited pair
    suited_suits = [s for s, cnt in suit_cnt.items() if cnt == 2]
    if suited_suits:
        s = suited_suits[0]
        suited_cards = [c >> 2 for c in hand if (c & 3) == s]
        high = max(suited_cards)
        key = f"suited_high_{_RANK_NAMES[high]}"
        if _key_exists(key):
            return key

    # 5. 3range (3+ cards with max_rank − min_rank ≤ 4)
    if len(ranks) >= 3:
        unique_ranks = sorted(set(ranks))
        for i in range(len(unique_ranks) - 2):
            lo = unique_ranks[i]
            # Find the furthest rank within 4 of lo that forms a 3-card group
            hi_candidates = [r for r in unique_ranks[i:] if r - lo <= 4]
            if len(hi_candidates) >= 3:
                key = f"3range_low_{_RANK_NAMES[lo]}"
                if _key_exists(key):
                    return key

    # 6. Adjacent (2 cards with consecutive ranks)
    unique_ranks = sorted(set(ranks))
    for i in range(len(unique_ranks) - 1):
        if unique_ranks[i + 1] - unique_ranks[i] == 1:
            lo = unique_ranks[i]
            key = f"adjacent_low_{_RANK_NAMES[lo]}"
            if _key_exists(key):
                return key

    return None


# Populated by WarmStartLookup at init time.
_KNOWN_CONDITIONS: set = set()


def _key_exists(key: str) -> bool:
    return key in _KNOWN_CONDITIONS


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WarmStartLookup:
    """
    Loads and caches warm-start probability tables.  Instantiate once at
    trainer startup; call get_features() at every step.

    Parameters
    ----------
    n_samples : int
        MC samples used when (re)building the marginal cache.  Ignored if
        the cache file already exists.
    seed : int
        RNG seed for the marginal MC, for reproducibility.
    """

    def __init__(self, n_samples: int = 3_000_000, seed: int = 42) -> None:
        global _KNOWN_CONDITIONS

        # --- marginal: {n: (NUM_BIDS,) float32 array} ---
        marginal_matrices = _load_or_build_marginal()
        self._marginal: Dict[int, np.ndarray] = {
            n: _matrix_to_prob_vec(mat) for n, mat in marginal_matrices.items()
        }

        # --- conditional: {cond_key: {n: (NUM_BIDS,) float32 array}} ---
        if not os.path.exists(_RANKED_COND_CACHE):
            raise FileNotFoundError(
                f"Extended conditional prob cache not found: {_RANKED_COND_CACHE}\n"
                "Run agent/data/compute_extended_conditional_probs.py first."
            )

        with open(_RANKED_COND_CACHE) as f:
            raw = json.load(f)

        conds_raw = raw["conditions"]
        self._conditional: Dict[str, Dict[int, np.ndarray]] = {}

        for cond_key, n_dict in conds_raw.items():
            self._conditional[cond_key] = {}
            for n_str, matrix in n_dict.items():
                n = int(n_str)
                # Merge RF (type 9) counts into SF+A (type 8, rank 12)
                merged = [list(row) for row in matrix]  # deep copy
                rf_counts = merged[9]
                for pr in range(13):
                    if rf_counts[pr] > 0:
                        merged[8][12] += rf_counts[pr]
                        merged[9][pr]  = 0
                self._conditional[cond_key][n] = _matrix_to_prob_vec(merged)

        _KNOWN_CONDITIONS = set(self._conditional.keys())

    # ------------------------------------------------------------------

    def get_features(
        self,
        own_hand: List[int],
        n: int,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        """
        Return warm-start features for one player at one decision point.

        Parameters
        ----------
        own_hand : list of card indices (0-51)
        n        : total pool size (sum of all active players' hand sizes)

        Returns
        -------
        marginal_vec     : (NUM_BIDS,) float32 — P(pool == bid | n)
        conditional_vec  : (NUM_BIDS,) float32 — P(pool == bid | n, condition)
                           equals marginal_vec when no condition matches
        condition_key    : str or None — matched condition (for aux-loss target)
        """
        if n < 5 or n > 25:
            # Outside computed range: return uniform as a safe fallback
            uni = np.full(NUM_BIDS, 1.0 / NUM_BIDS, dtype=np.float32)
            return uni, uni, None

        marginal = self._marginal[n]

        condition_key = match_condition(own_hand)
        if condition_key is not None and n in self._conditional.get(condition_key, {}):
            conditional = self._conditional[condition_key][n]
        else:
            conditional = marginal
            condition_key = None

        return marginal, conditional, condition_key

    def get_aux_target(
        self,
        condition_key: Optional[str],
        n: int,
    ) -> Optional[np.ndarray]:
        """
        Return the supervised target for the auxiliary prediction head (§5.4).

        This is the same as conditional_vec from get_features(), exposed
        separately so the trainer can apply it as a loss target without
        re-running condition matching.

        Returns None when condition_key is None (no condition matched;
        training on the auxiliary loss would only push toward the marginal,
        which the main features already provide — omit the aux loss in that case).
        """
        if condition_key is None:
            return None
        n_dict = self._conditional.get(condition_key)
        if n_dict is None or n not in n_dict:
            return None
        return n_dict[n]

    # ------------------------------------------------------------------
    # Convenience accessors

    @property
    def num_features(self) -> int:
        """Total dimension of (marginal_vec, conditional_vec) concatenated."""
        return 2 * NUM_BIDS

    @property
    def known_conditions(self) -> set:
        return set(_KNOWN_CONDITIONS)

    # ------------------------------------------------------------------
    # Mode-specific probability tables (loaded lazily on first access)

    def get_exact_rules_at_least(self, n: int) -> Optional[np.ndarray]:
        """
        Return P(pool contains 5-card exact match >= bid_i | n) for exact-rules mode.
        Returns None if the cache file has not been generated yet.
        Run agent/data/compute_exact_rules_probs.py to generate it.
        """
        if not hasattr(self, '_exact_rules'):
            if not os.path.exists(_EXACT_RULES_CACHE):
                return None
            with open(_EXACT_RULES_CACHE) as f:
                raw = json.load(f)
            self._exact_rules: Dict[int, np.ndarray] = {
                int(k): np.array(v["at_least"], dtype=np.float32)
                for k, v in raw.items() if k not in ("n_samples",)
            }
        return self._exact_rules.get(n)

    def get_five_kings_at_least(self, n: int) -> Optional[np.ndarray]:
        """
        Return P(pool_best >= bid_i | n, 53-card deck) for five-kings mode.
        Index 110 = Five of a Kind Kings.
        Returns None if the cache has not been generated yet.
        Run agent/data/compute_five_kings_probs.py to generate it.
        """
        if not hasattr(self, '_five_kings'):
            if not os.path.exists(_FIVE_KINGS_CACHE):
                return None
            with open(_FIVE_KINGS_CACHE) as f:
                raw = json.load(f)
            self._five_kings: Dict[int, np.ndarray] = {
                int(k): np.array(v["at_least"], dtype=np.float32)
                for k, v in raw.items() if k not in ("n_samples", "deck_size")
            }
        return self._five_kings.get(n)

    def coverage_report(self) -> None:
        """Print a summary of condition coverage and marginal availability."""
        print(f"WarmStartLookup coverage:")
        print(f"  NUM_BIDS              : {NUM_BIDS}")
        print(f"  Feature dim (2×bids)  : {self.num_features}")
        print(f"  Conditions loaded     : {len(_KNOWN_CONDITIONS)}")
        print(f"  Pool sizes (marginal) : {sorted(self._marginal.keys())}")
        cond_types = {}
        for k in _KNOWN_CONDITIONS:
            prefix = k.split("_")[0]
            cond_types[prefix] = cond_types.get(prefix, 0) + 1
        print(f"  Condition breakdown   : {dict(sorted(cond_types.items()))}")
