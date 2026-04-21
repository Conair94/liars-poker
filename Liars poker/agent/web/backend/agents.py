"""
Web agent implementations.  See agent/AGENT_CATALOG.md for the full registry.

Standard rules (52-card deck):
  RandomAgent        — uniform random legal action
  BlindBaselineAgent — marginal 50% threshold (ignores private cards)
  ConditionalAgent   — threshold strategy with private-card conditional priors

Exact Hand Rules mode (52-card deck, exact 5-card subset required):
  ExactRulesBlindAgent — threshold strategy using exact-rules probability table

Five-Kings mode (53-card deck, 5K Kings > SF):
  FiveKingsBlindAgent — threshold strategy calibrated for 53-card deck

AGENT_REGISTRY maps agent_key → metadata including the exact ruleset the agent
was designed for. Add new agents here; the web UI and new_game route read from
this registry so rules are always consistent with the agent.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR   = os.path.abspath(os.path.join(_BACKEND_DIR, "..", ".."))
_PAPER_DIR   = os.path.abspath(os.path.join(_AGENT_DIR, ".."))
for _p in (_PAPER_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.game.engine import MatchState                                     # noqa: E402
from agent.game.bids import CALL_ACTION, HH_ACTION, NUM_BIDS, bid_to_index   # noqa: E402
import numpy as np                                                             # noqa: E402


# ---------------------------------------------------------------------------
class RandomAgent:
    """Picks a uniformly random legal action."""

    def choose_action(self, state: MatchState) -> int:
        return random.choice(state.legal_actions())


# ---------------------------------------------------------------------------
class BlindBaselineAgent:
    """
    Marginal threshold strategy: bids the highest hand where
    P(pool >= bid | total cards = n) >= 50%, and calls when the standing
    bid exceeds that threshold.

    Data sources (in priority order):
      - WarmStartLookup marginal vector for n=5..25 (MC-derived)
      - blind_equilibrium p_at_least for n=2..4 (exact combinatorics)

    BLIND_EQ initial_bid is used as an opening-bid hint for n<=10 only
    when it does not exceed the 50% threshold.
    """

    def __init__(self) -> None:
        self._rng = random.Random()

    def _get_p_at_least(self, n: int, state: MatchState) -> Optional[np.ndarray]:
        """Return (NUM_BIDS,) P(pool >= bid_i | n) appropriate to game mode."""
        if n >= 5:
            try:
                lookup = _get_warm_start()
                # Exact rules mode: use dedicated exact-rules probability table if available
                if getattr(state, 'exact_rules', False):
                    exact_pal = lookup.get_exact_rules_at_least(n)
                    if exact_pal is not None:
                        return exact_pal
                # Five-kings mode: use 53-card deck table if available
                if getattr(state, 'five_kings', False):
                    fk_pal = lookup.get_five_kings_at_least(n)
                    if fk_pal is not None:
                        return fk_pal
                # Standard: marginal cumulative
                marginal, _, _ = lookup.get_features([], n)
                return np.flip(np.cumsum(np.flip(marginal))).astype(np.float32)
            except Exception:
                pass
        # n < 5 or WarmStart unavailable: exact blind equilibrium
        try:
            from agent.baseline.blind_equilibrium import get_blind_equilibrium
            eq = get_blind_equilibrium(n)
            return np.array(eq["p_at_least"], dtype=np.float32)
        except Exception:
            return None

    def choose_action(self, state: MatchState) -> int:
        n = sum(state.hand_sizes[s] for s in state.active_seats())

        p_at_least = self._get_p_at_least(n, state)
        if p_at_least is None:
            return self._rng.choice(state.legal_actions())

        rs    = state.round_state
        legal = state.legal_actions()

        # Threshold = highest bid index where P(pool >= bid) >= 50%
        threshold_idx = 0
        for i in range(NUM_BIDS - 1, -1, -1):
            if p_at_least[i] >= 0.5:
                threshold_idx = i
                break

        if rs.current_bid is None:
            # Use equilibrium initial_bid as opening hint (n<=10 only, must not exceed threshold)
            if n <= 10:
                try:
                    from agent.baseline.blind_equilibrium import get_blind_equilibrium
                    eq = get_blind_equilibrium(n)
                    initial = eq["initial_bid"]
                    if initial <= threshold_idx and initial in legal:
                        return initial
                except Exception:
                    pass
            return threshold_idx if threshold_idx in legal else legal[0]

        # Responding: call if standing bid is above threshold, else raise to threshold
        cur_idx = bid_to_index(rs.current_bid)
        if CALL_ACTION in legal and float(p_at_least[cur_idx]) < 0.5:
            return CALL_ACTION

        bid_candidates = [a for a in legal if a not in (CALL_ACTION, HH_ACTION)]
        if not bid_candidates:
            return CALL_ACTION

        for a in bid_candidates:
            if a >= threshold_idx:
                return a

        return bid_candidates[0]


# ---------------------------------------------------------------------------
# Module-level WarmStartLookup cache (loaded once on first use).
_WARM_START: Optional[object] = None


def _get_warm_start():
    global _WARM_START
    if _WARM_START is None:
        from agent.rnad.warm_start import WarmStartLookup
        _WARM_START = WarmStartLookup()
    return _WARM_START


class ConditionalAgent:
    """
    Threshold strategy using private-card conditional priors from WarmStartLookup.

    For each decision:
      - Computes P(pool >= bid | n, own_hand_condition) from the conditional vec.
      - Threshold bid = highest bid where that probability ≥ 50%.
      - If no standing bid: bid the threshold.
      - If standing bid is above threshold (P < 50%): call (it's likely a bluff).
      - If standing bid is at/below threshold: raise to the threshold.

    Falls back to BlindBaselineAgent (marginal threshold strategy) for n outside
    the WarmStartLookup range (n<5 or n>25).
    """

    def __init__(self) -> None:
        self._blind = BlindBaselineAgent()

    def choose_action(self, state: MatchState) -> int:
        n     = sum(state.hand_sizes[s] for s in state.active_seats())
        rs    = state.round_state
        seat  = rs.current_player
        legal = state.legal_actions()

        # Exact rules or five-kings: conditional tables aren't calibrated for these modes,
        # so fall back to BlindBaselineAgent which will use the mode-specific PAL tables.
        if getattr(state, 'exact_rules', False) or getattr(state, 'five_kings', False):
            return self._blind.choose_action(state)

        # WarmStartLookup covers n=5..25; for small n use blind equilibrium.
        if n < 5 or n > 25:
            return self._blind.choose_action(state)

        try:
            lookup = _get_warm_start()
            _, cond_vec, _ = lookup.get_features(rs.hands[seat], n)
        except Exception:
            return self._blind.choose_action(state)

        # Cumulative: P(pool >= bid_i | condition) = sum(cond_vec[i:])
        cond_p_at_least: list = np.flip(np.cumsum(np.flip(cond_vec))).tolist()

        # Threshold = highest bid index where P(pool >= bid) >= 0.5
        threshold_idx = 0
        for i in range(NUM_BIDS - 1, -1, -1):
            if cond_p_at_least[i] >= 0.5:
                threshold_idx = i
                break

        # --- First bid ---
        if rs.current_bid is None:
            if threshold_idx in legal:
                return threshold_idx
            # All bids are legal when no standing bid exists.
            return legal[0]

        # --- Responding to a standing bid ---
        cur_idx = bid_to_index(rs.current_bid)
        cur_p   = cond_p_at_least[cur_idx]

        # Standing bid is above our threshold (likely a bluff): call.
        if CALL_ACTION in legal and cur_p < 0.5:
            return CALL_ACTION

        # Standing bid is still within range: raise to the threshold (or just above).
        bid_candidates = [a for a in legal if a not in (CALL_ACTION, HH_ACTION)]
        if not bid_candidates:
            return CALL_ACTION

        for a in bid_candidates:
            if a >= threshold_idx:
                return a

        # All legal bids are already above threshold — smallest legal raise.
        return bid_candidates[0]


# ---------------------------------------------------------------------------
class ExactRulesBlindAgent:
    """
    "Peak probability" strategy for Exact Hand Rules mode (internal fallback).

    Uses exact_prob[i] = P(pool contains 5-card subset with best hand exactly == bid_i | n)
    from exact_rules_probs.json.  This is a per-bid probability, NOT a cumulative table.

    Strategy:
      call_threshold = 0.3 * max(exact_prob[n])
      Call when standing bid's exact prob < call_threshold.
      Bid: among legal raises, pick the one with the highest exact probability.

    Falls back to random if the cache has not been generated yet.

    See agent/AGENT_CATALOG.md for details.
    """

    def __init__(self) -> None:
        self._rng = random.Random()

    @staticmethod
    def _best_bid(candidates: list, exact_prob: np.ndarray) -> int:
        return max(candidates, key=lambda a: float(exact_prob[a]) if a < len(exact_prob) else 0.0)

    def choose_action(self, state: MatchState) -> int:
        n     = sum(state.hand_sizes[s] for s in state.active_seats())
        rs    = state.round_state
        legal = state.legal_actions()

        exact_prob: Optional[np.ndarray] = None
        if n >= 5:
            try:
                lookup = _get_warm_start()
                exact_prob = lookup.get_exact_rules_exact(n)
            except Exception:
                pass

        if exact_prob is None:
            return self._rng.choice(legal)

        max_p = float(np.max(exact_prob))
        call_threshold = 0.3 * max_p
        bid_candidates = [a for a in legal if a not in (CALL_ACTION, HH_ACTION)]

        if rs.current_bid is None:
            return self._best_bid(bid_candidates, exact_prob) if bid_candidates else legal[0]

        cur_idx = bid_to_index(rs.current_bid)
        cur_p = float(exact_prob[cur_idx]) if cur_idx < len(exact_prob) else 0.0
        if CALL_ACTION in legal and cur_p < call_threshold:
            return CALL_ACTION
        if not bid_candidates:
            return CALL_ACTION
        return self._best_bid(bid_candidates, exact_prob)


# ---------------------------------------------------------------------------
class ExactRulesConditionalAgent:
    """
    Conditional "peak probability" strategy for Exact Hand Rules mode.

    Adjusts per-bid exact probabilities using the agent's private hand via a
    Bayesian likelihood-ratio correction:

        adj_exact[i] ≈ exact_prob[i] × (cond_pmf[i] / marginal_pmf[i])

    Where cond_pmf and marginal_pmf are derived from the conditional and marginal
    p_at_least tables (CONDITIONAL_PAL / MARGINAL_PAL) converted to PMFs.
    Ratio is capped at 10 to prevent runaway values.

    Falls back to ExactRulesBlindAgent if no condition matches or cache missing.

    See agent/AGENT_CATALOG.md for details.
    """

    def __init__(self) -> None:
        self._blind = ExactRulesBlindAgent()

    @staticmethod
    def _pmf_from(pal: np.ndarray) -> np.ndarray:
        """Convert p_at_least vector to PMF."""
        pmf = np.zeros_like(pal)
        pmf[:-1] = np.maximum(0, pal[:-1] - pal[1:])
        pmf[-1]  = max(0.0, float(pal[-1]))
        return pmf

    def choose_action(self, state: MatchState) -> int:
        n     = sum(state.hand_sizes[s] for s in state.active_seats())
        rs    = state.round_state
        seat  = rs.current_player
        legal = state.legal_actions()

        if n < 5 or n > 25:
            return self._blind.choose_action(state)

        try:
            lookup = _get_warm_start()
            exact_prob = lookup.get_exact_rules_exact(n)
        except Exception:
            exact_prob = None

        if exact_prob is None:
            return self._blind.choose_action(state)

        # Likelihood-ratio adjustment from own hand condition.
        adj_exact = exact_prob.copy()
        try:
            marginal, cond_vec, _ = lookup.get_features(rs.hands[seat], n)
            marg_pmf = self._pmf_from(
                np.flip(np.cumsum(np.flip(marginal))).astype(np.float32)
            )
            cond_pmf = self._pmf_from(
                np.flip(np.cumsum(np.flip(cond_vec))).astype(np.float32)
            )
            mask = marg_pmf > 1e-9
            ratio = np.where(mask, np.minimum(cond_pmf / np.where(mask, marg_pmf, 1.0), 10.0), 1.0)
            adj_exact = exact_prob * ratio
        except Exception:
            pass  # Fall through with unadjusted exact_prob

        max_p = float(np.max(adj_exact))
        call_threshold = 0.3 * max_p
        bid_candidates = [a for a in legal if a not in (CALL_ACTION, HH_ACTION)]

        def best_bid(cands: list) -> int:
            return max(cands, key=lambda a: float(adj_exact[a]) if a < len(adj_exact) else 0.0)

        if rs.current_bid is None:
            return best_bid(bid_candidates) if bid_candidates else legal[0]

        cur_idx = bid_to_index(rs.current_bid)
        cur_p = float(adj_exact[cur_idx]) if cur_idx < len(adj_exact) else 0.0
        if CALL_ACTION in legal and cur_p < call_threshold:
            return CALL_ACTION
        if not bid_candidates:
            return CALL_ACTION
        return best_bid(bid_candidates)


# ---------------------------------------------------------------------------
class BiasedRandomAgent:
    """
    Biased random agent: accepts (bids) with probability `accept_prob` and calls
    with probability `1 - accept_prob`.  When bidding, picks uniformly from the
    next 5 legal bids in the partial order of hand strengths.

    The 50% variant is the standard RandomAgent.  This class parameterises the
    call/bid ratio to create agents that over-fold or over-bluff.

    Data: None.
    Modes: All modes (standard, exact, five-kings).
    """

    def __init__(self, accept_prob: float) -> None:
        self._accept_prob = accept_prob
        self._rng = random.Random()

    def choose_action(self, state: MatchState) -> int:
        rs = state.round_state
        legal = state.legal_actions()
        bid_candidates = [a for a in legal if a not in (CALL_ACTION, HH_ACTION)]

        # No current bid → must bid (nothing to call)
        if rs.current_bid is None or CALL_ACTION not in legal:
            pool = bid_candidates[:5] or legal
            return self._rng.choice(pool)

        if self._rng.random() < self._accept_prob:
            # Accept: make a bid from the next up-to-5 legal bids
            pool = bid_candidates[:5] or legal
            return self._rng.choice(pool)
        else:
            return CALL_ACTION


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------
# Each entry maps an agent key to its display metadata and the exact match
# configuration it was designed for. The web UI reads this to auto-populate
# rule settings when the user picks an agent.
#
# Rules fields map directly to kwargs accepted by new_match():
#   exact_rules  — bool: bid must hold exactly (vs. at-least)
#   high_hand    — bool: High Hand declaration action enabled
#   five_kings   — bool: 53-card deck with Five of a Kind Kings
#
# To add a new agent: append an entry here and implement its class below.
AGENT_REGISTRY: dict = {
    "biased30": {
        "display":      "Timid (30%)",
        "description":  "Accepts a bid (bids higher) only 30% of the time; calls bluff 70% of the time. When bidding, picks randomly from the next 5 valid hands.",
        "rules": {
            "exact_rules": False,
            "high_hand":   False,
            "five_kings":  False,
        },
        "rules_label":  "Standard — at-least rules, 52-card deck",
        "class":        "BiasedRandom30Agent",
    },
    "biased40": {
        "display":      "Passive (40%)",
        "description":  "Accepts a bid 40% of the time; calls bluff 60% of the time. When bidding, picks randomly from the next 5 valid hands.",
        "rules": {
            "exact_rules": False,
            "high_hand":   False,
            "five_kings":  False,
        },
        "rules_label":  "Standard — at-least rules, 52-card deck",
        "class":        "BiasedRandom40Agent",
    },
    "random": {
        "display":      "Coin Flip (50%)",
        "description":  "Picks any legal action uniformly at random — approximately 50/50 bid vs call. Useful as a baseline.",
        "rules": {
            "exact_rules": False,
            "high_hand":   False,
            "five_kings":  False,
        },
        "rules_label":  "Standard — at-least rules, 52-card deck",
        "class":        "RandomAgent",
    },
    "biased60": {
        "display":      "Active (60%)",
        "description":  "Accepts a bid 60% of the time; calls bluff 40% of the time. When bidding, picks randomly from the next 5 valid hands.",
        "rules": {
            "exact_rules": False,
            "high_hand":   False,
            "five_kings":  False,
        },
        "rules_label":  "Standard — at-least rules, 52-card deck",
        "class":        "BiasedRandom60Agent",
    },
    "biased70": {
        "display":      "Aggressive (70%)",
        "description":  "Accepts a bid 70% of the time; calls bluff only 30% of the time. When bidding, picks randomly from the next 5 valid hands.",
        "rules": {
            "exact_rules": False,
            "high_hand":   False,
            "five_kings":  False,
        },
        "rules_label":  "Standard — at-least rules, 52-card deck",
        "class":        "BiasedRandom70Agent",
    },
    "blind": {
        "display":      "Blind Oracle",
        "description":  "Bids at the ~50% probability threshold, ignoring private cards (N=2 backward-induction equilibrium).",
        "rules": {
            "exact_rules": False,
            "high_hand":   False,
            "five_kings":  False,
        },
        "rules_label":  "Standard — at-least rules, 52-card deck",
        "class":        "BlindBaselineAgent",
    },
    "conditional": {
        "display":      "Card Hawk",
        "description":  "Adjusts the 50% threshold bid using private-hand conditional probability tables.",
        "rules": {
            "exact_rules": False,
            "high_hand":   False,
            "five_kings":  False,
        },
        "rules_label":  "Standard — at-least rules, 52-card deck",
        "class":        "ConditionalAgent",
    },
    "exactconditional": {
        "display":      "Sharp Eye",
        "description":  "Peak-probability strategy for exact-rules mode, with Bayesian private-hand adjustment.",
        "rules": {
            "exact_rules": True,
            "high_hand":   True,
            "five_kings":  False,
        },
        "rules_label":  "Exact rules + High Hand declaration, 52-card deck",
        "class":        "ExactRulesConditionalAgent",
    },
    "fivekingsblind": {
        "display":      "Five Kings Blind",
        "description":  "50% threshold strategy calibrated for the 53-card Five Kings deck.",
        "rules": {
            "exact_rules": False,
            "high_hand":   False,
            "five_kings":  True,
        },
        "rules_label":  "Five Kings — 53-card deck (5K Kings ranks above Straight Flush)",
        "class":        "FiveKingsBlindAgent",
    },
    "cfr_nash_mb3": {
        "display":      "Nash Bot",
        "description":  "Approximate Nash equilibrium strategy computed by 20,000 iterations of Counterfactual Regret Minimization. Bids High Card / Pair only; uses mixed strategies to prevent rank inference. Single-round win rate: +7–20% vs baselines on exact-rules n=2.",
        "rules": {
            "exact_rules": True,
            "high_hand":   True,
            "five_kings":  False,
        },
        "rules_label":  "Exact rules + High Hand, 52-card deck (trained domain: n=2, max 3 bids)",
        "class":        "CFRNashAgent",
    },
}

def _make_cfr_nash():
    from agent.web.backend.cfr_nash_agent import CFRNashAgent
    return CFRNashAgent()


_AGENT_CLASS_MAP = {
    "RandomAgent":              lambda: RandomAgent(),
    "BiasedRandom30Agent":      lambda: BiasedRandomAgent(0.30),
    "BiasedRandom40Agent":      lambda: BiasedRandomAgent(0.40),
    "BiasedRandom60Agent":      lambda: BiasedRandomAgent(0.60),
    "BiasedRandom70Agent":      lambda: BiasedRandomAgent(0.70),
    "BlindBaselineAgent":       lambda: BlindBaselineAgent(),
    "ConditionalAgent":         lambda: ConditionalAgent(),
    "ExactRulesConditionalAgent": lambda: ExactRulesConditionalAgent(),
    "FiveKingsBlindAgent":      lambda: FiveKingsBlindAgent(),
    "CFRNashAgent":             _make_cfr_nash,
}


def build_agent(agent_key: str):
    """Instantiate the agent for a given registry key."""
    entry = AGENT_REGISTRY.get(agent_key)
    if entry is None:
        return RandomAgent()
    return _AGENT_CLASS_MAP[entry["class"]]()


class FiveKingsBlindAgent:
    """
    Marginal 50% threshold strategy calibrated for Five-Kings mode (53-card deck).

    Uses P(pool_best >= bid | n, 53-card deck) from five_kings_probs.json.
    Bid index 110 = Five of a Kind Kings is included in the probability table.
    Falls back to BlindBaselineAgent (standard 52-card probabilities) if the
    five-kings cache has not been generated yet.

    See agent/AGENT_CATALOG.md for details.
    """

    def __init__(self) -> None:
        self._blind = BlindBaselineAgent()
        self._rng   = random.Random()

    def choose_action(self, state: MatchState) -> int:
        n = sum(state.hand_sizes[s] for s in state.active_seats())
        rs    = state.round_state
        legal = state.legal_actions()

        p_at_least: Optional[np.ndarray] = None
        if n >= 5:
            try:
                lookup = _get_warm_start()
                p_at_least = lookup.get_five_kings_at_least(n)
            except Exception:
                pass

        if p_at_least is None:
            return self._blind.choose_action(state)

        threshold_idx = 0
        for i in range(len(p_at_least) - 1, -1, -1):
            if p_at_least[i] >= 0.5:
                threshold_idx = i
                break

        if rs.current_bid is None:
            return threshold_idx if threshold_idx in legal else legal[0]

        cur_idx = bid_to_index(rs.current_bid)
        if CALL_ACTION in legal and float(p_at_least[cur_idx]) < 0.5:
            return CALL_ACTION

        bid_candidates = [a for a in legal if a not in (CALL_ACTION, HH_ACTION)]
        if not bid_candidates:
            return CALL_ACTION
        for a in bid_candidates:
            if a >= threshold_idx:
                return a
        return bid_candidates[0]
