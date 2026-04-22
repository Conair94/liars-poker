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
from agent.game.bids import CALL_ACTION, HH_ACTION, NUM_BIDS, bid_to_index, all_bids   # noqa: E402
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
    Conditional strategy for Exact Hand Rules mode with game-theoretic fixes:

      1. Declare High Hand when the standing bid matches (or near-matches) the
         peak of the adjusted distribution — HH and CALL have symmetric ±1
         payoffs, and HH beats a raise when the standing bid is the most
         likely hand.
      2. Escalation-aware bidding: among legal raises, pick the SMALLEST bid
         whose exact probability clears a safety threshold (preserves bid
         space; avoids revealing info by always jumping to the global peak).
      3. Decision-theoretic call threshold: call when P(holds | hand) < 0.5,
         which is the zero-EV crossing of call-vs-raise in the ±1 payoff model.
         A secondary floor is kept for very early rounds where the peak is < 0.5.
      4. Light Bayesian update on opponent's bid: α-weighted multiplier that
         up-weights bids sharing the opponent's primary rank.
      5. Attempts to use exact-rules conditional tables when available
         (`WarmStartLookup.get_exact_rules_conditional(n, cond_key)`); falls
         back to the prior likelihood-ratio correction otherwise.

    Falls back to ExactRulesBlindAgent if the cache is missing.
    """

    def __init__(
        self,
        hh_band: float = 0.9,
        safety_frac: float = 0.5,
        call_prob_threshold: float = 0.5,
        floor_frac: float = 0.3,
        opp_bid_alpha: float = 0.5,
        opp_bid_up_mult: float = 1.3,
        opp_bid_down_mult: float = 0.9,
    ) -> None:
        self._blind               = ExactRulesBlindAgent()
        self.hh_band              = hh_band
        self.safety_frac          = safety_frac
        self.call_prob_threshold  = call_prob_threshold
        self.floor_frac           = floor_frac
        self.opp_bid_alpha        = opp_bid_alpha
        self.opp_bid_up_mult      = opp_bid_up_mult
        self.opp_bid_down_mult    = opp_bid_down_mult

    @staticmethod
    def _pmf_from(pal: np.ndarray) -> np.ndarray:
        pmf = np.zeros_like(pal)
        pmf[:-1] = np.maximum(0, pal[:-1] - pal[1:])
        pmf[-1]  = max(0.0, float(pal[-1]))
        return pmf

    def _adjust_for_own_hand(
        self,
        exact_prob: np.ndarray,
        lookup,
        own_hand: list,
        n: int,
    ) -> np.ndarray:
        """Apply hand-conditional adjustment. Prefer exact-rules conditional
        tables when generated; otherwise fall back to a likelihood-ratio on
        the at-least conditional tables."""
        # Exact-rules conditional table (Task 2.5): preferred when present.
        get_exact_cond = getattr(lookup, "get_exact_rules_conditional", None)
        if get_exact_cond is not None:
            try:
                _, _, cond_key = lookup.get_features(own_hand, n)
                if cond_key is not None:
                    ec = get_exact_cond(n, cond_key)
                    if ec is not None:
                        return ec.astype(np.float32)
            except Exception:
                pass

        # Fallback: likelihood-ratio via at-least tables.
        try:
            marginal, cond_vec, _ = lookup.get_features(own_hand, n)
            marg_pmf = self._pmf_from(
                np.flip(np.cumsum(np.flip(marginal))).astype(np.float32)
            )
            cond_pmf = self._pmf_from(
                np.flip(np.cumsum(np.flip(cond_vec))).astype(np.float32)
            )
            mask  = marg_pmf > 1e-9
            ratio = np.where(
                mask,
                np.minimum(cond_pmf / np.where(mask, marg_pmf, 1.0), 10.0),
                1.0,
            )
            return (exact_prob * ratio).astype(np.float32)
        except Exception:
            return exact_prob.copy()

    def _apply_opp_bid_belief(
        self,
        adj_exact: np.ndarray,
        current_bid,
    ) -> np.ndarray:
        """Weak-evidence Bayesian bump: given opponent's standing bid of primary
        rank r, up-weight bids sharing r and down-weight bids far from it."""
        if self.opp_bid_alpha <= 0.0 or current_bid is None:
            return adj_exact
        r_opp = current_bid.primary_rank
        mult = np.ones_like(adj_exact)
        for i, bid in enumerate(all_bids()):
            if i >= len(adj_exact):
                break
            if bid.primary_rank == r_opp:
                mult[i] = self.opp_bid_up_mult
            elif abs(bid.primary_rank - r_opp) >= 4:
                mult[i] = self.opp_bid_down_mult
        a = self.opp_bid_alpha
        return ((1.0 - a) * adj_exact + a * (adj_exact * mult)).astype(np.float32)

    def choose_action(self, state: MatchState) -> int:
        n     = sum(state.hand_sizes[s] for s in state.active_seats())
        rs    = state.round_state
        seat  = rs.current_player
        legal = state.legal_actions()

        if n < 5 or n > 25:
            return self._blind.choose_action(state)

        try:
            lookup     = _get_warm_start()
            exact_prob = lookup.get_exact_rules_exact(n)
        except Exception:
            exact_prob = None
        if exact_prob is None:
            return self._blind.choose_action(state)

        adj_exact = self._adjust_for_own_hand(exact_prob, lookup, rs.hands[seat], n)
        adj_exact = self._apply_opp_bid_belief(adj_exact, rs.current_bid)

        peak_idx = int(np.argmax(adj_exact))
        peak_p   = float(adj_exact[peak_idx])
        bid_candidates = [a for a in legal if a not in (CALL_ACTION, HH_ACTION)]

        # ------------------------------------------------------------------
        # No standing bid yet — opening bid (fix 2.2).
        # ------------------------------------------------------------------
        if rs.current_bid is None:
            if not bid_candidates:
                return legal[0]
            safety = self.safety_frac * peak_p
            viable = [a for a in bid_candidates if float(adj_exact[a]) >= safety]
            if viable:
                return min(viable)       # smallest safe-enough raise
            return max(bid_candidates,
                       key=lambda a: float(adj_exact[a]) if a < len(adj_exact) else 0.0)

        # ------------------------------------------------------------------
        # Responding to a standing bid.
        # ------------------------------------------------------------------
        cur_idx = bid_to_index(rs.current_bid)
        cur_p   = float(adj_exact[cur_idx]) if cur_idx < len(adj_exact) else 0.0

        # Fix 2.1: Declare HH when the standing bid matches (or near-matches)
        # the peak of our adjusted distribution.
        if HH_ACTION in legal and peak_p > 0.0:
            if cur_idx == peak_idx or cur_p >= self.hh_band * peak_p:
                return HH_ACTION

        # Fix 2.3: decision-theoretic call threshold.
        if CALL_ACTION in legal:
            call_by_prob  = cur_p < self.call_prob_threshold
            call_by_floor = cur_p < self.floor_frac * peak_p
            if call_by_prob or call_by_floor:
                return CALL_ACTION

        if not bid_candidates:
            return CALL_ACTION

        # Fix 2.2: smallest legal raise above safety threshold.
        safety = self.safety_frac * peak_p
        viable = [a for a in bid_candidates if float(adj_exact[a]) >= safety]
        if viable:
            return min(viable)
        return max(bid_candidates,
                   key=lambda a: float(adj_exact[a]) if a < len(adj_exact) else 0.0)


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
        "display":      "Random Biased (30% bid)",
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
        "display":      "Random Biased (40% bid)",
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
        "display":      "Random Uniform",
        "description":  "Picks any legal action uniformly at random. Useful as a baseline.",
        "rules": {
            "exact_rules": False,
            "high_hand":   False,
            "five_kings":  False,
        },
        "rules_label":  "Standard — at-least rules, 52-card deck",
        "class":        "RandomAgent",
    },
    "biased60": {
        "display":      "Random Biased (60% bid)",
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
        "display":      "Random Biased (70% bid)",
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
        "display":      "Blind Threshold (50%)",
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
        "display":      "Conditional Threshold",
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
        "display":      "Exact Rules Conditional",
        "description":  "Peak-probability strategy for exact-rules mode, with Bayesian private-hand adjustment.",
        "rules": {
            "exact_rules": True,
            "high_hand":   True,
            "five_kings":  False,
        },
        "rules_label":  "Exact rules + High Hand declaration, 52-card deck",
        "class":        "ExactRulesConditionalAgent",
    },
    "cfr_nash_mb3": {
        "display":      "CFR Nash (n=2, 20k iters)",
        "description":  "Approximate Nash equilibrium strategy computed by 20,000 iterations of Counterfactual Regret Minimization. Bids High Card / Pair only; uses mixed strategies to prevent rank inference. Single-round win rate: +7–20% vs baselines on exact-rules n=2.",
        "rules": {
            "exact_rules": True,
            "high_hand":   True,
            "five_kings":  False,
        },
        "rules_label":  "Exact rules + High Hand, 52-card deck (trained domain: n=2, max 3 bids)",
        "class":        "CFRNashAgent",
    },
    "exact_random": {
        "display":      "Random Uniform (exact)",
        "description":  "Picks any legal action uniformly at random under exact-rules. Weakest baseline.",
        "rules": {
            "exact_rules": True,
            "high_hand":   True,
            "five_kings":  False,
        },
        "rules_label":  "Exact rules + High Hand, 52-card deck",
        "class":        "RandomAgent",
    },
    "exact_biased30": {
        "display":      "Random Biased 30% (exact)",
        "description":  "Bids 30% of the time in exact-rules mode; calls bluff 70%. Benchmark agent.",
        "rules": {
            "exact_rules": True,
            "high_hand":   True,
            "five_kings":  False,
        },
        "rules_label":  "Exact rules + High Hand, 52-card deck",
        "class":        "BiasedRandom30Agent",
    },
    "exact_biased40": {
        "display":      "Random Biased 40% (exact)",
        "description":  "Bids 40% of the time in exact-rules mode; calls bluff 60%. Benchmark agent.",
        "rules": {
            "exact_rules": True,
            "high_hand":   True,
            "five_kings":  False,
        },
        "rules_label":  "Exact rules + High Hand, 52-card deck",
        "class":        "BiasedRandom40Agent",
    },
    "exact_biased60": {
        "display":      "Random Biased 60% (exact)",
        "description":  "Bids 60% of the time in exact-rules mode; calls bluff 40%. Benchmark agent.",
        "rules": {
            "exact_rules": True,
            "high_hand":   True,
            "five_kings":  False,
        },
        "rules_label":  "Exact rules + High Hand, 52-card deck",
        "class":        "BiasedRandom60Agent",
    },
    "exact_biased70": {
        "display":      "Random Biased 70% (exact)",
        "description":  "Bids 70% of the time in exact-rules mode; calls bluff 30%. Benchmark agent.",
        "rules": {
            "exact_rules": True,
            "high_hand":   True,
            "five_kings":  False,
        },
        "rules_label":  "Exact rules + High Hand, 52-card deck",
        "class":        "BiasedRandom70Agent",
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
