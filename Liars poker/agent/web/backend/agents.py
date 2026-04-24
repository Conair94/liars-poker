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

    def _compute_adj_exact(
        self,
        exact_prob: np.ndarray,
        lookup,
        own_hand: list,
        current_bid,
        n: int,
    ) -> np.ndarray:
        """Compute adjusted exact probabilities. Subclasses can override to add
        additional belief updates (e.g. opponent modelling)."""
        adj = self._adjust_for_own_hand(exact_prob, lookup, own_hand, n)
        adj = self._apply_opp_bid_belief(adj, current_bid)
        return adj

    def _call_threshold(self, state: MatchState) -> float:
        """Return the call probability threshold. Subclasses can override for
        adaptive behaviour based on round history."""
        return self.call_prob_threshold

    def _select_bid(
        self,
        candidates: list,
        adj_exact: np.ndarray,
        viable: list,
    ) -> int:
        """Choose a bid from candidates. Override in subclasses for mixed
        strategies; default is deterministic min(viable)."""
        if viable:
            return min(viable)
        return max(candidates, key=lambda a: float(adj_exact[a]) if a < len(adj_exact) else 0.0)

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

        adj_exact = self._compute_adj_exact(exact_prob, lookup, rs.hands[seat], rs.current_bid, n)

        peak_idx = int(np.argmax(adj_exact))
        peak_p   = float(adj_exact[peak_idx])
        bid_candidates = [a for a in legal if a not in (CALL_ACTION, HH_ACTION)]

        # ------------------------------------------------------------------
        # No standing bid yet — opening bid.
        # ------------------------------------------------------------------
        if rs.current_bid is None:
            if not bid_candidates:
                return legal[0]
            safety = self.safety_frac * peak_p
            viable = [a for a in bid_candidates if float(adj_exact[a]) >= safety]
            return self._select_bid(bid_candidates, adj_exact, viable)

        # ------------------------------------------------------------------
        # Responding to a standing bid.
        # ------------------------------------------------------------------
        cur_idx = bid_to_index(rs.current_bid)
        cur_p   = float(adj_exact[cur_idx]) if cur_idx < len(adj_exact) else 0.0

        # Declare HH when the standing bid matches (or near-matches) the peak.
        if HH_ACTION in legal and peak_p > 0.0:
            if cur_idx == peak_idx or cur_p >= self.hh_band * peak_p:
                return HH_ACTION

        # Decision-theoretic call threshold.
        call_threshold = self._call_threshold(state)
        if CALL_ACTION in legal:
            if cur_p < call_threshold or cur_p < self.floor_frac * peak_p:
                return CALL_ACTION

        if not bid_candidates:
            return CALL_ACTION

        # Smallest legal raise above safety threshold.
        safety = self.safety_frac * peak_p
        viable = [a for a in bid_candidates if float(adj_exact[a]) >= safety]
        return self._select_bid(bid_candidates, adj_exact, viable)


# ---------------------------------------------------------------------------
class ExactRulesMixedAgent(ExactRulesConditionalAgent):
    """
    ExactRulesConditionalAgent + softmax bid sampling to prevent information leakage.

    Under ExactRulesConditionalAgent the opening bid is deterministic: the agent
    always picks min(viable), so an opponent who knows the strategy can narrow the
    bidder's card rank from a single bid.  This agent samples from the viable set
    using a softmax weighted by adj_exact probabilities, controlled by
    `bid_temperature` τ.  τ→0 recovers the deterministic baseline; τ≈0.05 spreads
    mass across 2–4 nearby bids without sacrificing much EV.

    All other logic (HH declaration, call threshold, Bayesian rank bump) is
    inherited from ExactRulesConditionalAgent.

    Data: same as ExactRulesConditionalAgent.
    Mode: Exact Hand Rules + High Hand.
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
        bid_temperature: float = 0.05,
    ) -> None:
        super().__init__(
            hh_band=hh_band,
            safety_frac=safety_frac,
            call_prob_threshold=call_prob_threshold,
            floor_frac=floor_frac,
            opp_bid_alpha=opp_bid_alpha,
            opp_bid_up_mult=opp_bid_up_mult,
            opp_bid_down_mult=opp_bid_down_mult,
        )
        self.bid_temperature = bid_temperature

    def _softmax_sample(self, candidates: list, weights: np.ndarray) -> int:
        """Sample from candidates proportional to softmax(weights / τ)."""
        if len(candidates) == 1:
            return candidates[0]
        w = np.array(
            [float(weights[a]) if a < len(weights) else 0.0 for a in candidates],
            dtype=np.float64,
        )
        w = np.clip(w, 0.0, None)
        if self.bid_temperature <= 0.0 or w.sum() <= 0.0:
            return candidates[int(np.argmax(w))]
        w = np.exp(w / self.bid_temperature)
        w /= w.sum()
        return candidates[int(np.random.choice(len(candidates), p=w))]

    def _select_bid(self, candidates: list, adj_exact: np.ndarray, viable: list) -> int:
        pool = viable if viable else candidates
        return self._softmax_sample(pool, adj_exact)


# ---------------------------------------------------------------------------
class ExactRulesOpponentModelAgent(ExactRulesMixedAgent):
    """
    ExactRulesMixedAgent + principled self-model Bayesian update on opponent bids.

    When conditional tables exist for the current pool size (n ≥ 5), replaces
    the heuristic rank-bump with a principled posterior over opponent hand conditions:

        P(opp_cond | opp_bid = B) ∝ cond_table[opp_cond][n][B]

    The pool distribution is updated as a linear blend:

        adj_final = (1 − α) × our_adj + α × posterior_weighted_opp_dist

    When no conditional tables exist (n < 5 — all conditions require ≥ 2 cards
    but at n=2 each player holds only 1 card), falls back to the inherited
    heuristic rank-bump so the Bayesian path never regresses below rung 2.

    Data: same as ExactRulesConditionalAgent (requires extended_conditional_exact_probs.json).
    Mode: Exact Hand Rules + High Hand.
    """

    def __init__(
        self,
        hh_band: float = 0.9,
        safety_frac: float = 0.5,
        call_prob_threshold: float = 0.5,
        floor_frac: float = 0.3,
        bid_temperature: float = 0.05,
        opp_model_alpha: float = 0.25,
        # Heuristic params kept for fallback use at n < 5
        opp_bid_alpha: float = 0.5,
        opp_bid_up_mult: float = 1.3,
        opp_bid_down_mult: float = 0.9,
    ) -> None:
        super().__init__(
            hh_band=hh_band,
            safety_frac=safety_frac,
            call_prob_threshold=call_prob_threshold,
            floor_frac=floor_frac,
            opp_bid_alpha=opp_bid_alpha,
            opp_bid_up_mult=opp_bid_up_mult,
            opp_bid_down_mult=opp_bid_down_mult,
            bid_temperature=bid_temperature,
        )
        self.opp_model_alpha = opp_model_alpha

    def _opponent_model_update(
        self,
        our_adj: np.ndarray,
        lookup,
        current_bid,
        n: int,
    ) -> np.ndarray:
        """Bayesian blend using conditional tables. Returns our_adj (same object)
        when no tables are available so callers can detect a no-op via identity."""
        if self.opp_model_alpha <= 0.0 or current_bid is None:
            return our_adj

        get_exact_cond = getattr(lookup, "get_exact_rules_conditional", None)
        if get_exact_cond is None:
            return our_adj

        bid_idx = bid_to_index(current_bid)
        known_conds = list(getattr(lookup, "known_conditions", set()))
        if not known_conds:
            return our_adj

        weights = []
        cond_dists = []
        for cond_key in known_conds:
            dist = get_exact_cond(n, cond_key)
            if dist is None or len(dist) <= bid_idx:
                continue
            likelihood = float(dist[bid_idx])
            if likelihood <= 0.0:
                continue
            weights.append(likelihood)
            cond_dists.append(dist)

        if not weights:
            return our_adj  # same object — signals "no update"

        weights_arr = np.array(weights, dtype=np.float64)
        weights_arr /= weights_arr.sum()

        blended = np.zeros(len(our_adj), dtype=np.float32)
        for w, dist in zip(weights_arr, cond_dists):
            length = min(len(dist), len(blended))
            blended[:length] += float(w) * dist[:length]

        total = blended.sum()
        if total > 0:
            blended /= total

        a = self.opp_model_alpha
        return ((1.0 - a) * our_adj + a * blended).astype(np.float32)

    def _compute_adj_exact(
        self,
        exact_prob: np.ndarray,
        lookup,
        own_hand: list,
        current_bid,
        n: int,
    ) -> np.ndarray:
        adj = self._adjust_for_own_hand(exact_prob, lookup, own_hand, n)
        # Try principled Bayesian update; fall back to heuristic when no tables.
        bayesian = self._opponent_model_update(adj, lookup, current_bid, n)
        if bayesian is adj:
            # No conditional tables for this n — use inherited heuristic instead.
            adj = self._apply_opp_bid_belief(adj, current_bid)
        else:
            adj = bayesian
        return adj


# ---------------------------------------------------------------------------
class ExactRulesAdaptiveAgent(ExactRulesOpponentModelAgent):
    """
    ExactRulesOpponentModelAgent + adaptive call threshold.

    Tracks the opponent's observed bluff rate from MatchState.round_history and
    adjusts the call probability threshold accordingly:

      - High bluff rate  → lower threshold  (call challenges more readily)
      - Low bluff rate   → higher threshold (accept more bids as honest)

    The adjustment uses `adaptive_lr` to scale the shift around the base
    call_prob_threshold, clamped to [adaptive_min, adaptive_max].  Requires at
    least `min_history` resolved rounds before adapting; defaults to base
    threshold with insufficient data.

    Data: same as ExactRulesOpponentModelAgent.
    Mode: Exact Hand Rules + High Hand.
    """

    def __init__(
        self,
        hh_band: float = 0.9,
        safety_frac: float = 0.5,
        call_prob_threshold: float = 0.5,
        floor_frac: float = 0.3,
        bid_temperature: float = 0.05,
        opp_model_alpha: float = 0.4,
        adaptive_lr: float = 0.4,
        adaptive_min: float = 0.25,
        adaptive_max: float = 0.75,
        min_history: int = 3,
    ) -> None:
        super().__init__(
            hh_band=hh_band,
            safety_frac=safety_frac,
            call_prob_threshold=call_prob_threshold,
            floor_frac=floor_frac,
            bid_temperature=bid_temperature,
            opp_model_alpha=opp_model_alpha,
        )
        self.adaptive_lr  = adaptive_lr
        self.adaptive_min = adaptive_min
        self.adaptive_max = adaptive_max
        self.min_history  = min_history

    def _call_threshold(self, state: MatchState) -> float:
        """Adjust call threshold based on observed opponent bluff rate."""
        rs   = state.round_state
        seat = rs.current_player
        history = state.round_history

        # Collect rounds where we (seat) called the opponent's bid.
        bluffs = 0
        honest = 0
        for rr in history:
            if rr.is_high_hand:
                continue
            if rr.caller_seat == seat:
                # We called; check whether it was a bluff.
                if rr.call_succeeded:
                    bluffs += 1
                else:
                    honest += 1

        total = bluffs + honest
        if total < self.min_history:
            return self.call_prob_threshold

        bluff_rate = bluffs / total
        # Shift threshold down when opponent bluffs often, up when they're honest.
        # bluff_rate = 0.5 → no adjustment; >0.5 → lower threshold; <0.5 → raise it.
        adjustment = (bluff_rate - 0.5) * self.adaptive_lr
        adapted = self.call_prob_threshold - adjustment
        return max(self.adaptive_min, min(self.adaptive_max, adapted))


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
        "description":  "Peak-probability strategy for exact-rules mode, with Bayesian private-hand adjustment. Deterministic opener — baseline of the ladder.",
        "rules": {
            "exact_rules": True,
            "high_hand":   True,
            "five_kings":  False,
        },
        "rules_label":  "Exact rules + High Hand declaration, 52-card deck",
        "class":        "ExactRulesConditionalAgent",
    },
    "exact_mixed": {
        "display":      "Exact Rules Mixed Strategy",
        "description":  "Adds softmax bid sampling (τ=0.05) to the Conditional agent. Prevents rank leakage: the opener no longer deterministically reveals card rank. Ladder rung 2.",
        "rules": {
            "exact_rules": True,
            "high_hand":   True,
            "five_kings":  False,
        },
        "rules_label":  "Exact rules + High Hand declaration, 52-card deck",
        "class":        "ExactRulesMixedAgent",
    },
    "exact_opp_model": {
        "display":      "Exact Rules Opponent Model",
        "description":  "Adds principled Bayesian inference over opponent hand condition given their bid, replacing the heuristic rank-bump. Uses exact conditional tables to compute P(opp_cond | opp_bid). Ladder rung 3.",
        "rules": {
            "exact_rules": True,
            "high_hand":   True,
            "five_kings":  False,
        },
        "rules_label":  "Exact rules + High Hand declaration, 52-card deck",
        "class":        "ExactRulesOpponentModelAgent",
    },
    "exact_adaptive": {
        "display":      "Exact Rules Adaptive",
        "description":  "Full ladder: mixed bidding + opponent modelling + adaptive call threshold. Tracks opponent bluff rate across rounds and shifts call threshold accordingly. Ladder rung 4.",
        "rules": {
            "exact_rules": True,
            "high_hand":   True,
            "five_kings":  False,
        },
        "rules_label":  "Exact rules + High Hand declaration, 52-card deck",
        "class":        "ExactRulesAdaptiveAgent",
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
    "ExactRulesConditionalAgent":  lambda: ExactRulesConditionalAgent(),
    "ExactRulesMixedAgent":        lambda: ExactRulesMixedAgent(),
    "ExactRulesOpponentModelAgent": lambda: ExactRulesOpponentModelAgent(),
    "ExactRulesAdaptiveAgent":     lambda: ExactRulesAdaptiveAgent(),
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
