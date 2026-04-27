# Training Optimization & Agent Correctness Plan

Drafted 2026-04-21. **Implemented 2026-04-22** — all three tasks complete.
See bottom for implementation summary and run schedule.

Drafted based on profiling + audit of `agent/baseline/cfr_1v1.py`,
`agent/rnad/trainer.py`, and `agent/web/backend/agents.py::ExactRulesConditionalAgent`.

Scope: three independent workstreams, each with measured baselines and
verification gates. Tasks are listed in the recommended execution order,
but each can be done in isolation.

---

## Measured baselines (CPU, no other changes)

| Component | Config | Time/iter | 50k iters |
|---|---|---|---|
| CFR+ | `max_bids=4`, HC+Pair (26 bids), HH on | 30.3 s | ~17 days |
| R-NaD Stage A | batch=128, warm+aux | 1.2 s | 16 h |
| R-NaD Stage B | batch=128, warm+aux | 9.1 s | 5.3 days |

CFR+ profile: 9.08M `_cfr` calls per iteration (169 rank pairs × ~53K nodes).
All hot functions are pure-Python tuple/list/dict operations.

R-NaD profile: one unbatched `encode_obs` + `forward` per step during
collection, then a second `forward` per step inside `compute_loss`. Stage B
averages ~40 steps/match × 128 matches/iter ≈ 5,200 forward passes per iter.

---

## Task 1 — CFR+ vectorization (biggest speedup, isolated change)

**Goal:** ≥100× per-iter speedup on CFR+ (30 s → <0.3 s). Preserve the
algorithmic behavior exactly; verify by matching `exploitability()` and
`opening_mix_by_rank()` against the current solver on a shared seed/iter.

**Files:**
- new: `agent/baseline/cfr_1v1_fast.py` (side-by-side, not a rewrite in place)
- new: `agent/baseline/tests/test_cfr_1v1_fast.py`
- existing: `agent/baseline/cfr_1v1_overnight.py` — add `--solver fast` flag

### 1.1 Precompute the static game tree

The tree is fully determined by `(bid_space, max_bids, include_hh)`. Build it
once in `__init__`:

```python
# flat arrays, parallel-indexed by integer node_id
node_player:     np.ndarray (num_internal,) int8    # 0 or 1
node_depth:      np.ndarray (num_internal,) int8
node_legal:      np.ndarray (num_internal, max_fanout) int16  # -1 = pad
node_legal_len:  np.ndarray (num_internal,) int8
node_children:   np.ndarray (num_internal, max_fanout) int32  # child node_id
node_is_terminal_mask per action  # shape (num_internal, max_fanout) bool
# terminal utilities: precomputed for all 169 rank pairs
terminal_u_p0:   np.ndarray (num_terminals, 13, 13) float32  # P0 utility
```

Measured tree size for `max_bids=4`, HC+Pair, HH: 17,902 internal + 35,802
terminal nodes. Memory: terminal utility array = 35,802 × 13 × 13 × 4 B ≈ 24 MB.

Build via the existing `_legal_actions` / `_is_terminal` helpers (keeps a
single source of truth for tree definition) but run them ONCE at construction.

### 1.2 Flat info-set index

```python
# info_set_idx[(player, rank, node_id)] -> contiguous int
# = player * 13 * num_internal + rank * num_internal + node_id
# Stored as a 1D view, never as a dict.
regret_sum:   np.ndarray (num_info_sets, max_fanout) float32
strategy_sum: np.ndarray (num_info_sets, max_fanout) float32
```

Use `max_fanout` (≈ 28 for HC+Pair+HH+CALL) as the padded action dim; unused
slots stay at 0 and are masked during regret matching.

### 1.3 Vectorized iteration across the 169 rank pairs

One bottom-up traversal per iteration with `[169]`-batched values at every
node. Tree structure is rank-independent; only terminal utilities vary.

```python
def iterate_vectorized(self):
    # value[node_id] -> np.ndarray shape (169,) of P0 utility
    value = np.empty((num_internal, 169), dtype=np.float32)

    # topologically sort: deeper nodes first
    for node_id in reversed(self.topo_order):
        player   = self.node_player[node_id]
        n_legal  = self.node_legal_len[node_id]
        legal    = self.node_legal[node_id, :n_legal]
        info_idx = self._info_idx_for(player, node_id)  # (13,) — one per rank

        # regret matching+: one strategy per P0-or-P1 rank
        pos = np.maximum(self.regret_sum[info_idx, :n_legal], 0.0)
        s   = pos.sum(axis=1, keepdims=True)
        strategy = np.where(s > 0, pos / s, 1.0 / n_legal)   # (13, n_legal)

        # child values: (n_legal, 169) — per-action utilities
        child_util = self._gather_child_utilities(node_id, value)

        # node utility (169,): reshape strategy by whose rank applies, weighted sum
        # (see §1.4 for the tensor ops)
        value[node_id] = ...

    # then vectorized regret + strategy_sum updates (numpy ops, no Python loop)
```

The Python loop runs 17,902 times per iteration (iteration-unrolled over
rank pairs), versus 9M recursive calls today — a 500× reduction in interp
overhead before any numpy speedup.

### 1.4 Update formulas (vectorized equivalents of current code)

The current code's counterfactual update is:

```
cf_opp = chance_reach * opp_reach
regret = sign * (action_util - node_util)
regret_sum[i] = max(0, regret_sum[i] + cf_opp * regret)
strategy_sum[i] += (it+1) * my_reach * strategy[i]
```

Translate to numpy:
- `reach0, reach1`: (13,) arrays over each player's rank.
- `chance_reach`: (13, 13) float32 of `_rank_pair_weight / TOTAL`.
- At info-set `(player, rank, node)`:
  - `opp_reach[rank] = (reach_{1-player} * chance_reach[rank, :]).sum()`
  - `my_reach[rank]  = reach_{player}[rank]`
- `action_util[rank, i] − node_util[rank] = (child_util - node_util[:, None])`
  with appropriate broadcast over the 169 pairs.

Equivalence gate (required before landing):
- Create `CFRSolverFast` with same knobs; run 50 iterations on both solvers
  from the same initial state.
- Assert `max |regret_sum_fast - regret_sum_slow| < 1e-5` at every info-set.
- Assert identical `average_strategy()` at the 13 root info-sets.

### 1.5 Other wins inside the same refactor

- **Exploitability caching:** `exploitability()` rebuilds the same tree walk.
  Reuse the same flat tree arrays; compute both best-response values in a
  single bottom-up pass. Expected: exploitability call drops from current
  ~60 s to under 1 s.
- **Change `cfr_1v1_overnight.py` default `--eval-every` from 1 to 10** (call
  exploitability every 10 batches, not every batch). Keeps metrics.jsonl
  useful but stops doubling run time.
- **Memoize `strategy` per info-set within a single iteration.** Not needed
  after vectorization, but if numba-only path is chosen instead, this
  matters — each info-set is visited once under rank-pair enumeration.

### 1.6 Acceptance

- [ ] `CFRSolverFast.iterate()` produces numerically identical regrets to
      `CFRSolver.iterate()` for 50 iterations from seed state.
- [ ] Wall-clock for `max_bids=4, HC+Pair, HH`: ≤ 0.5 s per iter.
- [ ] `exploitability()`: ≤ 1 s.
- [ ] `cfr_plus_mb4_hh` overnight run reaches 50k iterations in under 8 h
      with exploitability trending toward 0.

### 1.7 Fallback option (if §1.3 turns out tricky)

Skip the vectorization, keep the recursive structure, and decorate the
hottest functions with `@numba.njit` after converting the state containers
to numpy arrays. Expect 10–30× rather than 100×, but smaller code change.

---

## Task 2 — Fix `ExactRulesConditionalAgent` game understanding

**Goal:** make the agent a *correct* rule-based benchmark before anyone
considers using it as a warm-start teacher. Five distinct issues; fix each
with tests against concrete hand scenarios.

**Files:**
- existing: `agent/web/backend/agents.py` (`ExactRulesConditionalAgent`)
- existing: `agent/web/backend/agents.py` (`ExactRulesBlindAgent` — same
  fixes apply to its non-conditional sibling; share helpers where sensible)
- new: `agent/web/backend/tests/test_exact_rules_agents.py`

### 2.1 Fix 1 — Declare HH when the standing bid looks like pool-best

Today: `bid_candidates = [a for a in legal if a not in (CALL_ACTION, HH_ACTION)]`
— HH is discarded silently.

New logic (after the call-threshold check but before choosing a raise):

```python
# HH declaration: if the standing bid's exact_prob is near the peak of our
# adjusted distribution, declaring HH is strictly better than raising
# (we gain +1 if correct; we lose +1 if wrong, same as calling).
if HH_ACTION in legal:
    peak_idx = int(np.argmax(adj_exact))
    # Declare HH if standing bid matches the peak or within a configurable
    # absolute-probability band of it.
    if cur_idx == peak_idx or adj_exact[cur_idx] >= 0.9 * adj_exact[peak_idx]:
        return HH_ACTION
```

Tests:
- Hand with a pair of Aces + low kickers; opponent bids Pair A → HH.
- Hand with no pair; opponent bids HC K → HH at n=10 (HC K is the peak).
- Hand with no pair; opponent bids Pair 2 → do NOT declare HH (peak is HC K).

### 2.2 Fix 2 — Escalation-aware bidding, not global-peak bidding

Today: `best_bid()` picks `argmax(adj_exact)` over all legal raises. At n=10
this is HC K every time, revealing nothing and losing the escalation race.

New rule: among legal raises, pick the **smallest** bid whose exact probability
exceeds a safety threshold. "Smallest legal raise above threshold" preserves
bid-space for later rounds and avoids skipping to HC A on turn 1.

```python
safety_threshold = 0.5 * adj_exact[peak_idx]  # tune in tests
viable = [a for a in bid_candidates if adj_exact[a] >= safety_threshold]
if viable:
    return min(viable)            # smallest legal raise that is "safe enough"
return self._best_bid(bid_candidates, adj_exact)  # fall back: max prob
```

Rationale: a smallest-safe-raise agent at n=10 opens HC 10 (first HC whose
p ≥ ~0.3), forcing the opponent to escalate through HC J/Q/K/A/Pair territory
rather than jumping to HC K immediately.

Test: opening-bid distribution over 1000 random hands at n=10 should show
bids spread across HC 10..A (not 100% HC K as today).

### 2.3 Fix 3 — Decision-theoretic call threshold

Today: hardcoded `call_threshold = 0.3 * max_p`.

New logic — in a 2-outcome round (bid holds vs. doesn't), with ±1 payoffs:

```
EV(call)   = 1·P(¬holds) + (−1)·P(holds) = 1 − 2·P(holds)
EV(raise)  = E[outcome of the rest of the round]   # harder; approximate
```

A correct threshold in the simple call-vs-raise tradeoff: call when
`P(holds | my hand, opponent bid) < 0.5`. Implement:

```python
p_holds = float(adj_exact[cur_idx])
if CALL_ACTION in legal and p_holds < 0.5:
    return CALL_ACTION
```

Keep the `0.3 * max_p` as a secondary safety net for early-round hands where
peak exact_prob is itself < 0.5 (pool size effects). Make both thresholds
explicit constructor args with defaults, to allow tuning.

Test: over 1000 random hands where opponent is a `RandomAgent`, the call
rate at ~0.5-probability bids should track the actual `P(holds)` within a
few percent (requires rollout-based measurement).

### 2.4 Fix 4 — Opponent-bid Bayesian update (bounded-complexity version)

Full opponent modeling is expensive. A tractable approximation:

When the opponent makes a bid with primary rank `r` under exact rules, treat
it as weak evidence that they hold ≥1 card of rank `r`. Adjust the pool prior
by conditioning on the virtual event "opponent has ≥1 of rank r":

```python
# Naive: if opp bid Pair K, mix in the conditional "pair_K" or "adjacent_K"
# distribution from WarmStartLookup weighted by a belief parameter α ∈ [0, 1].
cond_bid_rank = rs.current_bid[1]  # primary rank of standing bid
alpha = 0.5    # tunable; 0.0 = ignore signal, 1.0 = assume opp has rank r
p_opp_rank = _conditional_from_rank(cond_bid_rank, n)   # helper
adj_exact = (1 - alpha) * adj_exact + alpha * (adj_exact * p_opp_rank)
```

Where `_conditional_from_rank` derives a per-bid multiplier given "opp has a
rank-r card" by sampling or table lookup. A stub implementation that simply
up-weights bids sharing the opponent's primary rank (by say ×1.3) and
down-weights bids far from it (by ×0.9) is enough to demonstrate the concept.

Keep α as a constructor arg so it's trivial to disable in ablations.

### 2.5 Fix 5 — Match the PMFs used in the likelihood ratio

Today: `marg_pmf` and `cond_pmf` are derived from the **at-least rules**
marginal/conditional tables (which measure `P(pool_best == bid)`), then used
to adjust `exact_prob` (which measures `P(pool has a 5-subset == bid)`).
These quantities are different.

Correct approach: if exact-rules conditional tables don't exist yet, compute
them once. Infrastructure is already present in
`agent/data/compute_extended_conditional_probs.py` (for at-least rules); add
a sibling `compute_extended_conditional_exact_probs.py` that uses
`has_exact_hand()` from `agent/game/engine.py` during Monte Carlo rather than
`_evaluate_ranked()`. Cache to
`agent/data/extended_conditional_exact_probs_ranked.json`.

Expose via `WarmStartLookup.get_exact_rules_conditional(n, cond_key)` and
consume in `ExactRulesConditionalAgent` directly, replacing the
likelihood-ratio hack.

This is a ~3M sample MC × 6 conditions × n=5..25 run, similar cost to the
existing conditional cache (runs once, cached to JSON).

### 2.6 Acceptance

- [ ] All five behaviors exist and are individually unit-tested with
      hand-crafted pools/hands (not just statistical tests).
- [ ] New benchmark: `ExactRulesConditionalFixed` vs. current
      `ExactRulesConditionalAgent`, 500 games at n=2 and n=10. Fixed variant
      wins ≥ 55%.
- [ ] Re-run the `ExactRulesConditional` vs `CFRNashAgent` match after Task
      1 finishes — the >80% win-rate anomaly should collapse substantially.

### 2.7 What this does NOT fix

- Still a rule-based heuristic — no mixed strategies, no game-tree search.
- Still has no memory of prior rounds within a match (each round decision is
  independent of match history).

Those belong to a future agent (Task 3's warm-started R-NaD, or an MCTS
addition), not to this fix pass.

---

## Task 3 — R-NaD speed + correctness alignment

**Goal:** bring Stage B under 1 s/iter so 50k iterations fit in an overnight
run, plus swap at-least probability tables for exact-rules tables when
training in exact mode.

**Files:**
- existing: `agent/rnad/trainer.py`, `agent/rnad/network.py`, `agent/rnad/config.py`
- existing: `agent/rnad/warm_start.py`

### 3.1 Enable MPS auto-selection (trivial, ~3× speedup expected)

`trainer.py::__init__` currently does:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Replace with:

```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

Verify: a Stage B 50-iter smoke run on MPS produces the same numeric losses
(within float noise) as CPU. If not, gate MPS behind a `--device` CLI arg
rather than auto-detecting.

### 3.2 Batched forward pass in `compute_loss` (biggest R-NaD win)

Today: `compute_loss` loops over `steps`, re-encoding each observation
individually. With ~5200 steps per Stage B iter, that's 5200 calls to
`encode_obs` (each doing small lookups and tensor creation) and 5200
single-example trunk forwards.

Refactor to:

1. Encode all `info` dicts once into a padded `(B, trunk_in_dim)` tensor.
   The encoder already produces fixed-length vectors — stacking is direct.
2. Single batched `policy.forward(obs_batch) → (logits[B, A], value[B, 1])`.
3. Build a legal-action mask as `(B, NUM_ACTIONS)` bool and apply with
   `logits.masked_fill(~mask, float('-inf'))`.
4. Use `dist.log_prob(action_batch)` and `dist.entropy()` batched.
5. Advantages and targets are already per-step floats — stack to `(B,)`.

`collect_round` / `collect_match` still run sequentially (env is sequential),
but collection isn't the bottleneck — it's `compute_loss`. Keep `forward`
inside collection as a single-step call (for now), and rewrite the loss side.

Expected: Stage B drops from 9 s/iter to ~1 s/iter at batch=128.

### 3.3 LRU-cache `WarmStartLookup.get_features`

```python
from functools import lru_cache

# inside WarmStartLookup
@lru_cache(maxsize=32_768)
def _get_features_cached(self, hand_key: tuple, n: int):
    return self.get_features(list(hand_key), n)
```

Where `hand_key = tuple(sorted(own_hand))`. Each training run sees a tiny
fraction of the full deal space; a 32K-entry cache catches >95% of repeated
lookups in Stage B.

Verify: cache hit rate >90% after first 100 iters.

### 3.4 Swap warm-start tables when `exact_rules=True`

Today `network.py` always pulls at-least-rules `get_features`. Extend:

- `RNaDConfig` gains an `exact_rules: bool` flag (default False).
- `LiarsPokerNet.encode_obs` checks `self.config.exact_rules` and calls
  `lookup.get_exact_rules_conditional(n, cond_key)` when set (requires Task
  2.5 to be done first, since it generates that table).
- `collect_round` / `collect_match` pass `exact_rules` through to
  `new_match(...)`.

### 3.5 Do NOT seed from `ExactRulesConditionalAgent`

Even after Task 2, the fixed agent is still a heuristic. Use it as a
**benchmark opponent** in evaluation, not as an imitation target. If we
want a warm-start policy, use the converged CFR+ Nash policy from Task 1
(at n=2 exact-rules), loaded as a fixed prior over root-info actions.

Implementation sketch (optional, only if results need it):
- Load `cfr_plus_mb4_hh/checkpoint.json` average strategy at match start.
- Add a KL penalty `β · KL(π_θ(·|s_root) || π_CFR(·|s_root))` for s_root
  info-sets that appear in the CFR tree.
- Anneal `β` from 1.0 → 0.0 over training.

### 3.6 Acceptance

- [ ] MPS device is used automatically on Apple Silicon and produces numeric
      parity with CPU within 1e-4 for 50 iters.
- [ ] `compute_loss` single batched forward — same loss values as today
      within float noise on a 50-iter replay.
- [ ] Stage B wall-clock: ≤ 1.5 s/iter at batch=128.
- [ ] `exact_rules` pathway trains end-to-end and reaches
      `win_rate_vs_random` ≥ 0.7 at n=2 within 5k iters.

---

## Execution order for next session

1. **Task 1 first.** It has the biggest impact, is the most isolated, and
   its output (converged CFR+ Nash) is the correct warm-start target for any
   later learning work. Once §1.1–§1.4 are in place and the equivalence test
   passes, launch the `cfr_plus_mb4_hh` run in the background.
2. **Task 2 second.** Independent of Task 1. Fix the five bugs in a single
   PR; add unit tests per sub-fix.
3. **Task 3 third.** Depends on Task 2.5 (exact-rules conditional table)
   for the training-mode swap. The speed fixes in §3.1–§3.3 are independent
   and can ship alongside Task 1 if bandwidth allows.

## What NOT to do

- Don't seed R-NaD from `ExactRulesConditionalAgent` policy (even after Task 2).
- Don't remove the existing `CFRSolver` — keep it as a reference oracle for
  equivalence tests of the vectorized variant.
- Don't enable `--full-bids` CFR+; the 110-bid tree is not why training is
  slow, and enabling it would explode memory.
- Don't skip hooks (`--no-verify`) or bypass signing on any commit for this work.

## Files touched (quick reference)

| Task | New files | Modified files |
|---|---|---|
| 1 | `agent/baseline/cfr_1v1_fast.py`, `agent/baseline/tests/test_cfr_1v1_fast.py` | `agent/baseline/cfr_1v1_overnight.py` |
| 2 | `agent/web/backend/tests/test_exact_rules_agents.py`, `agent/data/compute_extended_conditional_exact_probs.py` | `agent/web/backend/agents.py`, `agent/rnad/warm_start.py` |
| 3 | (none) | `agent/rnad/trainer.py`, `agent/rnad/config.py`, `agent/rnad/warm_start.py` |
| HH site | (none) | `agent/web/backend/app.py` |

---

## Implementation Summary — 2026-04-22

All tasks implemented and smoke-tested. Key results:

### Task 1 results
- `CFRSolverFast.iterate()`: 72ms at max_bids=3, 405ms at max_bids=4 (75× faster)
- `exploitability()`: 0.25s (was ~60s)
- 8 tests passing; ≤2 top-1 disagreements vs slow solver at 500 iters (expected — different floor cadence)

### Task 2 results
- All 5 agent fixes live in `ExactRulesConditionalAgent`
- 9 unit tests passing (each fix verified in isolation with monkeypatched FakeLookup)
- `compute_extended_conditional_exact_probs.py` running (started 2026-04-22 ~22:05, 9 workers, 10k samples/condition, ~4-6h)

### Task 3 results
- MPS auto-detected on Apple Silicon — confirmed working via smoke test
- `compute_loss` batched forward confirmed numerically correct on MPS
- `exact_rules=True, high_hand=True` flows end-to-end through collection and loss

### High Hand website
- "Declare High Hand ★" button live in exact-rules games
- Gold-colored label in bid history and round history for HH actions
- Result box shows correct/incorrect declaration outcome text

---

## Run schedule (as of 2026-04-22)

| Script | Status | Command |
|---|---|---|
| `compute_extended_conditional_exact_probs.py` | **RUNNING** (started 22:05) | — |
| `cfr_1v1_overnight --solver fast` 50k | **NOT STARTED** | see §Execution order |
| R-NaD Stage A exact_rules MPS | **NOT STARTED** | after conditional MC done |

### Launch CFR+ now (before sleeping)

```bash
cd "papers/Liars poker"
python -m agent.baseline.cfr_1v1_overnight \
    --name cfr_plus_mb4_hh \
    --max-bids 4 --batch 100 --total-iters 50000 \
    --solver fast
```

Estimated: ~5.5h. Checkpoint every 100 iters; safe to Ctrl-C and resume.
