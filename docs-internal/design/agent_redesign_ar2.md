# AR-2 Sub-design — CallPolicy and BidPolicy via CFR+ distillation

- **Status:** Draft (design-first gate; implementation lands in a follow-up session)
- **Date:** 2026-05-01
- **Owner:** main
- **Parent plan:** [agent_redesign_plan.md](agent_redesign_plan.md) §AR-2
- **Parent design:** [agent_redesign.md](agent_redesign.md) §4.2, §4.3, §4.4, §6 Phase B, §7.8
- **Predecessors:** AR-0a/0b at `fb17d03`; AR-1 implementation + 12-cell sweep complete (2026-04-29 / 2026-04-30); HandModel winner pinned at `b64-h256-n2`.

This sub-design fixes the AR-2 open questions the plan flagged: deal-sample budget, `hh_fired` short-circuit during CFR+, entropy floor schedule for `n ≤ 4`, distillation loss, and tie-break for the 4-way mixing fingerprint. It also resolves the trunk-sharing question deferred from AR-1.

## Goals

- Produce two trained checkpoints — `DistilledCallPolicy` and `DistilledBidPolicy` — that, composed with the frozen AR-1 `LearnedHandModel`, beat `ExactRulesConditional` 1v1 by ≥ 5 pp head-to-head and have lower sampled-subgame exploitability.
- Reuse the existing CFR+ solver in [src/training/metrics/subgame_exploitability.py](../../src/training/metrics/subgame_exploitability.py) by refactoring the bidding-tree traversal into a standalone `CFRPlusSubgameSolver`. Do not write a second solver.
- Decide trunk-sharing: deferred from AR-1 §9. Resolved here as **shared trunk loaded from AR-1, two small heads** (parent §7.8).
- Treat HH as a free-standing deterministic gate (parent §4.4) shared between `ModularNashAgent` and reflect rules. Heads are conditioned on `hh_fired=False`.

## Non-goals

- R-NaD fine-tuning (AR-3).
- Adapting CFR+ to multi-round or 5p (AR-5). Distillation is single-round 1v1.
- Replacing AR-1's `LearnedHandModel`. HandModel is **frozen** during all of AR-2 — its weights do not appear in the optimizer's parameter list.
- A new auxiliary HandModel loss; that lives in AR-3 §6 Phase C.

---

## 1. The CFR+ subgame, precisely

For a sampled deal `D = (h0, h1)` with `n = |h0| + |h1|` and HandModel `q̂` frozen, the bidding subgame is:

- Two seats, `P0` always opens.
- Action space at each node: `legal_subgame_actions(cur_bid_idx)` = `{i : i > cur_bid_idx}` ∪ `{CALL_ACTION}` ∪ `{HH_ACTION when standing bid is the pool's best}`. Already implemented in `subgame_exploitability._legal_subgame_actions`.
- Terminal payoffs: ±1 from `_resolve_call_returns` and `_resolve_hh_returns` (already implemented).
- HandModel does **not** appear in the solver itself — CFR+ runs on the *true* deal, with the true pool, exactly as `subgame_exploitability` does today. HandModel's role in AR-2 is only as the input feature for the heads, not as a chance prior at solve time.

This is a small game: at `n=10` the bidding tree is at most a few thousand nodes (110 bid actions, but most are pruned by `cur_bid_idx > prev`). CFR+ to ε < 1e-3 takes well under a second per deal on CPU. Profile during the pilot run.

### 1.1 Why not "HandModel-induced beliefs as the chance prior"?

The plan checklist phrases it that way, but the right semantic is the one the existing solver already uses: **solve the perfect-information bidding subgame on the true deal**, then **distill the resulting average policy** into networks whose *only* hand-side input is the AR-1 `HandModel`'s posterior. Imperfect information re-enters at distillation time, not at solve time. This matches how `kuhn_cfr_plus_oracle` works and avoids the bias of solving a wrong game.

## 2. Refactor of `subgame_exploitability.py`

Extract a reusable solver:

```python
# src/training/cfr/subgame_solver.py
class CFRPlusSubgameSolver:
    def __init__(self, *, max_iters: int, eps: float, seed: int): ...
    def solve(
        self, hands: tuple[list[int], list[int]],
    ) -> SubgameSolution: ...

@dataclass(frozen=True)
class SubgameSolution:
    # Per-infostate average strategy at convergence.
    # Keys are (cur_bid_idx | None, current_player).
    avg_call_prob: dict[tuple[int | None, int], float]      # P(call)
    avg_hh_prob:   dict[tuple[int | None, int], float]      # P(HH)  -- 0 unless gate fires
    avg_bid_dist:  dict[tuple[int | None, int], np.ndarray] # shape (NUM_BIDS,)
    iters_used:    int
    final_eps:     float
```

The existing `_agent_value` / `_br_value` walk is rewritten in terms of this solver; `subgame_exploitability` becomes a thin caller. Tests for the existing exploitability metric must continue to pass byte-identically (within solver tolerance).

This is the only piece of AR-2 that touches existing code; everything else lives under `src/agents/learned/{callpolicy,bidpolicy}/` and `src/training/cfr_distillation.py`.

## 3. Deal-sample budget

### 3.1 Hand-size mixture

25% each of `n = 4, 6, 8, 10` (parent §5.1; matches AR-1). Total **deal count `N`** is the sweep axis; each deal contributes both `(P0, P1)` views — i.e. `2N` distillation rows of `(infostate, target_dist)` for BidPolicy and `2N` rows for CallPolicy (one per non-terminal infostate visited along the average-strategy support).

Per-deal expansion: each subgame-average-strategy walk contributes ~5–15 reachable infostates per seat (count visited under support of average strategy, weighted by reach probability). At `N = 10k` deals this gives ~150k–600k distillation rows per head — comparable to AR-1's 1M-row Phase-A dataset.

### 3.2 Pilot and sweep

**Pilot (one-shot, before the sweep):** `N = 1000`. Wall-clock budget ≤ 30 min CPU-single. Output: per-deal solver iters + final ε histogram. Goes/no-goes the budget for the full sweep.

**Sweep `configs/sweeps/ar2_distillation_count.yaml`:** one axis, `N ∈ {1k, 5k, 10k, 50k}`, 4 cells. Held-out validation set: a fixed 2k deals not in any train split; we plot `KL(distilled || cfr_plus_avg)` per `n` against `N` and pick the elbow. Acceptance: the curve must visibly plateau (slope < 5% per doubling of `N`) by the time the chosen `N` is reached, otherwise add a `100k` cell.

Reuse AR-0b sweep harness verbatim. The fix in [feedback_sweep_driver_fix.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_sweep_driver_fix.md) applies unchanged.

### 3.3 Pre-commitment to a small-N fallback

If even `N = 50k` does not reach a plateau, AR-2 ships at `N = 10k` with the fallback noted in §9 — empirically incomplete distillation is preferable to delaying AR-3.

## 4. Networks

### 4.1 Shared trunk (resolves AR-1 §9 "trunk sharing" question)

**Decision: load the AR-1 trunk frozen, attach two new heads.** The AR-1 winning checkpoint is `b64-h256-n2` — a Transformer over bid history + DeepSet over own hand + scalars, then 2 × (`Linear-LN-ReLU`, hidden 256). The trunk's pre-head 256-d activation is the **shared representation** for HandModel, CallPolicy, BidPolicy.

- The trunk is loaded with `--load-trunk path/to/handmodel/best.pt` and **frozen** (`requires_grad=False`) for all of AR-2.
- HandModel's bid head stays frozen too. Its only role in AR-2 is to produce `q` as a feature for the two new heads.
- Reasons for full freeze in AR-2 (rather than fine-tuning the trunk):
  1. AR-1 acceptance gate is already passed; trunk fine-tuning during distillation can only regress HandModel calibration without an auxiliary loss to defend against it (that loss lands in AR-3).
  2. Three heads × frozen trunk is the parent §7.8 "shared trunk + small heads" design at minimum risk.
  3. Cuts parameter count ~3× and inference ~3× as promised in §7.8.
  4. Lets `cfr_distillation.py` pre-compute and cache the 256-d trunk activations once per infostate; the heads then train on cached features at near-MLP speed.

### 4.2 CallPolicy (`P(call | info, q)`)

```
features = concat([
    trunk_repr(info),                  # 256
    q.q,                               # 110
    one_hot(info.standing_bid, 110),   # 110  (zero vector if standing_bid is None)
    [n / 25, q.q[standing_bid]],       # 2 scalars: pool-size and posterior-at-bid
])  # total 478
head = Linear(478 → 64) → LayerNorm → ReLU → Linear(64 → 1) → sigmoid
```

Output: `CallDecision(p_call, inputs={'q_at_bid': ..., 'peak_q': ..., 'n': ..., ...})`. `inputs` schema is open per parent §8.3.

Why this shape: CallPolicy is a binary classifier conditioned on a 478-d feature vector that already contains the sufficient statistic (`q.q[standing_bid]`). One hidden layer is enough; the network's job is to learn the bluff-propensity shift around that sufficient statistic, not to re-discover it.

### 4.3 BidPolicy (`π(action | info, q, ¬call)`)

```
features = concat([
    trunk_repr(info),     # 256
    q.q,                  # 110
    [n / 25],             # 1
])  # total 367
hidden  = Linear(367 → 128) → LayerNorm → ReLU → Linear(128 → NUM_BIDS=110)
logits  = hidden + log(q.q + 1e-12)         # warm-start from log q (parent §4.3)
masked  = mask_logits(logits, info.feasible_mask, info.legal_actions)
pi      = softmax(masked)
```

`hidden` is initialized small (orthogonal gain 0.01) so the head **starts** from `softmax(log q)` — i.e. equivalent to "bid the most likely true hand," exactly the parent §4.3 prior. Distillation drives it away from this prior toward the CFR+ average strategy.

Why include `q.q` as an explicit feature *and* in the logit prior: redundant but cheap, and makes the head capable of learning bid-frequency shifts that don't fall on the `log q` line (e.g. mixing strategies that hide hand strength).

### 4.4 HH gate — free-standing function

Parent §4.4 + checklist requirement. Implementation:

```python
# src/agents/learned/hh_gate.py
def should_declare_hh(
    belief: HandBelief,
    standing_bid: int | None,
    *,
    hh_band: float = 0.95,   # tunable; matches ExactRulesConditional default
) -> bool:
    if standing_bid is None:
        return False
    bi = standing_bid
    if int(np.argmax(belief.q)) != bi:
        return False
    return belief.q[bi] >= hh_band * belief.q.max()
```

This function is the **single source of truth** for HH in the modular agent and reflect rules — the existing `ExactRulesConditional` HH path can be refactored to call it, eliminating duplicated logic. Tests assert it fires iff `argmax(q) == bid_to_index(b)` AND `q[bi] >= hh_band * q.max()`, per the plan checklist.

`ModularNashAgent.action_probs` calls `should_declare_hh` first; on True returns the HH degenerate distribution; on False calls CallPolicy / BidPolicy with `hh_fired=False`.

## 5. CFR+ training of the targets

### 5.1 Handling `hh_fired` during CFR+

Plan checklist: "HH is a deterministic rule, not an action variable." Resolution:

- Treat HH as an action **only when** `should_declare_hh(true_belief_at_solve_time, standing_bid)` would fire under the analytic conditional (or under the AR-1 HandModel — see 5.1.1 below). At nodes where the gate would fire, action `HH_ACTION` is *forced* (not regret-matched). At all other nodes the action set excludes HH. This is identical to what `subgame_exploitability._legal_subgame_actions` does today — we are codifying it explicitly.
- Why force, not learn: §4.4 of the parent design proves HH is decision-theoretically optimal whenever the gate fires under the ±1 payoff model. CFR+ would converge to the same result, but forcing it eliminates the ε-tail on a deterministic rule and reduces the regret-matching state space.

#### 5.1.1 Belief at solve time

The CFR+ solver runs on the **true deal** (§1), so it knows the pool exactly. The HH gate is therefore deterministic at solve time: "HH fires iff `argmax(true pool's bid distribution) == standing_bid` AND the standing bid is the *pool's* best 5-card hand bid." This is a pure function of `(hands, standing_bid)`. No HandModel involvement during solving.

At inference time, the same gate runs on `belief.q` from AR-1's HandModel. This is the only place HandModel calibration matters for HH: if `q` puts most mass on the right bin, the gate fires correctly; if not, the gate misfires. AR-1's acceptance gate (Brier dominance) is the relevant guarantee.

### 5.2 Distillation loss

Plan checklist asks KL forward vs. KL reverse vs. cross-entropy. Resolution:

- **CallPolicy:** binary cross-entropy against `avg_call_prob`. Equivalent to KL(target ‖ pred) with one-hot edge cases handled.
- **BidPolicy:** **KL(target ‖ pred)** = forward KL = standard cross-entropy weighted by target probability. Reasons:
  - Mode-covering, not mode-seeking. The CFR+ solution is mixed; we want the student to cover the mix, not collapse onto a mode (which reverse-KL would do).
  - Reverse-KL can put *zero* probability on actions in the target's support, which collides with the masked-logit / feasibility constraints and produces unstable gradients near the mask boundary.
  - Cross-entropy on the average strategy is the textbook distillation loss for CFR-derived targets (e.g. DeepStack, Player of Games).
- Mass on infeasible actions in `target` is exactly zero (the solver respects the same legal-action mask), so no clipping is needed.

### 5.3 Entropy floor for `n ≤ 4`

Parent §7.6 calls this out and AR-2 owns the implementation. Resolution:

- During training: add a **schedule-dependent entropy regularizer** to the BidPolicy loss:
  ```
  L_total = KL(target ‖ pi) - β(n) · H(pi)
  ```
  with `β(n) = β_max · max(0, 1 - n / 5)` so `β(2) = 0.6 · β_max`, `β(4) = 0.2 · β_max`, `β(5+) = 0`. `β_max = 0.05` initially; tunable in the config but not swept (one knob at a time).
- At inference: a **lower-bound on per-step entropy** at `n ≤ 4` enforced by adding `α · 1` (uniform-over-feasible component) to `pi` and renormalizing, where `α` is the smallest constant such that `H(pi) ≥ H_floor(n)`. `H_floor(n) = H(uniform over feasible) · floor_frac(n)`, with `floor_frac(2)=0.6, floor_frac(3)=0.5, floor_frac(4)=0.4, floor_frac(5+)=0`. This addresses [feedback_test_opening_mix_flaky.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_test_opening_mix_flaky.md) structurally.

The training-time regularizer and inference-time floor are layered, not redundant: the regularizer makes the network *prefer* mixing; the floor *guarantees* a minimum.

### 5.4 Tie-break for `_select_bid` 4-way mixing fingerprint

Plan checklist names the legacy `_select_bid` 4-way mixing as needing a tie-break decision. Resolution: the legacy 4-way mixing is **deleted** in AR-2. The modular agent's mixing comes from `BidPolicy.bid_dist().pi` (a real probability vector sampled by the engine), not from `random.randint(4)`-flavoured fingerprinting. The "tie-break" question becomes: at sampling time, if `pi` has multiple actions with identical probability, how do we sample reproducibly?

- Engine sampling is already deterministic given `match_state.rng` (a `random.Random` seeded per match). `np.random.choice` with renormalized `pi` gives deterministic ties under that RNG. This is the existing convention and we keep it.

## 6. Datasets and pipeline

```
src/training/cfr_distillation.py
    ├── sample_deals(N, hand_size_mix) → list[Deal]
    ├── for each deal:
    │     SubgameSolution = CFRPlusSubgameSolver.solve(deal.hands)
    │     for each (infostate, target_call, target_bid_dist) in walk_avg_strategy(SubgameSolution):
    │          write row to data/runs/<run_id>/cfr_deals/{call,bid}_<shard>.npz
    ├── precompute trunk activations once per (deal, infostate) → cfr_deals/trunk_<shard>.npz
    └── train CallPolicy / BidPolicy heads on the cached features
```

Sharding is by deal index modulo 64 — keeps each `.npz` file under 100 MB at `N = 50k`. Train/val/test split is 80/10/10 *by deal* (never split rows from the same deal across splits — leakage).

## 7. Tests (must accompany implementation)

Per the plan checklist:

1. **Hand-crafted impossible standing bid.** Construct a deal where the standing bid is physically impossible at this `n` (e.g. Straight at `n=4`). Distilled `CallPolicy.call_prob(...).p_call > 0.95`. Solver target should also be ~1.0; this verifies the head learned the obvious case.
2. **Entropy floor at `n=2`.** 10³-deal property test: every `BidPolicy` output at `n=2` has `H(pi) ≥ H_floor(2)`. Asserts both training-time regularizer and inference-time floor work.
3. **HH gate truth table.** For 100 random `(belief, standing_bid)` pairs, `should_declare_hh` returns True iff `argmax(q) == standing_bid` AND `q[standing_bid] >= hh_band * q.max()`.
4. **Solver byte-equivalence.** Pre/post refactor: `subgame_exploitability(...)` returns the same numbers (within 1e-6) on a fixed 200-deal regression set.
5. **Trunk freeze invariance.** After one training epoch, trunk parameters' L2 norm is bitwise unchanged. (Cheap guard against accidentally including trunk in the optimizer.)
6. **Distillation reduces KL.** Sanity: validation `KL(target ‖ pi)` after training is at least 50% lower than at init (from `log q` warm-start).

Fast tests (1, 3, 5, 6) run in <30s each; tests (2, 4) marked `@pytest.mark.slow` per [feedback_pytest_slow_mark.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_pytest_slow_mark.md).

## 8. Acceptance gate

Per plan §AR-2:

1. **Win-rate gate.** A `ModularNashAgent` built from `(LearnedHandModel[frozen], DistilledCallPolicy, DistilledBidPolicy, should_declare_hh)` beats `ExactRulesConditional` 1v1 5-card by ≥ 5 pp at 200 games (95% CI excludes 50%).
2. **Exploitability gate.** Sampled-subgame exploitability at `n=10` strictly lower than `ExactRulesConditional`'s on the same 200-deal regression set used in §7 test 4.
3. **Distillation-budget gate.** The `KL(distilled ‖ avg_strategy)` curve over `N ∈ {1k, 5k, 10k, 50k}` has slope < 5% per doubling of `N` at the chosen final `N`. The chosen `N` is data-driven, not guessed.

Acceptance gates are evaluated **after** the sweep, not during.

## 9. Open questions deferred to AR-3 or follow-up

- **Auxiliary HandModel loss** to keep `q` calibrated under the policy distribution induced by AR-2's heads — deferred to AR-3 §6 Phase C as designed.
- **Population-based mixing / past-iterate anti-cycling** — that is the AR-3 lever; AR-2 distills against a static CFR+ target.
- **5p extension of `CFRPlusSubgameSolver`** — AR-5; non-trivial since CFR+ on `k > 2` zero-sum-team games needs a CFR-D-style modification. Not blocking 1v1.
- **Whether to unfreeze the trunk during AR-3 fine-tune** — AR-3 sub-design owns this. Default expectation: yes, with the §6 Phase C auxiliary HandModel loss as the calibration anchor.

---

**Stopping here per the design-first gate** ([feedback_design_first_gate.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_design_first_gate.md)). The AR-2 implementation session creates `src/training/cfr/subgame_solver.py` (the refactor), `src/training/cfr_distillation.py`, `src/agents/learned/callpolicy/`, `src/agents/learned/bidpolicy/`, `src/agents/learned/hh_gate.py`, `configs/sweeps/ar2_distillation_count.yaml`, and the `ModularNashAgent` wiring + tests.
