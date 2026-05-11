# AR-2 Implementation Checklist

- **Status:** In progress — Phase 1 complete (2026-05-01); Phases 2–3 complete (2026-05-02); Phase 4 complete (2026-05-03); Phases 5–6 complete (2026-05-11); Phases 7–9 pending
- **Date:** 2026-05-01
- **Parent design:** [agent_redesign_ar2.md](agent_redesign_ar2.md)
- **Parent plan:** [agent_redesign_plan.md](agent_redesign_plan.md) §AR-2
- **Predecessors:** AR-0a/0b at `fb17d03`; AR-1 winner pinned `b64-h256-n2`.

Sequenced so each phase's dependencies are already in place. Natural commit boundary at the end of each phase. Phase 1 is the only one that perturbs existing code — land it as its own commit with the byte-equivalence test before proceeding.

---

## Phase 1 — Solver refactor (touches existing code; do this first) ✅

- [x] Create `src/training/cfr/__init__.py` and `src/training/cfr/subgame_solver.py` exposing `CFRPlusSubgameSolver` and `SubgameSolution` per design §2.
- [x] Move bidding-tree primitives (`legal_subgame_actions`, `resolve_call_returns`, `resolve_hh_returns`, `build_canonical_match_state`, `pool_best_bid_idx`) into the shared module; `subgame_exploitability` now imports them.
- [x] `subgame_exploitability` keeps its public API (`_agent_value` / `_br_value` / `subgame_exploitability`) — they're metric-specific and unchanged.
- [x] **Implementation note (deviation from doc):** the existing `subgame_exploitability.py` did *not* contain a CFR+ solver — only memoized agent-policy + best-response DP. `CFRPlusSubgameSolver` is therefore a **new** implementation, not an extraction. It uses a forward/backward DP over the (cur_bid_idx, current_player) DAG (~220 reachable decision states) with regret-matching+ and linear (t-weighted) averaging — O(states × NUM_BIDS) per iteration. HH gating per design §5.1 is applied: the gate-firing successor states are forced HH terminals, never appearing as decision keys.
- [x] Byte-equivalence regression test (design §7.4): the 6 existing exploitability tests pass byte-identically post-refactor — that's the guard.
- [x] New solver tests in [tests/training/cfr/test_subgame_solver.py](../../tests/training/cfr/test_subgame_solver.py) cover: distribution validity, HH-gate exclusion, convergence rate, n=10 smoke, terminal zero-sum, and bid-mass-above-standing-bid invariant.
- [x] All 13 tests green (6 existing + 7 new).

## Phase 2 — HH gate (free-standing) ✅

- [x] Create [src/agents/learned/hh_gate.py](../../src/agents/learned/hh_gate.py) with `should_declare_hh(q, standing_bid, *, hh_band=0.95) -> bool` per design §4.4.
- [x] Refactor `ExactRulesConditionalAgent`'s HH path to call `should_declare_hh` (single source of truth — design §4.4). Two sites: `action_probs` and `choose_action`.
- [x] Test §7.3: [tests/agents/learned/test_hh_gate.py](../../tests/agents/learned/test_hh_gate.py) — truth-table over 100 random `(q, standing_bid)` pairs + 6 boundary tests; all 11 pass.
- [x] Regression: all 6 `test_action_probs.py` tests pass post-refactor (broader 140-test agents suite green except a pre-existing R-NaD import failure unrelated to this work).
- [x] **Deviation 1 — signature.** Function takes `q: np.ndarray` instead of `belief: HandBelief` (design §4.4 prose). Functionally equivalent — only `belief.q` is needed; both `ModularNashAgent` and `ExactRulesConditional` extract `q` and pass it. Avoids constructing a fully validated `HandBelief` (with `q_logits`, `feasible_mask`, `n`) at every call site.
- [x] **Deviation 2 — strict argmax semantics.** The pre-refactor `ExactRulesConditional` used `cur_idx == peak_idx OR cur_p >= hh_band * peak_p`, which would fire HH when the standing bid was a *near* peak. The design's strict-argmax form requires `argmax(q) == standing_bid`. The refactor adopts the strict semantics for all callers, so `ExactRulesConditional*` HH behavior is now slightly more conservative (fires only when the standing bid is the *true* peak, not when it's a close second). Expected ≤1 pp drift on 1v1 5-card benchmarks; re-measure post-AR-2.

## Phase 3 — Network packages (heads only; trunk frozen) ✅

Feature byte-layout pinned in [agent_redesign_ar2_feature_spec.md](agent_redesign_ar2_feature_spec.md). Trainer modules are stubs in this phase; loss bodies + datasets land in Phase 4.

- [x] Add `LearnedHandModelNet.trunk_forward(...)` returning the pre-head 256-d activation; refactor existing `forward` to call it. Byte-equivalent — guarded by AR-1 unit tests + new `test_trunk_forward_matches_full_forward`.
- [x] Create [src/agents/learned/callpolicy/](../../src/agents/learned/callpolicy/) (`__init__.py, network.py, config.py, trainer.py`).
  - [x] `network.py`: `CallPolicyNet` with `Linear(478→64)→LN→ReLU→Linear(64→1)→sigmoid` per design §4.2; `DistilledCallPolicy` Protocol wrapper + shared `build_call_features` numpy builder.
  - [x] `config.py`: dataclass with `hidden=64`, optimizer/LR, `--load-trunk` path; `input_dim` derived property.
- [x] Create [src/agents/learned/bidpolicy/](../../src/agents/learned/bidpolicy/) (`__init__.py, network.py, config.py, trainer.py`).
  - [x] `network.py`: `BidPolicyNet` with `Linear(367→128)→LN→ReLU→Linear(128→110)`; adds `+ log(q.q + 1e-12)` warm-start; masks via `info.feasible_mask[:NUM_BIDS]`; softmax in the wrapper. Final layer `orthogonal(gain=0.01)` per design §4.3.
  - [x] `config.py`: includes `β_max=0.05` and `floor_frac={2:0.6, 3:0.5, 4:0.4}` schedule per design §5.3 (consumed in Phase 4).
- [x] Both heads: `--load-trunk path/to/handmodel/best.pt` enforced in trainer; `requires_grad=False` on every trunk param at wrapper construction; optimizer registers head params only (verified by `test_trunk_excluded_from_optimizer`). Phase 3 uses uncached forward; Phase 4 distillation pipeline adds the trunk-activation cache per design §4.1.
- [x] Smoke tests in [tests/agents/learned/test_phase3_smoke.py](../../tests/agents/learned/test_phase3_smoke.py) — 10 tests covering trunk-forward parity, head construction, dim-mismatch error path, warm-start initialization (peaked-belief mode test), HH-fired empty-distribution shortcut, trunk-freeze invariance, and optimizer-param-set isolation. All green.
- [x] Trainer modules are stubs — `loss_step` raises `NotImplementedError` until Phase 4. `build_train_state` wires the optimizer + frozen trunk so Phase 4 only fills in losses.

## Phase 4 — Distillation pipeline ✅

Phase-4-specific design and step-level checklist live at [agent_redesign_ar2_phase4_design.md](agent_redesign_ar2_phase4_design.md) + [agent_redesign_ar2_phase4_checklist.md](agent_redesign_ar2_phase4_checklist.md).

- [x] `src/training/cfr_distillation.py` per design §6: `sample_deals_mixture` (per-seat hand_size {2,3,4,5} → pool {4,6,8,10}); `walk_avg_strategy` BFS; sharded `.npz` (`deal_idx % shard_count`); 80/10/10 split-by-deal in `split.json`.
- [x] Trunk-activation precompute pass: `trunk_<shard>.npz` aligned 1:1 with `bid_<shard>.npz`.
- [x] Loss functions:
  - [x] CallPolicy: BCE-with-logits against soft `avg_call_prob` (design §5.2). Added `CallPolicyNet._raw_logits` accessor for numerical stability.
  - [x] BidPolicy: cross-entropy form of `KL(target ‖ pi)` + `-β(n)·H(pi)` with `β(n) = β_max · max(0, 1 - n/5)`; `log_pi.clamp_min(-30)` defensive guard against `0·-inf` NaN (design §5.3).
- [x] Inference-time entropy floor: closed-form bisection on the convex mixture `(1-α)·pi + α·u`; applied only when `info.pool_size in floor_frac`. New helper `apply_entropy_floor` in `bidpolicy.network`.
- [x] Tests: 11 new fast (`test_phase4_losses.py` 5; `test_entropy_floor.py` 4; `test_cfr_distillation_smoke.py` 2 fast) + 1 slow §7.6 KL-reduction test (8.05 → 3.81 on N=32 smoke).
- [x] **Deviation 1 — `target_bid` masked to engine feasibility before normalization.** The solver's bid action set is "any bid > cur_bid_idx" (broader than the engine's hand-feasibility mask, which excludes bids no pool of this size can satisfy). At row emission we mask the solver's avg by `info.feasible_mask[:NUM_BIDS]` and renormalize so distillation targets live in the same space the network's softmax outputs. ~0.02% mass shifted on n=4 at max_iters=100; negligible at higher iter caps.

## Phase 5 — Modular agent wiring ✅

- [x] `ModularNashAgent.action_probs`: call `should_declare_hh` first → degenerate HH dist if True; else delegate to `CallPolicy` then `BidPolicy` with `hh_fired=False`. Implemented at [src/agents/learned/modular_nash_agent.py](../../src/agents/learned/modular_nash_agent.py).
- [x] Legacy `_select_bid` 4-way mixing fingerprint is not used by the new agent — `ModularNashAgent.choose_action` samples directly from `BidPolicy.bid_dist().pi` via `state.rng.choices` (design §5.4). The existing `ExactRulesMixedAgent._select_bid` is retained as a benchmark baseline (referenced by acceptance gate §8 and inherited by `ExactRulesOpponentModelAgent` / `ExactRulesAdaptiveAgent`).
- [x] Registered as `modular_nash` in `src/agents/registry.py` (`AGENT_REGISTRY` + `_AGENT_CLASS_MAP`). Factory `_make_modular_nash` raises `NotImplementedError` with a clear pointer to Phase 7 until head save/load lands alongside the distillation training script. Direct construction (`ModularNashAgent(hand_model, call_policy, bid_policy)`) works for tests.
- [x] 5 wiring smoke tests at [tests/agents/learned/test_modular_nash_smoke.py](../../tests/agents/learned/test_modular_nash_smoke.py): opener sums to 1, mid-round CALL/bid split (`p_call : (1-p_call)·pi`), HH gate fires when peak ≡ standing bid, `choose_action` returns legal, sampling is RNG-reproducible. All 5 pass.

## Phase 6 — Tests ✅

Phase-6-specific design and step-level checklist live at [agent_redesign_ar2_phase6_design.md](agent_redesign_ar2_phase6_design.md) + [agent_redesign_ar2_phase6_checklist.md](agent_redesign_ar2_phase6_checklist.md).

- [x] §7.1: hand-crafted impossible-bid deal → `p_call > 0.95`. Two tests at [tests/agents/learned/test_phase6_impossible_bid.py](../../tests/agents/learned/test_phase6_impossible_bid.py): solver target ≥ 0.98 at the `(infeasible_bid, 1)` state; mini-distillation (300 steps on the single infostate, random-init tiny trunk, no checkpoint) reaches `p_call > 0.95`.
- [x] §7.2 (slow): `n=2` 10³-deal property test for entropy floor at [tests/agents/learned/test_phase6_entropy_floor_n2.py](../../tests/agents/learned/test_phase6_entropy_floor_n2.py). 1000 infostates (500 opener + 500 mid-round) through `DistilledBidPolicy.bid_dist()`; all satisfy `entropy ≥ floor_frac[2] · log(feasible_count) − 1e-3`. Runtime 1 s.
- [x] §7.3: HH gate truth table (covered in Phase 2 above).
- [x] §7.4 (slow): solver byte-equivalence (covered in Phase 1 above).
- [x] §7.5: trunk-freeze invariance — L2 norm of trunk params unchanged after one epoch. Test at [tests/agents/learned/test_phase6_trunk_freeze.py](../../tests/agents/learned/test_phase6_trunk_freeze.py): 50 `BidPolicy.loss_step` + `optimizer.step()` iters on synthetic data; `_trunk_l2(before) == _trunk_l2(after)` bitwise; also asserts every trunk param has zero accumulated gradient.
- [x] §7.6: validation `KL(target ‖ pi)` after training is ≥ 50% lower than at init (covered in Phase 4 smoke).
- [x] Regression: full agents/ + training/ suite 187 passed (1 pre-existing defunct-R-NaD failure unrelated, per [feedback_cfr_rnad_defunct.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_cfr_rnad_defunct.md)).
- [x] **Deviation — §7.1 mini-distillation.** Design called for testing a fully-distilled head; without a real distillation run available pre-Phase-7, the implementation mini-trains a fresh `CallPolicyNet` (random-init tiny trunk, no checkpoint) on the *single* impossible-bid infostate repeated as batch=32 for 300 BCE steps against the solver label. Converges to `p_call > 0.95` reliably. Documented in the Phase 6 design §1 "Deviation note".

## Phase 7 — Pilot run + sweep

- [ ] **Pilot:** `N=1000`, single CPU run. Histogram per-deal solver iters and final ε. Confirm wall-clock < 30 min before launching the sweep (design §3.2).
- [ ] Create `configs/sweeps/ar2_distillation_count.yaml` with `N ∈ {1k, 5k, 10k, 50k}`. Reuse AR-0b harness (apply [feedback_sweep_driver_fix.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_sweep_driver_fix.md) — double-`-m` + `PYTHONPATH`).
- [ ] Held-out 2k-deal validation set, fixed and shared across cells.
- [ ] Plot `KL(distilled ‖ cfr_plus_avg)` vs `N` per `n`. If slope < 5% per doubling not reached at `N=50k`, add `100k` cell or fall back to `N=10k` per design §3.3.

## Phase 8 — Acceptance gate

- [ ] Build `ModularNashAgent` from `(LearnedHandModel[frozen], DistilledCallPolicy, DistilledBidPolicy, should_declare_hh)`.
- [ ] **Win-rate gate:** ≥ 5 pp over `ExactRulesConditional` at 200 games 1v1 5-card; 95% CI excludes 50%.
- [ ] **Exploitability gate:** sampled-subgame exploitability at `n=10` strictly below `ExactRulesConditional` on the 200-deal regression set from §7.4.
- [ ] **Distillation-budget gate:** chosen `N` is at the elbow per §3.2.

## Phase 9 — Docs + memory

- [ ] Memory: write `project_ar2_distillation.md` (impl outcomes, sweep winner `N`, distilled checkpoint path); update `MEMORY.md` index.
- [ ] CHANGELOG entry.
- [ ] Update `project_ar2_subdesign.md` status to "implemented" with run_id and acceptance numbers.

---

## Out of scope (do not touch)

- HandModel weights (frozen — design §1 non-goals).
- R-NaD (AR-3).
- Auxiliary HandModel calibration loss (AR-3 §6 Phase C).
- 5p extension of solver (AR-5).
