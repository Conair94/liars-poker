# AR-2 Implementation Checklist

- **Status:** In progress — Phase 1 complete (2026-05-01); Phases 2–9 pending
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

## Phase 2 — HH gate (free-standing)

- [ ] Create `src/agents/learned/hh_gate.py` with `should_declare_hh(belief, standing_bid, *, hh_band=0.95) -> bool` per design §4.4.
- [ ] Refactor `ExactRulesConditional`'s HH path to call `should_declare_hh` (single source of truth — design §4.4).
- [ ] Test §7.3: truth-table over 100 random `(belief, standing_bid)` pairs.

## Phase 3 — Network packages (heads only; trunk frozen)

- [ ] Create `src/agents/learned/callpolicy/{__init__.py, network.py, config.py, trainer.py}`.
  - [ ] `network.py`: 478-d feature concat → `Linear(478→64)→LN→ReLU→Linear(64→1)→sigmoid` per design §4.2.
  - [ ] `config.py`: dataclass with `hidden=64`, optimizer/LR, `--load-trunk` path.
- [ ] Create `src/agents/learned/bidpolicy/{__init__.py, network.py, config.py, trainer.py}`.
  - [ ] `network.py`: 367-d feature concat → `Linear(367→128)→LN→ReLU→Linear(128→110)`; add `+ log(q.q + 1e-12)` warm-start; mask via `info.feasible_mask` ∩ `info.legal_actions`; softmax. Init final layer `orthogonal(gain=0.01)` so init policy ≈ `softmax(log q)` (design §4.3).
  - [ ] `config.py`: includes `β_max=0.05` and `floor_frac` schedule per design §5.3.
- [ ] Both heads: `--load-trunk path/to/handmodel/best.pt`, `requires_grad=False` on every trunk param including HandModel's bid head. Trunk activations cached per (deal, infostate) per design §4.1.

## Phase 4 — Distillation pipeline

- [ ] Create `src/training/cfr_distillation.py` per design §6:
  - [ ] `sample_deals(N, hand_size_mix={4:.25, 6:.25, 8:.25, 10:.25})` reusing the existing deal sampler.
  - [ ] For each deal: call `CFRPlusSubgameSolver.solve(hands)`; `walk_avg_strategy(...)` over reachable infostates under support of average strategy.
  - [ ] Force HH action where the deterministic gate fires on the true pool; exclude HH otherwise (design §5.1).
  - [ ] Shard `.npz` rows by `deal_idx % 64` to `data/runs/<run_id>/cfr_deals/{call,bid,trunk}_<shard>.npz`.
  - [ ] 80/10/10 split **by deal**, never by row (design §6).
- [ ] Trunk-activation precompute pass writes `trunk_<shard>.npz` once.
- [ ] Loss functions:
  - [ ] CallPolicy: BCE vs `avg_call_prob` (design §5.2).
  - [ ] BidPolicy: forward KL `KL(target ‖ pi)` + entropy regularizer `-β(n)·H(pi)` with `β(n) = β_max · max(0, 1 - n/5)` (design §5.3).
- [ ] Inference-time entropy floor: add `α · uniform_over_feasible` and renormalize so `H(pi) ≥ H_floor(n)` with `floor_frac = {2:0.6, 3:0.5, 4:0.4, 5+:0}` (design §5.3).

## Phase 5 — Modular agent wiring

- [ ] `ModularNashAgent.action_probs`: call `should_declare_hh` first → degenerate HH dist if True; else delegate to `CallPolicy` then `BidPolicy` with `hh_fired=False`.
- [ ] Delete legacy `_select_bid` 4-way mixing fingerprint (design §5.4); sample `pi` from `BidPolicy.bid_dist().pi` via the existing match RNG.
- [ ] Register agent in `src/agents/registry.py` per [feedback_agent_registry_pattern.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_agent_registry_pattern.md).

## Phase 6 — Tests

- [ ] §7.1: hand-crafted impossible-bid deal → `p_call > 0.95`.
- [ ] §7.2 (slow): `n=2` 10³-deal property test for entropy floor.
- [ ] §7.3: HH gate truth table (covered in Phase 2 above).
- [ ] §7.4 (slow): solver byte-equivalence (covered in Phase 1 above).
- [ ] §7.5: trunk-freeze invariance — L2 norm of trunk params unchanged after one epoch.
- [ ] §7.6: validation `KL(target ‖ pi)` after training is ≥ 50% lower than at init.

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
