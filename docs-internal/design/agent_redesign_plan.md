# Agent Redesign — Implementation Plan / Checklist

- **Status:** Draft (plan stage)
- **Date:** 2026-04-28
- **Owner:** main
- **Companion to:** [agent_redesign.md](agent_redesign.md) (design doc)
- **Predecessor commit:** P5 closed at `89e3443`; markdown cleanup
  (P6) at `d4066fc`. Trunk is clean.

This document is the executable counterpart to `agent_redesign.md`.
The design doc is the *what and why*; this is the *in what order, by
which file, with which acceptance gate*. Per the project-wide
design-first gate
([feedback_design_first_gate.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_design_first_gate.md)),
every phase below opens with a sub-design doc before any code.
Sessions implement; this file does not.

## How to use this document

- Each phase has a fixed structure: **Goal → Inputs → Outputs →
  Checklist → Acceptance gate**.
- A box is `[ ]` open, `[~]` in-progress, `[x]` done. Boxes are the
  unit of work — each is small enough to land in a single session, or
  it is split.
- "Acceptance gate" is the binary `pass / fail` answer the next phase's
  design doc references. Do not advance until it passes.
- Stable run-id format: `<phase-tag>-<YYYYMMDDTHHMMSS>-<slug>-<git8>`.
  Phase tag is e.g. `ar1`, `ar2`. This appears in `data/runs/` and
  `data/sweeps/` outputs.

---

## Phase ordering at a glance

```
                    ┌── AR-0a (shared types + protocols + checkpoint schema)
   AR-0 (design)  ──┤
   = this doc        └── AR-0b (sweep driver + bench_sweep + compare)
                         │
                         ▼
                    AR-1 (HandModel: design → impl → Phase-A pretrain)
                         │
                         ▼
                    AR-2 (CallPolicy + BidPolicy: design → impl →
                          Phase-B CFR+ distillation)
                         │
                         ▼
                    AR-3 (R-NaD fine-tune; Phase-C end-to-end)
                         │
                         ▼
                    AR-4 (Stage-1 acceptance: 1v1 5-card)
                         │
                         ▼
                    AR-5 (Stage-2 lift: 5-player 5-card)
```

AR-0a and AR-0b are *not* in the design doc's phase table — they were
implicit. We surface them here because AR-1 cannot start until the
shared `Infostate` / `HandBelief` / etc. types and the sweep
infrastructure exist. Both are small (~1 session each) but
load-bearing for everything that follows.

---

## Pre-flight (do once, before AR-0a)

- [ ] Re-run the full `tests/` suite on trunk and capture baseline
      pass/fail. Pre-existing flakes
      ([test_opening_mix](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_test_opening_mix_flaky.md),
      `test_bid_count`) are noted but not fixed in this work — just
      recorded so regressions vs. trunk are unambiguous.
- [ ] ~~Confirm `cfr_plus_mb4_hh` checkpoint is on disk.~~ **Skipped.**
      `CFRNashAgent` and `cfr_plus_mb4_hh` are defunct; no new tests or
      benchmarks should reference them.
- [ ] Verify `extended_conditional_exact_probs.json` cache exists in
      `data/probs/`; AR-1 Phase-A sanity-check compares against it.
- [ ] Open a tracking branch `agent-redesign` off main. Every phase
      below merges into this branch; `agent-redesign → main` PR after
      AR-4.

---

## AR-0a — Shared types, protocols, and checkpoint schema

**Goal.** Land the §8 I/O contracts as runnable Python so every
subsequent phase imports the same dataclasses and Protocols. No model
training. Just types, masks, and one round-trip test.

**Inputs.** Design doc §8.

**Outputs.**

- `src/agents/contracts.py` — `Infostate`, `HandBelief`,
  `CallDecision`, `BidDistribution`, `AgentDecision`,
  `HandModel` / `CallPolicy` / `BidPolicy` Protocols.
- `src/agents/contracts_io.py` — JSON (de)serialization for
  `Infostate` and the trace fields used by `decision_capture.py`.
- `src/agents/checkpoints.py` — the §8.6 checkpoint schema:
  `save_component(path, component, config, state_dict, iter, parent_run=None)`,
  `load_component(path) -> dict`. One canonical I/O point.
- Tests under `tests/agents/`.

**Checklist.**

- [ ] Sub-design doc `agent_redesign_ar0a.md` covering: exact dtypes,
      mask invariant assertions (debug-only? always?), JSON canonical
      form for `Infostate.bid_history`, and the `parent_run` semantics
      for cross-run composition.
- [ ] Implement `Infostate` with `from_match_state(state) -> Infostate`.
      Property-based test: round-trip 1000 random `MatchState`s
      through `Infostate` and back; `legal_actions` and
      `feasible_mask` match `state.legal_actions()` ∩ `_is_bid_feasible`.
- [ ] Implement `HandBelief`, `CallDecision`, `BidDistribution`,
      `AgentDecision` dataclasses with the §8 invariant checks.
- [ ] Implement `HandModel` / `CallPolicy` / `BidPolicy` Protocols
      and a no-op `IdentityHandModel` (uniform-over-feasible) used
      only by tests.
- [ ] Implement checkpoint save/load. Round-trip a dummy state_dict
      through both and assert key parity.
- [ ] Extend
      [src/training/decision_capture.py](../../src/training/decision_capture.py)
      to accept an optional `AgentDecision` and serialize the
      `belief` / `call` / `bid` / `hh_fired` fields per design §8.7.
      Existing callers unchanged (new fields default to `None`).
- [ ] Update
      [src/agents/policy.py](../../src/agents/policy.py)
      to add an `AgentDecision`-returning helper alongside the
      existing `action_probs(agent, state) -> dict[int, float]`.
      The old function does not change behavior.

**Acceptance gate.**

- All new tests pass; existing `tests/` suite has no new failures.
- A scratch script can: build any registry agent → wrap in the new
  `AgentDecision` adapter → write a JSONL trace → reload it → recover
  the same `action_probs`. End-to-end round trip works for at least
  `RandomAgent` and `ExactRulesConditionalAgent`. (`CFRNashAgent` is
  **defunct** — do not write new tests for it.)

---

## AR-0b — Sweep driver, bench sweep, comparator

**Goal.** Land the §9 parallel-research pipeline before any sweepable
agent exists. Validates the harness on the existing heuristic ladder
so AR-1 onward gets a working tool, not a promise.

**Inputs.** AR-0a outputs; design doc §9.

**Outputs.**

- `src/training/sweep.py` — sweep driver (`python -m training.sweep …`).
- `src/training/bench_sweep.py` — post-sweep pairwise benchmark.
- `src/training/compare.py` — Pareto / calibration / decision-diff
  reporter.
- `configs/sweeps/` directory, with one example config
  (`heuristic_ladder_smoketest.yaml`) that "trains" nothing and just
  benchmarks the existing ladder against itself — proves the harness.
- `data/sweeps/<sweep_id>/{config.yaml,index.json,bench/results.parquet,report/}`
  layout established.

**Checklist.**

- [ ] Sub-design doc `agent_redesign_ar0b.md` covering: subprocess vs.
      multiprocess (recommend subprocess for crash isolation), config
      schema for axis-product vs. explicit list, Parquet columns for
      `results.parquet`, atomic-write strategy for `manifest.json`.
- [ ] Implement `training.sweep` with `--max-parallel`, `--device`,
      `--cpu-quota`, `--resume`. Atomic manifest writes. Refuses to
      start a config flagged `production: true` if `git status` is
      dirty.
- [ ] Implement `training.bench_sweep` reusing
      [src/training/benchmark.py](../../src/training/benchmark.py)
      pairwise infra. Output Parquet with columns
      `(run_id, opponent, metric, value, ci_low, ci_high, n_games)`.
- [ ] Implement `training.compare`: Pareto plot
      `(LBR exploitability, win-rate vs. heuristic ladder)`,
      calibration plot for HandModel (skipped when no HandModel in
      run), decision-diff (sample N matches, log turns where two runs
      diverge).
- [ ] Smoketest config `configs/sweeps/heuristic_ladder_smoketest.yaml`:
      runs `bench_sweep` over all `Exact*` agents pairwise. Replicates
      the existing 2026-04-24 benchmark numbers within MC noise.
- [ ] CHANGELOG entry.

**Acceptance gate.**

- Smoketest sweep produces a `results.parquet` whose top-level
  win-rate column for `ExactRulesConditional` matches the recorded
  85.8% within 95% CI given the configured `--games-per-pair`.
- A second `--resume` invocation against the same sweep id is a no-op
  (all runs detected as completed).
- Killing a run mid-flight and re-invoking resumes from the last
  manifest write — no double-counted games, no data loss.

---

## AR-1 — HandModel (Phase A pretrain)

**Goal.** Train a calibrated `q(pool | info)` that strictly improves
on the analytic conditional table by also conditioning on opponent bid
history.

**Inputs.** AR-0a + AR-0b outputs; design doc §4.1, §6 Phase A.

**Outputs.**

- `src/agents/learned/handmodel/{network.py,trainer.py,config.py,dataset.py}` —
  new package, parallel to `learned/rnad/`.
- `data/runs/<ar1-run_id>/` checkpoints + JSONL rollouts.
- One trained `LearnedHandModel` checkpoint passing AR-1 acceptance.
- Reusable `AnalyticHandModel` and `HeuristicHandModel` adapters in
  `src/agents/learned/handmodel/baselines.py` so the §9.4 ablation
  matrix has all three flavors immediately.

**Checklist.**

- [ ] Sub-design doc `agent_redesign_ar1.md`: network depth /width,
      bid-history encoder choice (transformer vs. small RNN),
      exact tokenization scheme for `(seat, bid_idx)`, position
      embedding for round-position, dataset size for Phase A,
      train/val split strategy, calibration metric definition
      (Brier vs. NLL, per-`n` vs. pooled), early-stop criterion.
- [ ] Implement `network.py` — reuse the card-embedding + DeepSet
      pattern from
      [src/agents/learned/rnad/network.py](../../src/agents/learned/rnad/network.py)
      for the private-hand encoder; new bid-history encoder.
- [ ] Implement `dataset.py` — rollout collector that drives
      `ExactRulesAdaptive` self-play and writes
      `(Infostate, true_pool_best_bid_index)` JSONL to
      `data/runs/<run_id>/rollouts/`. ~10⁶ decisions across hand sizes
      2..5 mixture.
- [ ] Implement `trainer.py` — supervised cross-entropy on the masked
      logit space. Hand size mixture = 25% each of n ∈ {2,3,4,5} per
      side (1v1 → pool 4..10).
- [ ] Implement `AnalyticHandModel` (delegates to
      `WarmStartLookup.get_exact_rules_conditional`) and
      `HeuristicHandModel` (extracts `_compute_adj_exact` from
      `ExactRulesOpponentModelAgent` behind the `HandModel` Protocol).
- [ ] Tests under `tests/agents/learned/handmodel/`:
  - [ ] Mask invariant: `q[~feasible_mask].sum() < 1e-6` on 1000
        random infostates.
  - [ ] Calibration vs. analytic baseline on `n ∈ {5..25}` —
        `LearnedHandModel` Brier ≤ `AnalyticHandModel` Brier on the
        held-out set, by at least the noise floor.
  - [ ] `belief_batch` returns identical results to
        looped `belief` calls.
- [ ] Sweep config `configs/sweeps/ar1_handmodel_arch.yaml`: vary
      depth, hidden dim, bid-history encoder type. Run via AR-0b
      sweep driver.
- [ ] Update `MEMORY.md` with `project_ar1_handmodel.md`.
- [ ] CHANGELOG entry.

**Acceptance gate.**

- `LearnedHandModel` strictly Pareto-dominates `AnalyticHandModel` on
  Brier across `n ∈ {5..25}` on the held-out set.
- Mask invariants hold on a 10⁵-sample property test.
- Sweep over architectures produces a single recommended config
  written into `configs/agents/handmodel_v1.yaml` with its `run_id`
  pinned.

---

## AR-2 — CallPolicy and BidPolicy (Phase B CFR+ distillation)

**Goal.** With HandModel frozen, distill a strong starting point for
both decision heads from per-deal CFR+ solutions on the bidding
subgame.

**Inputs.** AR-1's frozen `LearnedHandModel`; design doc §4.2, §4.3,
§6 Phase B.

**Outputs.**

- `src/agents/learned/callpolicy/{network.py,trainer.py,config.py}`.
- `src/agents/learned/bidpolicy/{network.py,trainer.py,config.py}`.
- `src/training/cfr_distillation.py` — sampled-deal CFR+ pipeline
  reusing the solver from
  [src/training/metrics/subgame_exploitability.py](../../src/training/metrics/subgame_exploitability.py).
- `data/runs/<ar2-run_id>/{cfr_deals/*.npz, callpolicy/, bidpolicy/}`.

**Checklist.**

- [ ] Sub-design doc `agent_redesign_ar2.md`: deal-sample count budget
      (with empirical loss-vs-count plot from a pilot run), how to
      handle the `hh_fired` short-circuit during CFR+ (HH is a
      deterministic rule, not an action variable), entropy floor
      schedule for `n ≤ 4`, distillation loss (KL forward vs. reverse
      vs. cross-entropy), tie-break for the `_select_bid` 4-way
      mixing fingerprint.
- [ ] Refactor existing `subgame_exploitability.py` solver into a
      reusable `CFRPlusSubgameSolver` if not already shaped that way.
- [ ] Implement `cfr_distillation.py` — for each sampled deal: solve
      the bidding subgame with HandModel-induced beliefs as the chance
      prior, log per-infostate `(p_call, pi)` to NPZ.
- [ ] Implement `CallPolicy` and `BidPolicy` networks. Both share the
      AR-1 trunk via a `--load-trunk` flag (per design §7.8).
- [ ] Implement BidPolicy logit warm-start from `log q` (design §4.3).
- [ ] Implement HH gate as a free-standing function
      `should_declare_hh(belief, standing_bid) -> bool` so it can be
      shared between `ModularNashAgent` and reflect rules.
- [ ] Tests:
  - [ ] On a hand-crafted deal where the analytic optimal call is
        obvious (standing bid is impossible at this `n`), distilled
        CallPolicy outputs `p_call > 0.95`.
  - [ ] BidPolicy entropy at `n=2` is bounded below by the configured
        entropy floor on a 10³-sample property test.
  - [ ] HH gate fires iff `argmax q == bid_to_index(b)` and
        `q[bid_to_index(b)] >= hh_band * peak_q`.
- [ ] Sweep config `configs/sweeps/ar2_distillation_count.yaml`:
      vary deal-sample count `∈ {1k, 5k, 10k, 50k}`, plot loss curves
      via the AR-0b comparator.
- [ ] Update `MEMORY.md` with `project_ar2_distillation.md`.
- [ ] CHANGELOG entry.

**Acceptance gate.**

- A `ModularNashAgent` built from `(LearnedHandModel,
  DistilledCallPolicy, DistilledBidPolicy)` beats
  `ExactRulesConditional` head-to-head at 1v1 5-card by ≥ 5 percentage
  points (200 games, two-sided 95% CI excludes 50%).
- Sampled-subgame exploitability of the same agent is strictly lower
  than `ExactRulesConditional`'s.
- Loss-vs-deal-count curve has visibly plateaued — picking a final
  deal count is data-driven, not guessed.

---

## AR-3 — R-NaD fine-tune (Phase C end-to-end)

**Goal.** Polish the AR-2 starting point via population-based
self-play. This is where the existing `learned/rnad/` trainer comes
back into play — but with HandModel and a sane initialization, not
from scratch.

**Inputs.** AR-2 unified checkpoint; design doc §6 Phase C, §7.2.

**Outputs.**

- Refactored `src/agents/learned/rnad/trainer.py` that consumes the
  modular components as separate heads on the shared trunk.
- `src/training/population.py` — frozen-historical population manager
  (design §7.2).
- A series of unified checkpoints under `data/runs/<ar3-run_id>/`.
- Final R-NaD checkpoint passing AR-3 acceptance.

**Checklist.**

- [ ] Sub-design doc `agent_redesign_ar3.md`: η schedule for R-NaD on
      three heads, when to checkpoint into the population (LBR-drop
      trigger from §7.2), opponent-mix schedule (80/10/10 ratio
      validation), auxiliary HandModel-loss weight, freeze schedule
      (do we ever freeze HandModel during C?).
- [ ] Refactor R-NaD trainer to read from the AR-0a unified
      checkpoint format and write back to it.
- [ ] Implement `training.population` — a directory-backed list of
      frozen checkpoints with an LBR-triggered append rule.
- [ ] Implement the auxiliary HandModel loss against ground-truth
      pool outcomes from each rollout (design §6 Phase C).
- [ ] Tests:
  - [ ] Full Phase-C run on a tiny config (50 iters, n=2 only)
        terminates without NaN, produces a valid unified checkpoint.
  - [ ] Population manager appends iff LBR drops by configured
        threshold; tested with synthetic LBR series.
- [ ] Sweep config `configs/sweeps/ar3_rnad_eta.yaml`: vary η, entropy
      coefficient, population-mix ratio.
- [ ] Update `MEMORY.md` with `project_ar3_rnad.md`.
- [ ] CHANGELOG entry.

**Acceptance gate.**

- Final AR-3 checkpoint Pareto-dominates the AR-2 distilled checkpoint
  on (LBR exploitability, sampled-subgame exploitability) and on
  pairwise win rate vs. the heuristic ladder.
- No regressions on calibration: HandModel Brier (with the auxiliary
  loss) is no worse than AR-1's frozen Brier within MC noise.

---

## AR-4 — Stage-1 acceptance (1v1 5-card)

**Goal.** Declare Stage-1 done against the design doc's stop criterion
(§5.1).

**Inputs.** AR-3 final checkpoint.

**Outputs.**

- A signed acceptance report at
  `data/runs/<ar4-run_id>/acceptance.md` listing all metric values vs.
  thresholds.
- Either:
  - GO: AR-5 begins, OR
  - NO-GO: a remediation sub-doc identifying which threshold failed
    and what AR-3.x rerun (or AR-2.x re-distillation) addresses it.

**Checklist.**

- [ ] Run pairwise benchmark vs. every heuristic ladder agent at every
      `n ∈ {2,3,4,5}` per side. 200 games per pair.
- [ ] Run LBR exploitability with the depth that AR-1's design doc
      pinned. Log per-`n`.
- [ ] Run sampled-subgame exploitability with the deal count from
      AR-2.
- [ ] ~~Re-benchmark `cfr_plus_mb4_hh`.~~ **Skipped.** `CFRNashAgent` is
      defunct; acceptance comparisons use only the heuristic `Exact*` ladder.
- [ ] Decision-diff sample of 50 matches between final agent and
      `ExactRulesConditional`; spot-check the 10 highest-divergence
      turns by hand.
- [ ] Write `acceptance.md`. Include all numbers + the GO / NO-GO
      decision.
- [ ] Update `README.md` Project Status table to add an "Agents"
      row. Update `CHANGELOG.md`.

**Acceptance gate (the actual Stage-1 stop criterion).**

- ≥ 60% pairwise win rate vs. every heuristic ladder agent at every
  `n ∈ {2,3,4,5}`.
- LBR exploitability < `ExactRulesConditional`'s LBR.
- Sampled-subgame exploitability < `ExactRulesConditional`'s.

---

## AR-5 — Stage-2 lift (5-player 5-card)

**Goal.** Generalize the Stage-1 agent to 5-player play via
fine-tuning, not retraining.

**Inputs.** AR-4 GO checkpoint; design doc §5.2.

**Outputs.**

- 5p-capable `ModularNashAgent` checkpoint.
- 5p extension of sampled-subgame exploitability metric (open
  question §11.2 in the design doc).

**Checklist.**

- [ ] Sub-design doc `agent_redesign_ar5.md`: per-seat embedding
      strategy for 5p (design §7.3 — relative seat offset), 5p LBR
      cost analysis, 5p subgame-exploitability "team" definition,
      whether HH gate (design §11.3) needs replacement.
- [ ] Extend `Infostate` adapter for 5p (`hand_sizes` is already a
      tuple — verify nothing else needs widening).
- [ ] Extend bid-history encoder for the longer 5p sequence.
- [ ] Extend `subgame_exploitability.py` for `k > 2`.
- [ ] Fine-tune AR-4 checkpoint at 5-player 5-card. Freeze schedule
      from sub-design doc.
- [ ] Run pairwise round-robin tournaments and 5p LBR.
- [ ] Acceptance report `acceptance_5p.md`.
- [ ] Update `MEMORY.md` and `README.md`. Update `CHANGELOG.md`.

**Acceptance gate.**

- AR-5 agent wins ≥ 30% of 5p round-robin matches against a population
  of `{Stage-1 self-play, frozen historicals, heuristic ladder lifted
  to 5p}` (random would win 20%; 30% is "decisively above random").
- 5p LBR exploitability < lifted `ExactRulesConditional`.

---

## Cross-cutting concerns

These must be true at every phase; flagged here so they don't get
buried in any one checklist.

- [ ] **Naming.** Final class name for the modular learned agent is
      decided before AR-1 lands. Working name is `ModularNashAgent`
      (design §11.5). Per
      [feedback_agent_naming.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_agent_naming.md)
      it must be descriptive — confirm with user during AR-1's design
      sub-doc review.
- [ ] **Registry.** Every learned-agent flavor (`Analytic` /
      `Heuristic` / `Learned` × {handmodel, callpolicy, bidpolicy})
      registers via `AGENT_REGISTRY` + `_AGENT_CLASS_MAP` per
      [feedback_agent_registry_pattern.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_agent_registry_pattern.md).
      No if/else chains.
- [ ] **CPU first.** All training defaults to CPU per
      [feedback_mps_cpu_speed.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_mps_cpu_speed.md).
      MPS / CUDA are opt-in via `--device`.
- [ ] **Reflect rules.** As each new component lands, update
      [src/training/reflect.py](../../src/training/reflect.py)
      to consume the new trace fields. No standalone phase for this —
      it rides along with AR-1, AR-2, AR-3.
- [ ] **Markers.** Long-running tests use
      `@pytest.mark.slow` per
      [feedback_pytest_slow_mark.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_pytest_slow_mark.md).
- [ ] **eval.py ruleset params.** Any new evaluation path follows
      [feedback_eval_ruleset_params.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_eval_ruleset_params.md):
      pass `exact_rules` and `high_hand` from config, never default.

---

## Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| AR-1 HandModel doesn't beat the analytic table | Medium | Blocks AR-2 acceptance | Acceptance gate is Pareto-dominance per `n`; if it fails, fall back to `AnalyticHandModel` and proceed; we still get the AR-2/AR-3 gains, just lose the bid-history-conditioning headroom |
| AR-2 CFR+ distillation is too slow at scale | Medium | Stretches AR-2 schedule | Sweep over deal counts is mandatory in AR-2; pilot run at 1k deals before committing to 50k |
| AR-3 R-NaD diverges from the AR-2 starting point | Medium | AR-3 acceptance fails | Population mix + auxiliary HandModel loss are the structural defenses; if they fail, freeze HandModel during C and reduce η — both are config flips |
| 5p sampled-subgame exploitability is intractable | Low | Stretches AR-5 schedule | Acceptable to ship AR-5 with LBR + pairwise win rate only; subgame exploitability is the "nice-to-have" honesty check, not the gate |
| Sweep driver + bench harness eats more time than planned | Medium | Slips AR-1 start | AR-0b is a hard prerequisite — own its session and don't conflate it with AR-1; the sweep harness pays for itself by AR-1 anyway |
| Pre-existing `test_opening_mix` flake masks real regressions | Low | Confusion only | Baseline pass/fail captured in pre-flight; new failures judged against that baseline |

---

## Done definition for this plan

This plan is *fully executed* when:

- AR-4 acceptance is GO, AND
- AR-5 acceptance is GO, AND
- The `agent-redesign` branch is merged to `main`, AND
- README's Project Status table shows AR-1..AR-5 as ✅, AND
- `MEMORY.md` has up-to-date `project_ar*` entries reflecting the
  final state and pointers to the canonical checkpoints.

Until then, the latest in-progress phase's acceptance gate is the
single load-bearing question — everything else is in service of it.
