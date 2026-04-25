# Training Pipeline Refactor — Plan / Checklist

**Status:** Draft for execution
**Date:** 2026-04-24
**Companion to:** `TRAINING_PIPELINE_DESIGN.md` (rev. 3)
**Next artifact:** `TRAINING_PIPELINE_TESTS.md` (per-module test checklist, produced alongside P1)

This document turns the design decisions in `TRAINING_PIPELINE_DESIGN.md` §13 and the "Now" bucket in §14.12 into a concrete, session-sized checklist. Every phase is scoped to fit in one focused session (possibly two for the larger ones). Each phase has **preconditions**, a **task checklist**, **exit criteria**, and **session-handoff notes** so work can resume cold.

## How to Use This Document

1. Work one phase at a time. Do not start phase *N+1* until *N*'s exit criteria are all green.
2. Tick checklist items `[x]` as they land. Never delete unchecked items — mark blocked as `[!]` with a one-line reason.
3. At end of session, update the **Session Handoff** block at the bottom so the next session lands cold and productive.
4. If a task grows past its phase's scope, split it — don't let phases bleed.
5. User-action items (marked **USER**) block the session until the user confirms they're done.

---

## Phase Overview

| Phase | Goal | Estimated sessions |
| --- | --- | --- |
| **P0** | Preflight — tracking stack, ADR infra, baseline snapshot | 1 |
| **P1** | `src/` skeleton + game package move + unified test tree | 1–2 |
| **P2** | Frontend retirement — undeploy, archive, README update | 1 |
| **P3** | Decision logging schema + reflect v1 (rule-based) | 1–2 |
| **P4** | OpenSpiel adapter + exploitability + small-game oracle | 2 |
| **P5** | Modular agent refactor (heuristics frozen, CFR, R-NaD) | 2–3 |
| **P6** | Markdown consolidation + acceptance validation + doc polish | 1 |
| Total | | **~9–12 sessions** |

Dependencies: P2 depends only on P0. P3 depends on P1. P4 depends on P1 + P3. P5 depends on P4. P6 depends on all prior.

---

## P0 — Preflight

**Goal:** install the habits and tooling that every later phase will use, before they're needed. Skipping any of these turns later sessions into firefighting.

### Preconditions

- Design doc rev. 3 approved (done 2026-04-24).
- No uncommitted changes on `main` (snapshot the benchmark baseline first).

### Checklist

- [x] **USER:** Create a Weights & Biases account (free academic plan). Project `liars-poker` will be auto-created on first `wandb.init` call. User to set `WANDB_ENTITY=<username>` env var so the username is not committed to the repo.
- [x] Add dev dependencies: `hydra-core`, `wandb`, `pytest`, `pytest-cov`, `ruff`, `mypy` to a new `pyproject.toml` at repo root. (Optional `openspiel` extra also added for P4.)
- [x] Create `docs-internal/design/adr/` with `README.md` explaining the ADR format (one file per irreversible decision, dated, titled, status, context, decision, consequences).
- [x] Write **ADR-001: Frontend retirement** (2026-04-24 — rationale from design doc §8).
- [x] Write **ADR-002: OpenSpiel adoption** (rationale from Q10).
- [x] Write **ADR-003: Infrastructure-first project posture** (rationale from Q1).
- [!] Snapshot current benchmark as the regression baseline — **deferred** per user direction 2026-04-24. Rationale: most agents will be re-benchmarked post-refactor anyway, so the §12 A1 acceptance check will be evaluated against whichever benchmark is current when P5 lands rather than a frozen P0 snapshot. Not a blocker.
- [x] Add a `CHANGELOG.md` at repo root, seeded with the baseline snapshot as v0.
- [ ] Commit: "P0: preflight — ADR log, tracking stack, pyproject.toml."

### Exit Criteria

- [ ] `wandb login` works locally. **USER to verify before P3** (W&B is not actually invoked in P0).
- [ ] `hydra-core` importable from a Python REPL in the project env. **USER to verify after `pip install -e ".[dev]"`.**
- [x] `docs-internal/design/adr/` contains ADRs 001–003.
- [~] `data/runs/20260424-baseline-premigration/metrics.json` — **deferred** (see checklist note above).

### Session Handoff Notes

Once P0 is complete, note in the final session summary which heuristic-ladder win-rates are the frozen targets for §12 A1 (read them out of `metrics.json`). They must not move by more than 2pp after the refactor.

---

## P1 — `src/` Skeleton + Game Package + Unified Tests

**Goal:** establish the new directory shape without changing any behavior. Pure file moves and import fixups.

### Preconditions

- P0 complete.
- Baseline benchmark archived (so a post-P1 rerun can be compared byte-for-byte on the heuristic ladder).

### Checklist

- [x] Create top-level directories: `src/`, `paper/` (others created on demand: `data/`, `archive/` later).
- [x] Move LaTeX sources from `Liars poker/` (`.tex`, `.bib`, `figures/`, all `.aux/.log/.pdf` build artifacts) → `paper/`. Update `paper/README.md` with the build command.
- [x] Move probability-table generators (`compute_*.py`, `poker_math_exact.py`, `generate_prob_tables.py`) from `Liars poker/` → `src/training/probs/`.
- [!] Move probability-table JSON caches from `Liars poker/agent/data/*.json` → `data/probs/`. **Deferred:** kept in old location to keep each commit atomic; will move alongside the consumer that pulls them (commit between P1.5 and P1.6).
- [x] Move `Liars poker/agent/game/` → `src/game/`. Verify zero imports from `src/agents/` — `game/` is a leaf package.
- [ ] Move `Liars poker/agent/baseline/` → `src/agents/heuristic/` **as-is, no logic changes** (Q3: frozen).
- [ ] Move `Liars poker/agent/search/` → `src/agents/search/`.
- [ ] Move `Liars poker/agent/rnad/` → `src/agents/learned/rnad/`.
- [ ] Move `Liars poker/agent/checkpoints/` → `data/checkpoints/` (one-shot, Q8).
- [ ] Consolidate all `tests/` directories into one tree at repo root, mirroring `src/` (Q9). Remove the scattered `__pycache__` and `.pytest_cache` artifacts.
- [ ] Rewrite `pyproject.toml` to point at `src/` and `tests/`.
- [ ] Fix all imports broken by the moves. Use `ruff check --fix` to catch the easy ones.
- [ ] Run full test suite + baseline benchmark; confirm no regressions.
- [ ] Write `src/README.md` (≤100 lines) describing subpackages.
- [ ] Top-level `README.md` rewrite: project summary, how to run benchmarks, pointer to `docs-internal/design/`.
- [ ] Commit: "P1: move to src/ layout; zero behavior change."

### Exit Criteria

- Heuristic ladder benchmark win-rates are within ±0.5pp of P0 baseline (tight tolerance — this is a no-behavior-change phase).
- `pytest tests/` passes.
- `src/README.md` ≤ 100 lines; repo root has ≤ 3 `.md` files (`README.md`, `CHANGELOG.md`, `CLAUDE.md`).
- Nothing of substance left in `Liars poker/` except its name (directory can be removed in P6).

### Session Handoff Notes

List every import path that changed, so downstream phases know where to find things. If any test was temporarily skipped to unblock the move, file it as a blocker in P1's checklist.

---

## P2 — Frontend Retirement

**Goal:** execute §8.1 of the design doc. Take the public web demo offline; archive its code.

### Preconditions

- P0 complete (P1 not required — retirement is independent).

### Checklist

- [ ] **USER:** In GitHub repo Settings → Pages, set "Source" to `None`. Confirm the `https://<user>.github.io/liars-poker/` URL 404s after a few minutes.
- [ ] Move `docs/` → `archive/web-2026-04/docs/`.
- [ ] Move `Liars poker/agent/web/` → `archive/web-2026-04/web/`.
- [ ] Delete or disable any `.github/workflows/*.yml` that builds or deploys Pages.
- [ ] Remove "Play online" link and Pages badge from root `README.md`. Add a one-line note: "Local-dev only until agents are strong enough to re-host."
- [ ] Write **ADR-004: Frontend archived on disk** (points to `archive/web-2026-04/` and notes the 2026-04-24 retirement date). Distinct from ADR-001 which decided the retirement; this one records the *execution*.
- [ ] Add `archive/README.md` explaining that everything under `archive/` is frozen and read-only.
- [ ] Commit: "P2: retire public web demo; archive frontend."

### Exit Criteria

- `https://<user>.github.io/liars-poker/` returns 404.
- `archive/web-2026-04/` exists with `docs/` and `web/` inside.
- Top-level `docs/` folder no longer exists.
- No GitHub Action runs on push to `main` that touches Pages.
- README no longer advertises online play.

### Session Handoff Notes

If the user has not yet disabled Pages in the GitHub UI, do not merge the archive-move commit — it would leave the repo in a state where GH Pages deploys from a missing directory. Block on the user.

---

## P3 — Decision Logging + Reflect v1

**Goal:** stand up the structured decision log (`decisions.jsonl`) and a rule-based flaw report. Wrap existing agents without refactoring them yet.

### Preconditions

- P1 complete (agents importable under `src/agents/`).
- A benchmark run-script exists under `src/training/benchmark.py`.

### Checklist

- [ ] Define `DecisionRecord` dataclass in `src/training/logging.py` matching the schema in design doc §6 (`run_id`, `game_id`, `turn`, `agent`, `state`, `choices`, `chosen`, `reasoning_tag`, `outcome`).
- [ ] Add a `@emit_decision` decorator or base-class hook on the existing agent action methods, so every current agent writes a record without touching its internal logic.
- [ ] Extend `benchmark.py` to accept a `--log-decisions` flag and write `data/runs/<run_id>/decisions.jsonl`.
- [ ] Write `src/training/reflect.py` implementing the v1 rule set from design doc §9:
  - Infeasible-bid tripwire.
  - Missed-call rule (P(call) < 0.3 when P(standing bid exists) < 0.1).
  - Low-EU choice rule.
  - Stale-bid-repetition cluster.
  - Rank-leak entropy check.
- [ ] `reflect` output → `data/runs/<run_id>/summary.md` with one table per rule, ranked by flag rate per (agent, opponent, ruleset).
- [ ] Wire W&B: every benchmark run pushes `metrics.json` + flaw-counts as a W&B run. Tag runs with git SHA and config hash.
- [ ] CLI: `python -m training reflect <run_id>` (Q6: on-demand, not post-every-benchmark).
- [ ] Smoke test: run a 50-game benchmark on the heuristic ladder; confirm a `decisions.jsonl` is produced, loadable via `pandas.read_json(..., lines=True)`, and that `reflect` produces a summary in <30s.
- [ ] Acceptance test: confirm the recent feasibility-filter fix (2026-04-24) means the infeasible-bid tripwire fires **zero times** on a clean run — if it does fire, that's a real bug.
- [ ] Write `TRAINING_PIPELINE_TESTS.md` scaffold for this module — test cases for each rule, schema validation, JSONL round-trip.
- [ ] Commit: "P3: decision logging + reflect v1 + W&B integration."

### Exit Criteria

- A 500-game benchmark produces a `decisions.jsonl` of the expected size.
- `reflect <run_id>` produces `summary.md` in < 30 s (design doc N3).
- W&B project shows the run with config + metrics.
- Infeasible-bid tripwire count is 0 on a clean heuristic-ladder run (sanity check of the existing fix).

### Session Handoff Notes

Explicitly note whether any rule was left un-implemented (blocker) vs. deferred (accepted scope cut). The rank-leak entropy rule depends on having a posterior exposed from the hand model — if that's not yet available in the heuristic-ladder wrap, defer it to P5 when modular interfaces land.

---

## P4 — OpenSpiel Adapter + Exploitability + Small-Game Oracle

**Goal:** register Liar's Poker as an OpenSpiel game; get exploitability and Kuhn/Leduc-style oracle validation for free.

### Preconditions

- P1 and P3 complete.
- `openspiel` installed (add to `pyproject.toml`).

### Checklist

- [ ] Write `src/interop/openspiel_adapter.py` implementing the OpenSpiel `Game` and `State` interfaces for exact-rules Liar's Poker at hand size 5 (Q2 target).
- [ ] Tests: round-trip 100 random games through both engines; assert identical legal-action sets and terminal rewards at every step.
- [ ] Register the game with `pyspiel.register_game(...)` so `pyspiel.load_game("liars_poker_exact")` works.
- [ ] Write `src/training/metrics/exploitability.py` wrapping OpenSpiel's `exploitability.exploitability()`. Works on any policy expressible as `tabular_policy` or via a callable adapter for neural policies.
- [ ] Extend `benchmark.py` to emit `exploitability` alongside win-rate in `metrics.json`. Every report card now shows both.
- [ ] Small-game oracle:
  - Define a reduced variant (e.g. 2-player × 2-card Liar's Poker, limited deck) as `liars_poker_kuhn`.
  - Solve it exactly using OpenSpiel's CFR+ to convergence.
  - Save the reference policy to `data/oracles/liars_poker_kuhn_policy.npz`.
  - Add a pytest `tests/oracles/test_kuhn_convergence.py` that runs CFR+ for N iterations and asserts exploitability → 0 (regression guard on the solver itself).
- [ ] Document the small-game variants in `docs-internal/design/small_games.md` with their exact rules and the reference policies' key behaviors.
- [ ] Write **ADR-005: OpenSpiel game ID and state encoding** — records the canonical ID and information-state encoding so future agents don't re-litigate it.
- [ ] Update `TRAINING_PIPELINE_TESTS.md` with: adapter round-trip tests, exploitability correctness test (CFR+ on Kuhn should converge to 0 exploitability).
- [ ] Commit: "P4: OpenSpiel adapter + exploitability metric + small-game oracle."

### Exit Criteria

- `pyspiel.load_game("liars_poker_exact")` works.
- Running `benchmark.py` emits an `exploitability` field for each agent.
- CFR+ on the Kuhn-sized variant converges to < 1e-3 exploitability in the Kuhn test.
- Round-trip game-play test between our engine and the adapter passes 1000 random games.

### Session Handoff Notes

OpenSpiel exploitability expects policies in a specific format. If a current agent's policy isn't directly convertible, wrap it in a `callable_policy` shim in this phase — don't defer the wiring to P5.

---

## P5 — Modular Agent Refactor

**Goal:** implement the `HandModel` / `BidPolicy` / `CallPolicy` interfaces (design doc §7) and migrate all existing agents onto them. Heuristic ladder is *wrapped only*, not touched (Q3).

### Preconditions

- P4 complete.
- OpenSpiel adapter stable for at least one full benchmark run.

### Checklist

- [ ] Define `src/agents/core/interfaces.py` with `HandModel`, `BidPolicy`, `CallPolicy`, `ScoredAction`, `Decision` Protocols.
- [ ] Define `src/agents/core/base.py` with the `Agent` composition class and the `act()` method that also emits decision records.
- [ ] Implement the shared `ExactRulesHandModel` in `src/agents/core/hand_models/exact_rules.py` — used by the frozen ladder and by CFR.
- [ ] **Heuristic ladder migration (frozen — no logic changes):**
  - [ ] Wrap `BlindEquilibriumAgent` as `Agent(ExactRulesHandModel, BlindBidPolicy, BlindCallPolicy)`.
  - [ ] Same for `ExactRulesConditionalAgent`, `ExactRulesBlindAgent`, `ExactRulesMixedAgent`.
  - [ ] Regression test: win-rates within ±2pp of P0 baseline (design doc §12 A1).
- [ ] **CFR / CFR+ migration:**
  - [ ] Wrap the solver's policy as `CFRBidPolicy` + `CFRCallPolicy` over the shared hand model.
  - [ ] Acceptance: pre- and post-migration self-play exploitability identical.
- [ ] **R-NaD migration:**
  - [ ] Expose the network as `RNaDBidPolicy` + `RNaDCallPolicy`. Hand-model posterior is emitted from the network's value head for logging.
  - [ ] Acceptance: `eval.py` reports an identical win-rate against the blind-equilibrium opponent.
- [ ] **Registry rewrite:** replace any `if/else` chains with `AGENT_REGISTRY + _AGENT_CLASS_MAP` (per existing memory).
- [ ] **Agent-card front-matter:** every agent file gets a YAML header block (algorithm summary, hyperparams, known failure modes, last-benchmarked date).
- [ ] Write **ADR-006: modular agent contract** — records that every future agent must implement the three Protocols.
- [ ] Extend decision logging so the `reasoning_tag` field is set by the agent's `BidPolicy`/`CallPolicy` explicitly (replaces the inferred tag from P3).
- [ ] Extend `TRAINING_PIPELINE_TESTS.md` with interface-conformance tests (each protocol has a test that exercises it with a mock).
- [ ] Commit: "P5: modular agent interfaces; heuristics wrapped, CFR/R-NaD migrated."

### Exit Criteria

- All existing agents implement `Agent(hand_model, bid_policy, call_policy)`.
- Heuristic-ladder win-rates within ±2pp of P0 baseline.
- CFR exploitability unchanged pre- vs. post-migration.
- R-NaD vs. blind-equilibrium win-rate unchanged pre- vs. post-migration.
- Adding a new agent is a one-file operation (design doc §12 A2).

### Session Handoff Notes

The migration order matters: heuristic → CFR → R-NaD, because each gets progressively harder to validate. If a phase-session runs out before R-NaD is done, stop at a clean commit boundary and hand off; don't leave partial agent migrations on `main`.

---

## P6 — Markdown Consolidation + Final Acceptance

**Goal:** collapse the markdown sprawl, validate every design-doc acceptance criterion, and produce a clean end-state.

### Preconditions

- P1–P5 all complete with green exit criteria.

### Checklist

- [ ] Move the following into `docs-internal/design/`:
  - [ ] `Liars poker/AGENT_DESIGN.md` → `docs-internal/design/legacy/AGENT_DESIGN.md`.
  - [ ] `Liars poker/agent/AGENT_CATALOG.md` → `docs-internal/design/legacy/AGENT_CATALOG.md`.
  - [ ] `Liars poker/IMPLEMENTATION_PLAN.md` → `docs-internal/design/legacy/IMPLEMENTATION_PLAN.md`.
  - [ ] `Liars poker/agent/TRAINING_OPTIMIZATION_PLAN.md` → `docs-internal/design/legacy/TRAINING_OPTIMIZATION_PLAN.md`.
  - [ ] `Liars poker/LITERATURE_SURVEY.md` → `docs-internal/design/literature_survey.md`.
  - [ ] `New-features.md` → `docs-internal/design/new_features_backlog.md`.
- [ ] Write `docs-internal/design/INDEX.md` linking to every design doc, ADR, and legacy artifact with a one-line description each.
- [ ] Remove the (now empty) `Liars poker/` directory entirely.
- [ ] Final top-level README pass: ≤150 lines, points to `docs-internal/design/INDEX.md` for anything deeper.
- [ ] Update `CLAUDE.md` to reflect the new layout (point at `src/`, `data/`, `paper/`).
- [ ] Run full validation against design doc §12 acceptance criteria:
  - [ ] **A1.** Heuristic-ladder win-rates within ±2pp of P0 baseline.
  - [ ] **A2.** Adding a new agent is demonstrably one file (do it as a smoke test — register a `CoinFlipAgent` and remove it in the same PR).
  - [ ] **A3.** `reflect <run_id>` completes in < 30 s on a 500-game run.
  - [ ] **A4.** Repo root has ≤ 3 `.md` files; `src/README.md` ≤ 100 lines.
  - [ ] **A5.** Every new output lives under `data/runs/<run_id>/`. Grep for writes outside this tree.
  - [ ] **A6.** A fresh session can reach productive work from `README.md` within 200 lines of docs.
- [ ] Write a final CHANGELOG entry marking end of refactor.
- [ ] Commit: "P6: consolidate docs; validate §12 acceptance; refactor complete."

### Exit Criteria

- Every item in design doc §12 is green.
- `docs-internal/design/INDEX.md` exists and is complete.
- `Liars poker/` directory no longer exists.
- Next session can open the repo, read README, and start research work — not refactor work.

---

## Cross-Cutting Workstreams

These run alongside phases, not as separate phases. They're called out so they don't get forgotten.

### Experiment Tracking (14.9 — starts in P0, continues)

- Every benchmark run: `wandb.init(project="liars-poker", config=<resolved_yaml>, tags=[git_sha])`.
- Every `reflect` run: attach `summary.md` + rule counts as a W&B artifact.
- Maintain a W&B report page with the "latest per-agent exploitability" auto-pulled from recent runs.

### ADR Log (14.11 — starts in P0, continues)

- One ADR file per irreversible decision. Format: `NNN-short-title.md` with sections Status / Context / Decision / Consequences.
- Written **as the decision is made**, not retrospectively.
- Current queue: ADRs 001–006 listed in phase checklists above.

### Process Hygiene (14.11)

- Auto-generated paper tables: starting in P5, `paper/` pulls numbers from `data/runs/<canonical-run>/metrics.json` rather than hardcoding.
- Agent-cards: land during P5 migration; never add an agent without one afterward.
- Reproducibility CI: a GitHub Action runs a tiny benchmark (20 games) on every PR and diffs `metrics.json` against a frozen reference. Failures block merge. Land this in P5.

### Testing Checklist Doc (`TRAINING_PIPELINE_TESTS.md`)

The third artifact per `New-features.md` §2 is written **incrementally**, not in one sitting:

- **P3:** adds decision-logging schema tests + reflect rule tests.
- **P4:** adds OpenSpiel adapter round-trip tests + exploitability convergence tests.
- **P5:** adds interface-conformance tests for HandModel / BidPolicy / CallPolicy.
- **P6:** final pass — verify every §12 acceptance criterion has a test behind it.

Each phase's commit should include the relevant test additions.

---

## User-Action Items (blocking)

Items where the user must do something outside of Claude's tools. Each phase cannot exit until the corresponding item is green.

| Phase | Item | Where |
| --- | --- | --- |
| P0 | Create W&B account + project `liars-poker`; save API key | wandb.ai |
| P0 | Approve dependency additions (`hydra-core`, `wandb`, `openspiel`, …) | review `pyproject.toml` diff |
| P2 | Disable GH Pages in repo settings | GitHub Settings → Pages → Source: None |
| P4 | Confirm OpenSpiel install succeeds on your machine (it has C++ build requirements) | `pip install open_spiel` |
| P5 | Confirm you're OK with the one-shot checkpoint move (Q8) — old paths will break | one-time review |

---

## Out of Scope for This Plan

These are tracked in design doc §14.12 "Later" and "Paper phase" buckets. Each gets its own design doc when its turn comes; they do **not** appear in this checklist:

- 14.3 Hand abstraction / bucketing
- 14.5 ReBeL / continual resolving
- 14.6 PSRO
- 14.7 JAX engine rewrite
- 14.8 Human-play dataset
- 14.10 Theory + empirics pairing

If a session finds itself doing any of these, stop and spin up a separate design doc first (per `New-features.md` §2).

---

## Session Handoff Template

Copy this block at the end of every session where plan-phase work was done. It lets the next session start cold without re-reading transcripts.

```markdown
### Session Handoff — <date>

**Phase:** P<n>
**Phase status:** <in-progress / complete / blocked>
**Commits in this session:** <git log --oneline since last session>
**Checklist items ticked:**
 - [x] ...
**Blocked items:** <if any — owner, reason>
**Next session should start with:** <literal first action>
**Gotchas discovered:** <anything surprising worth remembering>
**Time spent:** <ballpark — helps re-estimate future phases>
```

---

## Current Phase: **P1 — `src/` Skeleton + Game Package + Unified Tests**

*(update this pointer as phases complete)*

---

### Session Handoff — 2026-04-24 (P0)

**Phase:** P0 — Preflight
**Phase status:** complete (with one deferred item, accepted)
**Commits in this session:** `e9057c0` — *P0: preflight — ADR log, pyproject.toml, CHANGELOG*
**Checklist items ticked:**

- [x] `pyproject.toml` at repo root with dev + optional `openspiel` extras
- [x] `docs-internal/design/adr/README.md` — ADR format spec
- [x] ADR-001 Frontend retirement
- [x] ADR-002 OpenSpiel adoption
- [x] ADR-003 Infrastructure-first project posture
- [x] `CHANGELOG.md` seeded with v0 marker + P0 entry
- [x] W&B account created (entity `conair92-university-of-maryland`, project `liars-poker` auto-created on first `wandb.init`)
- [x] Plan-doc P0 checklist updated with status

**Blocked items:** none. The baseline-benchmark snapshot was *deferred*, not blocked — explicit user decision: agents will be re-benchmarked post-refactor anyway, so §12 A1 will be evaluated against whichever benchmark is current at P5 rather than against a frozen P0 snapshot. Recorded as `[!]` in the P0 checklist and as a `### Deferred` block in `CHANGELOG.md`.

**User-side prerequisites for P3 (not P1):** verify `wandb login` works locally before P3 starts. P1 itself does not invoke W&B.

**Next session should start with:** open `TRAINING_PIPELINE_PLAN.md` to the P1 section and execute the directory moves *one logical group at a time*, committing after each so a regression can be bisected to a single move. Suggested first move: LaTeX sources `Liars poker/*.tex,*.bib,figures/` → `paper/` (lowest-risk, no Python imports involved).

**Gotchas discovered:**

- Spell-check warnings on `wandb`, `openspiel`, `pyproject` are pre-existing style throughout the plan doc; ignore.
- The user's W&B entity contains hyphens (`conair92-university-of-maryland`); use the literal string in `wandb.init(entity=...)` calls. Stored in `memory/reference_wandb_entity.md`.
- Memory entry `reference_wandb_entity.md` is the single source of truth for the W&B entity/project pair — update it there, not in code, if either changes.
- The benchmark CLI lives at `Liars poker/agent/benchmark.py`. P1 moves it; rerun any baseline command via the new `src/` path afterward.

**Time spent:** short — one session, no execution work, mostly prose authoring.

---

### Session Handoff — 2026-04-25 (P1, in progress: 3/11 commits done)

**Phase:** P1 — `src/` Skeleton + Game Package + Unified Tests
**Phase status:** in-progress — paused after commit 3 of an 11-commit plan
**Commits in this session:**

- `e225ff3` — *P1.1: move LaTeX sources to paper/* (incl. `paper/README.md`, root `.gitignore`)
- `091a3ad` — *P1.2: move probability scripts to src/training/probs/*
- `4783353` — *P1.2 fixup: include path-setup edits in moved probs scripts* (recovered edits accidentally left unstaged on top of a `git mv` — see Gotcha)
- `c7cf573` — *P1.3: move agent/game/ to src/game/*

**Pre-P1 baseline (recorded for regression comparison):** 81 tests passed, 1 pre-existing failure (`agent/game/tests/test_bids.py::test_bid_count` — `NUM_ACTIONS == NUM_BIDS + 1` was 110+1 expected vs 112 actual; HH_ACTION adds an extra action and the assertion was never updated). torch tests in `agent/rnad/tests/` are skipped in this conda env (no torch installed). Full suite: ~8 min, dominated by CFR+ tests.

**Move plan (committed-by-commit, agreed with user; option (A) for the registry):**

1. ✅ LaTeX → `paper/`
2. ✅ Prob scripts → `src/training/probs/` (JSONs deferred)
3. ✅ `agent/game/` → `src/game/` (incl. its tests/)
4. **NEXT:** `agent/baseline/` → `src/agents/heuristic/` + `_DATA_DIR` re-pointed; `_PAPER_DIR` paths replaced. Move `blind_equilibrium.json`, `cfr_1v1.json`, `cfr_1v1_run/` along with the consumers (or split out as commit 5.5 — see deferred item).
5. `agent/rnad/` → `src/agents/learned/rnad/` + same pattern; warm_start consumes many JSONs.
6. Move all `agent/data/*.json` (probability tables) → `data/probs/` and `cfr_1v1_run/` → `data/runs/`. Update remaining consumers (web/backend if still in place).
7. `src/agents/search/` placeholder (current `agent/search/` is empty on disk).
8. `agent/checkpoints/` → `data/checkpoints/`.
9. Tests consolidation: all `*/tests/` → unified `tests/` tree at repo root mirroring `src/`.
10. **Extract `AGENT_REGISTRY` from `web/backend/agents.py` → `src/agents/registry.py`** so `benchmark.py` can keep working after P2 archives `web/`. (Option A, agreed with user.)
11. `benchmark.py` → `src/training/benchmark.py`.
12. Rewrite `pyproject.toml` to point at `src/` and `tests/`; `ruff check --fix` for stragglers.
13. `src/README.md` (≤100 lines) + top-level `README.md` rewrite. Final commit.

**Key gotcha — git mv + edit interaction:** `git mv A B` followed by editing `B` does NOT auto-stage the edit. The rename is staged; the subsequent edit sits in the working tree as an unstaged modification on top. `git add B` again to capture it. This bit me on commit 2 (the script path-setup edits) and required a fixup commit. **For each upcoming commit:** after `git mv` + edits, always run `git status` and confirm the diff before committing — not just that the rename is recorded.

**Pyc hygiene:** `.gitignore` now ignores `__pycache__/` going forward, but a lot of stale `.pyc` files were tracked in the repo. I've untracked the ones I encountered (under `Liars poker/` and `Liars poker/agent/data/`), but more remain in `Liars poker/agent/baseline/`, `agent/rnad/`, `agent/web/backend/`. Each commit picks them up as deletions. Don't fight it — let them ride along with the relevant package move.

**Compatibility scaffolding (will be removed by end of P1):** none added. Instead, downstream consumers (`baseline/`, `rnad/`, `web/`) currently have **broken imports** to `agent.game.*` and `poker_math_exact`; this is intentional and bisectable. They get fixed in commits 4–5 and 9. Per-commit smoke tests scope to packages already moved, not to the whole tree. Full test suite + benchmark run gates the final commit (12).

**Path-setup convention used in moved scripts:**

```python
HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # adjust depth per file
```

`REPO_ROOT` is the repo root (containing `Liars poker/`, `src/`, `paper/`). All output paths now hard-coded as absolutes from `REPO_ROOT` so cwd doesn't matter.

**Next session should start with:** open this handoff, then begin commit 4 — move `Liars poker/agent/baseline/` → `src/agents/heuristic/`. Files involved:

- `blind_equilibrium.py`, `cfr_1v1.py`, `cfr_1v1_fast.py`, `cfr_1v1_overnight.py` (and their `tests/` subdir).
- Each has a `_PAPER_DIR` block at the top adding `Liars poker/` and `Liars poker/agent/` to sys.path. Replace with: `_PROBS_DIR = .../src/training/probs/` for poker_math_exact, and `_SRC_DIR = ../..` for `from game.bids import ...`.
- Each has `_AGENT_DIR/data/<name>.json` for cache files. Either re-point to `<repo>/Liars poker/agent/data/<name>.json` (defer JSON move) or to `<repo>/data/probs/<name>.json` (move JSONs in same commit). Recommend the second — tighter atomicity.
- Tests under `agent/baseline/tests/` import from `agent.baseline.*` and `agent.game.*` — rewrite to `agents.heuristic.*` and `game.*`.
- Run a smoke check after the move: `cd src && python -m pytest agents/heuristic/tests --no-header -q --tb=line`.

**Time spent:** ~1.5 hours; 4 commits. Pace estimate: a single contiguous session can probably finish 4 more commits (through commit 7 — checkpoints). Then a second session for tests-tree consolidation, registry extraction, benchmark move, pyproject, READMEs, and full-suite verification. P1 will likely take 3 sessions total, not 1–2 as the plan estimated.

---
