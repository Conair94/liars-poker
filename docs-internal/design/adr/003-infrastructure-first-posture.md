# ADR-003: Infrastructure-first project posture

- **Status:** Accepted
- **Date:** 2026-04-24
- **Supersedes:** —

## Context

The project has reached a point where novel research output (state-of-the-art Nash agents for card-based Liar's Poker, paper-quality results) is gated by infrastructure quality, not by ideas. Symptoms:

- Bugs that silently invalidated experiments survived for weeks: ruleset params dropped at the `eval.py` layer; the High Hand button hidden in JS; CFR's `include_hh` kwarg silently defaulting to no-HH mode.
- Long-running training runs finished without W&B tracking, ADRs, or run-config snapshots, making post-hoc analysis dependent on git archaeology.
- Multiple parallel training scripts and overnight jobs accumulated under `Liars poker/` with no shared logging schema, no exploitability metric, and no regression baseline.

We had to choose between two postures for the next phase of work:

- **Paper-first:** push immediately on novel content (hand abstraction, ReBeL, PSRO) and patch infrastructure reactively as it breaks. Optimizes for short-term research velocity.
- **Infrastructure-first:** pause novel-content work; first land the refactor (`src/` layout, modular `Agent` interface, decision logging, exploitability, OpenSpiel adapter, ADR log, W&B). Then resume research on a stable substrate.

## Decision

Adopt an infrastructure-first posture for the duration of the `TRAINING_PIPELINE_PLAN.md` refactor (estimated ~9–12 sessions, P0–P6). Until P6's exit criteria are green, no work on §14.12 "Later" items (hand abstraction, ReBeL, PSRO, JAX rewrite, human-play dataset). The CFR/CFR+ and R-NaD redesigns themselves are scope-fenced to wrapping existing implementations behind the new modular interfaces — no algorithmic changes during the refactor.

## Consequences

**Positive:**

- Every later research push lands on a substrate where: every decision is logged with `run_id`, every benchmark emits exploitability, every irreversible decision is recorded as an ADR, and every run is in W&B with config + metrics.
- Reduces silent-bug surface by an order of magnitude (decision logging + exploitability + reflect rules catch failures the heuristic ladder cannot).
- The paper's eventual results table can be auto-generated from `data/runs/<canonical>/metrics.json`, not hand-typed.

**Negative:**

- Defers visible research progress by ~9–12 sessions. We accept this; the cost of running a flawed experiment that has to be re-run is higher than the cost of the refactor.
- During the refactor, agent quality does not improve. The 2026-04-24 baseline (ExactCond 85.8%, CFR Nash 49.2% vs. ladder) is the frozen reference until P5 completes.
- New infrastructure (Hydra, W&B, OpenSpiel) adds learning surface for a single-author project. Mitigated by introducing each tool exactly when its phase needs it, not all at once.

## Execution

The full refactor is tracked in `TRAINING_PIPELINE_PLAN.md`. Resumption of "Later" research items is gated on P6 exit criteria, not on calendar time.
