# Legacy Design Docs

Historical record of the project's pre-refactor (P0–P5) design and planning
documents. **Nothing here is load-bearing for current code** — it is preserved
for context on prior decisions, the refactor plan, and earlier experiments.

For current state, see the parent [`docs-internal/design/`](..) directory:

- ADRs in [`adr/`](../adr) record irreversible decisions.
- Active design docs (`p5_design.md`, `p5_2_exploitability.md`,
  `small_games.md`) describe in-flight or recently-landed work.

## What's here

| File | Origin | Why archived |
|---|---|---|
| `paper_bootstrap.md` | `Liars poker/CLAUDE.md` | Paper-side script + game-rule reference. The script paths reference the pre-P1 layout (`Liars poker/`); current scripts live under `src/training/probs/`. The "Official Game Rules" and card encoding sections are still accurate but are now also documented in [`src/game/bids.py`](../../../src/game/bids.py) and the ADRs. |
| `AGENT_DESIGN.md` | `Liars poker/AGENT_DESIGN.md` | Original R-NaD-centric agent design. Useful as background reading for the upcoming agent rewrite (`New-features.md` §3.4–3.5). |
| `LITERATURE_SURVEY.md` | `Liars poker/LITERATURE_SURVEY.md` | M0 deliverable — survey of card-game RL literature. Will be revisited when designing the next-gen agent. |
| `IMPLEMENTATION_PLAN.md` | `Liars poker/IMPLEMENTATION_PLAN.md` | 2026-04-24 mid-refactor plan. Items 2 & 3 done; item 1 (CFR+ stall) is superseded by P5-#2 honest exploitability metrics. |
| `TRAINING_PIPELINE_DESIGN.md` | `Liars poker/` | Refactor design rev. 3 — drove P0–P5. |
| `TRAINING_PIPELINE_PLAN.md` | `Liars poker/` | Phase-by-phase checklist for P0–P6 with session handoffs. |
| `TRAINING_PIPELINE_TESTS.md` | `Liars poker/` | Per-module test scaffold notes for P3–P5. |
| `AGENT_CATALOG.md` | `Liars poker/agent/` | Pre-refactor agent zoo description. Superseded by the live `AGENT_REGISTRY` in [`src/agents/registry.py`](../../../src/agents/registry.py). |
| `agent_dir_readme.md` | `Liars poker/agent/README.md` | Pre-refactor `agent/` directory README. Superseded by [`src/README.md`](../../../src/README.md). |
| `TRAINING_OPTIMIZATION_PLAN.md` | `Liars poker/agent/` | 2026-04-22 optimization sprint plan. All items implemented. |
| `New-features.md` | repo root | 2026-04-23 todo list that drove P1–P5. Items 1, 2, 3.1, 3.2, 3.3 are done; 3.4 (Nash/CFR rewrite) and 3.5 (modular agent stack) are the next major project phase. |

## Reuse policy

Treat these files as read-only. If something here is still useful, **lift the
specific section into a current doc** (an ADR, a P-series design doc, or a
module docstring) rather than editing in place — that keeps the legacy archive
a clean snapshot of where the project came from.
