# ADR-001: Frontend retirement

- **Status:** Accepted
- **Date:** 2026-04-24
- **Supersedes:** —

## Context

The repository ships a static GitHub Pages frontend (`docs/` plus assets generated from `Liars poker/agent/web/`) that lets visitors play Liar's Poker against the trained agents in-browser. While useful as a demo, the frontend has become a maintenance liability:

- Every backend change to agents, rulesets, or bid-space semantics required a parallel JS-side fix-up. The High Hand button being silently hidden until 2026-04-24 is a recent example of the cost.
- Current agents (CFR Nash 49.2% vs. ladder, ExactCond 85.8%) are not strong enough that a public-facing demo provides scientific value commensurate with the maintenance.
- Project scope has shifted toward research-grade training (CFR+ self-play, R-NaD, OpenSpiel-based exploitability). The frontend is a distraction from that work, not a contributor to it.

Options considered:

- **A.** Keep the frontend, freeze JS feature work until agents improve. Rejected — drift continues silently.
- **B.** Consolidate the duplicated game logic into a single source of truth (originally proposed in `TRAINING_PIPELINE_DESIGN.md` §8). Rejected — pays a refactor cost for an artifact that has no current scientific use.
- **C.** Retire the public frontend; archive the code on disk for later resurrection. **Selected.**

## Decision

Retire the public web demo. Take the GitHub Pages deployment offline and move `docs/` and `Liars poker/agent/web/` to `archive/web-2026-04/`. All future work is local-development-only until agents are strong enough to justify re-hosting.

## Consequences

**Positive:**

- Eliminates the dual-edit burden across Python and JS.
- Removes a class of silent-divergence bugs (rulesets, bid space, button visibility).
- Frees attention for the research pipeline.

**Negative:**

- The `https://<user>.github.io/liars-poker/` URL goes 404. Anyone with a bookmark loses access.
- Resurrecting a public-facing demo later is non-trivial: the JS code in `archive/` will have drifted from the Python source of truth and needs to be rewritten against the modular `Agent` interface defined in `TRAINING_PIPELINE_DESIGN.md` §7.
- Loses a useful "explain the project to a non-technical reader" handle; we accept this trade.

## Execution

Execution of this decision is tracked in P2 of `TRAINING_PIPELINE_PLAN.md` and recorded in ADR-004 once complete.
