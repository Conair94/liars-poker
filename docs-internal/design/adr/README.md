# Architecture Decision Records (ADRs)

This folder records irreversible (or hard-to-reverse) design decisions made during the project. Each ADR is one Markdown file capturing the context at the time of the decision so future contributors do not re-litigate it.

## Format

Filename: `NNN-short-kebab-title.md` (zero-padded, monotonically increasing).

Each ADR contains:

- **Title** — `# ADR-NNN: <decision in one phrase>`
- **Status** — `Proposed` / `Accepted` / `Superseded by ADR-XXX` / `Deprecated`
- **Date** — ISO date of the decision
- **Context** — why the decision is needed; constraints, prior state, options considered
- **Decision** — what was chosen, in one sentence then a short paragraph
- **Consequences** — both positive and negative downstream effects; what gets harder

## Conventions

- Write the ADR at the moment of the decision, not retrospectively.
- Never edit an accepted ADR's body. To change a decision, write a new ADR that supersedes it and update the old one's status to `Superseded by ADR-XXX`.
- ADRs are append-only history. Day-to-day design notes belong in `docs-internal/design/`, not here.
- Keep each ADR ≤ 1 page. If it grows longer, the decision is probably actually multiple decisions; split them.

## Index

| # | Title | Status | Date |
| --- | --- | --- | --- |
| 001 | Frontend retirement | Accepted | 2026-04-24 |
| 002 | OpenSpiel adoption | Accepted | 2026-04-24 |
| 003 | Infrastructure-first project posture | Accepted | 2026-04-24 |
