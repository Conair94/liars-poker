# Changelog

All notable changes to this project are recorded here. The repository uses a single linear changelog rather than per-release tags during the refactor (P0–P6); release tags resume after P6.

Format roughly follows [Keep a Changelog](https://keepachangelog.com/). Dates are ISO (YYYY-MM-DD).

## [Unreleased]

### Added

- **P1 (2026-04-25)** — `src/` skeleton + game package + unified test tree. Pure file moves, zero behavior change. 13 commits; full test suite green (88 passed, 1 pre-existing `test_bid_count` failure, 2 slow tests deselected by default). See `TRAINING_PIPELINE_PLAN.md` P1 handoff for per-commit detail.
- **P2 foundations (2026-04-25)** — `archive/README.md` (frozen-on-disk policy), ADR-004 (frontend archived on disk). Actual `git mv` of `docs/` and `Liars poker/agent/web/` → `archive/web-2026-04/` is gated on user disabling GitHub Pages.
- `pyproject.toml` at repo root with dev dependencies (`hydra-core`, `wandb`, `pytest`, `pytest-cov`, `ruff`, `mypy`) and an optional `openspiel` extra.
- W&B integration prep: entity `conair92-university-of-maryland`, project `liars-poker` (auto-created on first `wandb.init` call). Account verified 2026-04-24.
- `docs-internal/design/adr/` — Architecture Decision Records.
  - ADR-001: Frontend retirement.
  - ADR-002: OpenSpiel adoption.
  - ADR-003: Infrastructure-first project posture.
- `CHANGELOG.md` (this file), seeded as the v0 baseline marker.

### Deferred

- P0 baseline benchmark snapshot under `data/runs/20260424-baseline-premigration/` — deferred per user direction (data will be regenerated naturally during later phases; no need to freeze a pre-refactor snapshot when most agents will be re-evaluated post-refactor anyway). The §12 A1 acceptance criterion (heuristic-ladder win-rates within ±2pp of baseline) will be evaluated against whichever benchmark is current when P5 lands, not against a P0 snapshot.

## v0 — 2026-04-24

Marker for "everything before the refactor began." See `git log --until=2026-04-24` for state. Notable inheritance:

- Backend: in-house Liar's Poker engine, CFR+ fast solver, R-NaD trainer (CPU, exact rules, batched), heuristic ladder of ExactRulesConditional / ExactRulesBlind / ExactRulesMixed / BlindEquilibrium agents.
- Benchmark (2026-04-24, post eval-fix): ExactCond 85.8%, CFR Nash 49.2% vs. ladder.
- Frontend: live GitHub Pages demo at `https://<user>.github.io/liars-poker/` (slated for retirement in P2).
- Paper: `Liars poker/Liars-poker.tex` with marginal + conditional probability tables for n=5..25.
