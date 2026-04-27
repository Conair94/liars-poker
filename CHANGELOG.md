# Changelog

All notable changes to this project are recorded here. The repository uses a single linear changelog rather than per-release tags during the refactor (P0–P6); release tags resume after P6.

Format roughly follows [Keep a Changelog](https://keepachangelog.com/). Dates are ISO (YYYY-MM-DD).

## [Unreleased]

### Added

- **P5-3 (2026-04-27)** — Reflect-rule deferred items closed (per-choice `p` plumbed end-to-end).
  - `src/training/decision_capture.py`: `LoggingAgentWrapper` now queries `agents.policy.action_probs` *before* invoking `choose_action`, so `Choice.p` is populated for all agents (mixed agents emit their full distribution; deterministic agents emit a one-hot). RNG advance preserved by ordering the query first.
  - `src/training/reflect.py`: implements two of the three rules previously deferred to P5.
    - `rule_missed_call` — flags turns where `P(call) < 0.30` while the standing bid is Pair-or-stronger.
    - `rule_rank_leak` — opening-bid concentration > 60% within one private-rank bucket while < 10% across the others (≥30 openings sampled per bucket).
    - The remaining rule (low-EU choice) needs a value model and stays deferred; surface schema unchanged.

- **P5-2 Phase D (2026-04-27)** — Per-agent exploitability wired into `benchmark.py`.
  - `--lbr` and `--subgame` flags emit per-agent metrics under `agent_exploitability` in `metrics.json`, matching the design doc §2c schema.
  - `--exploitability-deals` (default 50) and `--lbr-depth` (default 2) bound cost; `--exploitability-deals 200+` for paper-grade runs.
  - Eligible agents = those in `--groups` whose ruleset is exact + HH (the metric adapter is `python_liars_poker_exact`).

- **P5-2 Phase C (2026-04-27)** — Sampled subgame exploitability (exact via memoized tree DP).
  - `src/training/metrics/subgame_exploitability.py`: per deal, computes exact NashConv on the post-deal bidding subgame via memoized DP keyed on `(current_bid_idx, current_player)` (~222 abstract states). Assumes the agent's policy is approximately history-blind within the subgame; canonical minimum-history `MatchState` constructed for the agent query.
  - Avoids OpenSpiel's CFR+ entirely (BR vs. agent yields exact NashConv directly without solving for Nash, since chance is collapsed). Faster and exact compared to the design doc's CFR+-on-subgame plan; documented in module docstring.
  - `tests/training/metrics/test_subgame_exploitability.py`: 6/6 pass.

- **P5-2 Phase B (2026-04-27)** — Local Best Response (LBR) exploitability metric.
  - `src/training/metrics/lbr.py`: `lbr_exploitability(agent, deals, depth, ...)` computes a depth-bounded best-response lower bound on exploitability against the registered single-round 52-card adapter. Per-seat results plus 95% gaussian CI. Pruning modes: `policy_support` (default, ε=0.01) and `all`.
  - `tests/training/metrics/test_lbr.py`: pointwise BR-dominates-agent invariant (with `all` candidates), reward-range bounds, summary-shape, and forced-terminal regression tests. 5/5 pass.
  - `RandomAgent`: documented why no `action_probs` override (one-hot fallback yields tractable single-MC-rollout estimates; uniform-110 expansion is intractable on the full game).

- **P5-1 (2026-04-26)** — High Hand wired into the OpenSpiel adapter.
  - `src/interop/openspiel_adapter.py`: `python_liars_poker_exact` now exposes HH at action index 111 (matching engine `HH_ACTION`); `_FULL_NUM_ACTIONS = NUM_BIDS + 2 = 112`. New `_resolve_high_hand` mirrors `MatchState._resolve_high_hand` for the single-round projection: declarer wins iff pool's normalized best hand exactly equals the standing bid; ±1 zero-sum reward.
  - `tests/interop/test_openspiel_roundtrip.py`: added `test_full_adapter_roundtrip_1000_games_high_hand` (parity against engine with `high_hand=True`) and two hand-crafted HH resolution tests. Existing no-HH parity test now filters HH from the adapter's legal set.
  - ADR-005 amended: HH-enabled action layout and parity statement.
  - `docs-internal/design/p5_design.md`: P5 design doc + checklist (#1 done; #2 modular agent interface and #3 reflect rules scoped for follow-up sessions).
  - Closes the P4 follow-up #1; unblocks P5-#2 (per-agent exploitability projection).

- **P4 (2026-04-26)** — OpenSpiel adapter + exploitability metric + small-game oracle.
  - `src/interop/openspiel_adapter.py`: registers two games via `pyspiel.register_game`.
    - `python_liars_poker_kuhn` — 2-player × 1-card × 3-rank Kuhn-sized variant; tractable for tabular CFR.
    - `python_liars_poker_exact` — single-round, 52-card, exact-rules adapter (configurable `num_players`, `hand_size`). HH **disabled** in this iteration; tagged TODO(P5) since all future games use HH.
  - `src/training/metrics/exploitability.py`: wraps OpenSpiel's tabular exploitability; `kuhn_cfr_plus_solve()` + `callable_to_tabular()` helpers.
  - `src/training/metrics/build_kuhn_oracle.py`: persists the Kuhn reference policy to `data/oracles/liars_poker_kuhn_policy.npz` (24 infosets, exploitability 6.7e-5 at 5k iters).
  - `tests/interop/test_openspiel_roundtrip.py`: 1000 random games of legal-action + terminal-return parity vs. `game.engine`; Kuhn truth/lie behavior tests.
  - `tests/oracles/test_kuhn_convergence.py`: solver regression guard (CFR on Kuhn → <1e-3 in 1000 iters; monotone improvement).
  - `benchmark.py`: `--exploitability [--exploitability-iters N]` populates `output["oracle_exploitability"]` in metrics.json; per-pair `exploitability_a`/`exploitability_b` slots present (null until P5 supplies projection layer).
  - ADR-005 (OpenSpiel game ID and state encoding) + `docs-internal/design/small_games.md` (small-game variant catalog).
  - Plan + tests doc updated (`Liars poker/TRAINING_PIPELINE_PLAN.md`, `Liars poker/TRAINING_PIPELINE_TESTS.md`).
  - **Known follow-ups:** (1) wire HH into adapter before P5 training runs; (2) the adapter is single-round, so OpenSpiel exploitability is per-round, not per-match — full-match exploitability deferred to P6 if needed.

- **P3 (2026-04-26)** — Decision logging + reflect v1 + W&B integration.
  - `src/training/logging.py`: `DecisionRecord` dataclass + `DecisionLogger` JSONL writer (design §6 schema).
  - `src/training/decision_capture.py`: `LoggingAgentWrapper` — non-invasive per-turn recording for any agent exposing `choose_action(state)`. No agent code touched.
  - `src/training/reflect.py`: v1 rule engine — infeasible-bid tripwire (0 on clean run ✓), stale-bid-repetition proxy. Three rules deferred to P5 (need per-choice `p`/`eu`): missed-call, low-EU, rank-leak. CLI: `python -m training.reflect <run_id>` → `summary.md`.
  - `benchmark.py`: `--log-decisions` (writes `data/runs/<run_id>/decisions.jsonl` + `metrics.json`), `--wandb` (pushes win-rate matrix + flaw counts to W&B, tagged with git SHA + config hash), `--run-name`, deterministic `run_id` per design §5.
  - `Liars poker/TRAINING_PIPELINE_TESTS.md`: test-case scaffold for all new modules.
  - Pending: full 500-game Exit-Criteria run with `--log-decisions --wandb` (smoke test passed at 5 games/pair; formal closure needs 500).

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
