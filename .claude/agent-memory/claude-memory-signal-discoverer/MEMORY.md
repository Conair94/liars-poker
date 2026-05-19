# Signal Discoverer Agent Memory

## Scan History

### 2026-04-20 — Session scan (no recall script available; analyzed KEY_EVENTS from caller)
- Sessions analyzed: 1 session (2026-04-20 Agent Zoo build session)
- Candidates surfaced: FILL_GAP: 3, UPDATE: 1, NOISE: 0, CONTRADICT: 0
- Candidates written to project memory:
  - FILL_GAP → `project_benchmark_zoo.md` (benchmark infrastructure, CLI, add-agent checklist)
  - FILL_GAP → `project_benchmark_results.md` (baseline win rates, ExactRulesConditional 81% vs CFR Nash)
  - FILL_GAP → `feedback_agent_registry_pattern.md` (build_agent() factory pattern, no if/else chains)
  - UPDATE → `project_milestones.md` (already updated in-session; confirmed correct, no edits needed)
- All candidates accepted (written without user rejection)

### 2026-04-24 — Session scan (KEY_EVENTS from caller; recall script available but not needed)

- Sessions analyzed: 1 session (2026-04-24 eval.py fix + JS frontend fix session)
- Candidates surfaced: FILL_GAP: 4, UPDATE: 1 (benchmark_results already current), NOISE: 1 (Signal 3, UI-only), CONTRADICT: 0
- Candidates written to project memory:
  - FILL_GAP → `feedback_eval_ruleset_params.md` (new_match() must receive exact_rules + high_hand throughout eval chain)
  - FILL_GAP → `feedback_js_frontend_ruleset_params.md` (startGame() must derive both exactRules and highHand from agent group)
  - FILL_GAP → `feedback_cfrsolver_param_name.md` (CFRSolver uses include_hh not high_hand)
  - FILL_GAP → `feedback_torch_load_weights_only.md` (PyTorch 2.6: weights_only=False for config-bundled checkpoints)
  - UPDATE → `project_benchmark_results.md` (already contained Run 3 post-eval-fix data; no edit needed)
- Signal 3 (round result message wording) classified NOISE — UI string branching, does not generalize
- Signal 5 (benchmark countup = n=2 matches CFR training) classified project fact, confirmed in existing benchmark_results.md
- All candidates accepted (written without user rejection)

### 2026-05-11 — Session scan (13 sessions via recall script, AR-2 P1–P5)

- Sessions analyzed: 13 (2026-04-29 → 2026-05-11), covering AR-1 sweep results + AR-2 Phases 1–5
- Candidates surfaced: FILL_GAP: 6, UPDATE: 0, NOISE: ~3 (per-phase ephemera), CONTRADICT: 0
- Candidates returned to caller as punch list (caller chooses what to write)
- Key themes: (1) user accepts judgment-call summaries before impl; (2) recurring small-behavior-deviation disclosure pattern; (3) defunct R-NaD import failure treated as known noise across every session; (4) memory-consolidation reminders the assistant keeps emitting

### 2026-05-18 — Session scan (10 sessions via recall script, AR-2 P7)

- Sessions analyzed: 10 (2026-05-11 → 2026-05-18), AR-2 Phase 7 pilot → sweep → factory wiring
- Candidates surfaced: FILL_GAP: 6, UPDATE: 0, NOISE: 0, CONTRADICT: 0
- Candidates returned to caller as punch list (caller chooses what to write):
  - `feedback_gitignore_run_artifacts_first` — gitignore *.npz/cfr_deals/ BEFORE first sweep commit; reset --mixed beats history rewrite when commits are local-only
  - `feedback_sweep_harness_flag_format` — sweep harness can't emit store_true flags; use --key value or invert defaults
  - `project_ar2_phase7_complete` — Phase 7 closed 96be92d, N=10k chosen, Phase 8 acceptance gate next
  - `reference_ar2_sweep_summary` — invocation pattern with full sweep-id (not just timestamp)
  - `feedback_ar2_distillation_row_count` — ~218 rows/deal, design spec was ~100× off, no reach_prob column
  - `feedback_extract_learnings_new_session` — run /extract-learnings in fresh session (user-confirmed 2026-05-18)
- Key themes: (1) git-history blob incident is the highest-signal gotcha; (2) Phase 7 closure is project-state worth pinning for Phase 8 resume; (3) user keeps confirming the design-first/deviation-disclosure pattern
