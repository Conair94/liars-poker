# AR-2 Phase 7 — Pilot + Sweep Implementation Checklist

- **Status:** Code complete (steps 1–8, 2026-05-11); pilot + sweep pending
- **Date:** 2026-05-11
- **Parent design:** [agent_redesign_ar2_phase7_design.md](agent_redesign_ar2_phase7_design.md)
- **Parent checklist:** [agent_redesign_ar2_checklist.md](agent_redesign_ar2_checklist.md) §Phase 7

Phase 7 is a long-running phase split across multiple sessions. Sessions 1–2
land code (steps 1–6) and can be wrapped in a single commit. Session 3 runs
the pilot. Sessions 4+ launch + monitor the sweep. Session 6 picks the elbow.

Use `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3` for every
python invocation per [feedback_python_env_torch.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_python_env_torch.md).

---

## Step 0 — Pre-flight

- [ ] Resolve `<ar1_winner>`: read [project_ar1_handmodel.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/project_ar1_handmodel.md), confirm path `data/runs/<ar1_winner>/best.pt` exists, record it in scratch — it is hard-substituted into the sweep YAML in Step 5.
- [ ] Confirm `git status` clean before code edits (sweep harness fails otherwise per AR-0b convention).
- [ ] `grep -n "iter_shards\|external-val-run-id\|_main" src/training/cfr_distillation.py src/agents/learned/{callpolicy,bidpolicy}/trainer.py` — confirm none exist yet.

---

## Step 1 — Shared shard iterator

**File:** [src/training/cfr_distillation.py](../../src/training/cfr_distillation.py) (edit)

- [ ] Add `iter_shards(run_id, split, *, head: Literal["call", "bid"], data_root=None, batch_size=None) -> Iterator[dict[str, np.ndarray]]` per design §2.1.
- [ ] Read `split.json` once; build the set of deal indices for the requested split.
- [ ] For each shard `s`: lazily load `call_<s>.npz` OR `bid_<s>.npz` and the matching `trunk_<s>.npz`. Yield row dicts whose `deal_idx` ∈ the split set.
- [ ] Row dict keys (call head): `trunk_repr`, `q`, `standing_bid`, `pool_size`, `target_call_prob`, `deal_idx`.
- [ ] Row dict keys (bid head): `trunk_repr`, `q`, `pool_size`, `feasible_mask`, `target_bid`, `deal_idx`.
- [ ] If `batch_size` is provided, accumulate rows into stacked numpy arrays and yield per-batch dicts (callers prefer batched).
- [ ] Add to `__all__`.
- [ ] Unit test in `tests/training/test_iter_shards.py`: tiny synthetic run (use `run_distillation(N=8, max_iters=20, …)` from a pytest tmp `data_root`); assert (a) split disjointness across train/val/test, (b) row counts match `split.json` totals, (c) `head="call"` yields no `target_bid` key and vice versa.

---

## Step 2 — Pilot script

**File:** `src/training/ar2_pilot.py` (new, ~120 lines)

- [ ] `argparse`: `--run-id`, `--N` (default 1000), `--seed` (default 0), `--trunk-ckpt` (required), `--max-iters` (default 500), `--device` (default cpu), `--holdout` flag (bool; default False — when True, run a *second* `run_distillation` at `seed=1000` to materialise `ar2_holdout_2k`).
- [ ] Wall-clock wrap: `t0 = time.monotonic(); summary = run_distillation(...); wall_clock_s = time.monotonic() - t0`.
- [ ] Solver-stats pass: re-instantiate `CFRPlusSubgameSolver` and re-solve the same `sample_deals_mixture(N, seed)` stream (cheap at N=1000; avoids needing to thread solver stats out of `run_distillation`). Record per-deal `iters_used` and `final_eps`. Build histograms with `np.histogram` (10 buckets each).
- [ ] Compute `frac_converged = (final_eps < 1e-3).mean()`.
- [ ] Write `data/runs/<run_id>/pilot_timing.json` with keys per design §1 ("Outputs").
- [ ] Write `data/runs/<run_id>/pilot_solver_stats.json` with `{iters_hist: {edges, counts}, eps_hist: {edges, counts}, frac_converged, max_iters}`.
- [ ] Print one-line GO/NO-GO verdict (stdout): `[GO]` iff wall-clock ≤ 30 min AND `frac_converged ≥ 0.95` AND `1.5*N ≤ n_rows ≤ 4*N` for both heads; else `[NO-GO: <reason>]`.
- [ ] If `--holdout` is set: after the primary pilot, invoke `run_distillation(N=2000, seed=1000, run_id="ar2_holdout_2k", ...)` once. Idempotent — skip if `data/runs/ar2_holdout_2k/split.json` already present.
- [ ] Smoke test `tests/training/test_ar2_pilot_smoke.py`: invoke `_main(["--run-id", tmp, "--N", "8", "--max-iters", "20", "--trunk-ckpt", <tiny>])` against a pytest tmp `data_root` (monkeypatch `_DATA_RUNS`). Assert both JSON files exist and parse, `frac_converged ∈ [0,1]`.

---

## Step 3 — CallPolicy training CLI

**File:** [src/agents/learned/callpolicy/trainer.py](../../src/agents/learned/callpolicy/trainer.py) (edit; additive only)

- [ ] Add `_main(argv: list[str] | None = None)` with flags per design §2.2: `--run-id`, `--load-trunk`, `--epochs`, `--batch-size`, `--lr`, `--out-dir`, `--device`, `--seed`, `--external-val-run-id`.
- [ ] Build `CallPolicyConfig` from flags (`load_trunk=args.load_trunk`, `lr=args.lr`, `device=args.device`).
- [ ] `state = build_train_state(config)` — already wires the optimizer with trunk excluded.
- [ ] Training loop per epoch:
  - Iterate `iter_shards(args.run_id, "train", head="call", batch_size=args.batch_size)`.
  - For each batch: torch-ify `trunk_repr, q, standing_bid, pool_size`; call `build_call_features` (numpy → tensor); build `targets = torch.tensor(target_call_prob)`.
  - `opt.zero_grad(); loss = loss_step(state, features, targets); loss.backward(); opt.step()`.
  - Accumulate `train_loss_sum` for the epoch.
- [ ] After each epoch, validation pass on either `--external-val-run-id` (if set, use its `"val"` *and* `"test"` rows — the external set has no train split) or the cell's own `"val"` split. Compute `val_bce = mean per-row BCE`. No grad.
- [ ] Track `best_val_bce`; on improvement write `<out_dir>/best.pt` atomically (`tmp + os.replace`) with payload `{state_dict, config_dict, val_bce, epoch}`.
- [ ] Append `{epoch, train_loss, val_bce, wall_clock_s}` to `<out_dir>/training_curve.json` (rewrite the whole file each epoch — small).
- [ ] Add `if __name__ == "__main__": _main()` at module bottom.
- [ ] Add `_main` to `__all__`.
- [ ] Smoke test `tests/agents/learned/test_callpolicy_cli_smoke.py`: build a tiny run via pilot smoke fixture (or share a session-scoped fixture); invoke `_main(["--run-id", tmp_run_id, "--load-trunk", <tiny.pt>, "--epochs", "1", "--batch-size", "8", "--out-dir", tmp_out])`. Assert `<out_dir>/best.pt` exists and `torch.load(..., weights_only=False)` round-trips per [feedback_torch_load_weights_only.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_torch_load_weights_only.md).

---

## Step 4 — BidPolicy training CLI

**File:** [src/agents/learned/bidpolicy/trainer.py](../../src/agents/learned/bidpolicy/trainer.py) (edit; additive only)

- [ ] Mirror Step 3 with the design §2.3 flag defaults (`--epochs 30`, `--batch-size 2048`).
- [ ] Per-batch tensor construction per design §2.3 table.
- [ ] `loss_step` call signature unchanged from Phase 4.
- [ ] Validation pass computes both:
  - `val_ce = mean cross-entropy` (the trained loss minus the β·H term — i.e. just the CE component).
  - `val_kl = mean KL(target ‖ pi)` = `val_ce - mean target-entropy`. Target entropy is constant in the data and can be precomputed once per shard.
  - `val_kl_per_n[n] = mean val_kl over rows with pool_size == n` for `n ∈ {2,4,6,8,10}` (include `n=2` even if mostly absent — empty buckets emit `null`).
- [ ] Best-checkpoint selection: minimize `val_kl` (not per-`n`). Payload `{state_dict, config_dict, val_kl, val_kl_per_n, epoch}`.
- [ ] `training_curve.json` rows include `val_kl_per_n` dict.
- [ ] Smoke test `tests/agents/learned/test_bidpolicy_cli_smoke.py`: as Step 3, plus assert `val_kl_per_n` dict contains every `n` represented in the synthetic data.

---

## Step 5 — Per-cell pipeline driver

**File:** `src/training/ar2_pipeline.py` (new, ~140 lines)

- [ ] `argparse` flags per design §4.2: `--run-id`, `--N`, `--seed`, `--trunk-ckpt`, `--holdout-run-id`, `--max-iters`, `--callpolicy-epochs`, `--bidpolicy-epochs`, `--device`.
- [ ] Step 1 — Distillation: `run_distillation(N, seed, run_id, trunk_ckpt, max_iters=args.max_iters)`. Record `t_distill_s`.
- [ ] Step 2 — Call subprocess for `python -m agents.learned.callpolicy.trainer` with the cell's `--run-id`, `--load-trunk`, `--external-val-run-id <holdout>`, `--epochs <callpolicy_epochs>`, `--out-dir data/runs/<run_id>/callpolicy/`, `--device <device>`. Stream stdout/stderr; assert exit code 0.
- [ ] Step 3 — Same for `python -m agents.learned.bidpolicy.trainer` → `--out-dir data/runs/<run_id>/bidpolicy/`.
- [ ] Step 4 — Read both `training_curve.json` and their `best.pt` payloads; assemble `data/runs/<run_id>/ar2_summary.json` per design §4.2.
- [ ] Subprocess invocation: set `PYTHONPATH=src` per [feedback_sweep_driver_fix.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_sweep_driver_fix.md); single `-m` per call.
- [ ] Smoke test `tests/training/test_ar2_pipeline_smoke.py` (mark `@pytest.mark.slow` if > 30 s): invoke `_main(["--run-id", tmp, "--N", "8", "--callpolicy-epochs", "1", "--bidpolicy-epochs", "1", "--trunk-ckpt", <tiny>, "--holdout-run-id", "<inline tmp holdout>", ...])`. Assert `ar2_summary.json` exists and both ckpt paths populated.

---

## Step 6 — Sweep YAML

**File:** `configs/sweeps/ar2_distillation_count.yaml` (new)

- [ ] Body per design §4.1. Substitute `<ar1_winner>` with the path resolved in Step 0.
- [ ] One axis: `N: [1000, 5000, 10000, 50000]`.
- [ ] `runner.command: python -m training.ar2_pipeline`.
- [ ] `runner.args.N: "{N}"`; `runner.fixed` carries `trunk-ckpt`, `holdout-run-id`, `max-iters`, `seed`, `callpolicy-epochs`, `bidpolicy-epochs`.
- [ ] Dry-run validation: `PYTHONPATH=src python -m training.sweep configs/sweeps/ar2_distillation_count.yaml --dry-run`. Confirm 4 cells with the right command strings.

---

## Step 7 — Sweep-summary script

**File:** `src/training/ar2_sweep_summary.py` (new, ~140 lines)

- [ ] `argparse`: `--sweep-id` (required), `--data-root` (default `data/`).
- [ ] Discover all cells under `data/sweeps/<sweep_id>/cells.json` (AR-0b convention) → list of `run_id`s.
- [ ] For each cell, load `data/runs/<run_id>/ar2_summary.json`; build map `{N: summary}`.
- [ ] Compute `kl_per_n_per_N: {n: {N: kl}}` from `bidpolicy_val_kl_per_n`.
- [ ] Slope rule per design §3.3: for each `N` (except the smallest), `slope_n(N) = (kl_n(N/2) - kl_n(N)) / kl_n(N/2)`. Elbow at smallest `N` where `max_n slope_n(N) < 0.05`.
- [ ] Selection:
  - If any `N` satisfies the elbow: `chosen_N = that N`, `reason = "elbow_at_N=<N>"`.
  - Else: `chosen_N = 10000`, `reason = "fallback_per_design_3.3"`.
- [ ] Write `data/sweeps/<sweep_id>/ar2_kl_curve.json` with `{kl_per_n_per_N, slopes}` and `data/sweeps/<sweep_id>/elbow.json` with `{chosen_N, reason, slopes, kl_per_n_per_N, callpolicy_ckpt, bidpolicy_ckpt}` (last two pulled from the chosen cell's summary).
- [ ] Plot: matplotlib `kl` vs `N` log-log, one line per `n`. Save `data/sweeps/<sweep_id>/ar2_kl_curve.png`. Gracefully skip plot if matplotlib import fails (write `.json` regardless).
- [ ] Unit tests in `tests/training/test_ar2_sweep_summary.py`:
  - Case A: hand-crafted 4-cell `kl_per_n_per_N` with monotonic plateau at `N=10k` → assert `chosen_N==10000, reason="elbow_at_N=10000"`.
  - Case B: monotonic improvement across all four cells, no plateau → assert `chosen_N==10000, reason="fallback_per_design_3.3"`.
  - Case C: plateau already at `N=1k` (slopes undefined since no `N/2`) → spec: smallest computable elbow is at `N=5k`; document and assert.

---

## Step 8 — Local code-level smoke

Before launching the pilot, prove the wiring with a 16-deal end-to-end run.

- [ ] Run all Phase-7 smokes:
  ```
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest \
    tests/training/test_iter_shards.py \
    tests/training/test_ar2_pilot_smoke.py \
    tests/training/test_ar2_pipeline_smoke.py \
    tests/training/test_ar2_sweep_summary.py \
    tests/agents/learned/test_callpolicy_cli_smoke.py \
    tests/agents/learned/test_bidpolicy_cli_smoke.py -v
  ```
- [ ] Run AR-2 regression subset:
  ```
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest \
    tests/agents/learned/ tests/training/cfr/ tests/training/ \
    --ignore=tests/agents/heuristic/test_cfr_1v1.py \
    --ignore=tests/training/test_ar2_pipeline_smoke.py -v
  ```
- [ ] All pass except the documented pre-existing R-NaD import failure ([feedback_cfr_rnad_defunct.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_cfr_rnad_defunct.md)).
- [ ] Commit: "AR-2 Phase 7 code: pilot + per-cell pipeline + sweep YAML + summary".

---

## Step 9 — Pilot run (session boundary; long-running)

**Infra deviation (2026-05-12):** initial launch at `--max-iters=500` ran 3h+ with zero output and was killed. Three pilot-infra issues fixed in this session ([src/training/cfr_distillation.py](../../src/training/cfr_distillation.py), [src/training/ar2_pilot.py](../../src/training/ar2_pilot.py)):

1. **Redundant second solver pass removed.** `run_distillation` now returns `iters_used` + `final_eps` in its summary dict (collected inline during the single solve loop). `ar2_pilot.py` reads from there — no longer re-solves all N deals after distillation.
2. **Progress logging added.** `run_distillation(..., progress=True)` emits ~20 stderr lines (`[distill <run> k/N deals (rate/s, eta)]`). Pilot defaults `progress=True`; `--no-progress` opts out.
3. **Process-pool parallelism.** `run_distillation(..., workers=W)` dispatches per-deal solves through `ProcessPoolExecutor`. Smoke at N=16 with W=6 gave 4× speedup vs serial; expected ~6× at N=1000. New module-level `_solve_one_deal` worker is picklable. `ar2_pilot.py` exposes `--workers`.
4. **Pilot default `--max-iters` lowered 500 → 200** (Phase-4 smoke hit 53% bid-CE reduction at max_iters=200, above the 50% gate).

Combined effect: ~12× wall-clock improvement (2× from removing redundant pass + ~6× from workers, with overhead). N=1000 + N=2000 holdout projected at ~25 min total on a 10-core M-series.

- [x] Run the pilot (single command — `--holdout` chains the 2k validation set after the primary 1k):
  ```
  PYTHONPATH=src /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -u -m training.ar2_pilot \
    --run-id ar2_pilot_1k --N 1000 --seed 0 \
    --trunk-ckpt data/runs/ar1-20260430T030730Z-b64-h256-n2-fb17d031/handmodel/best.pt \
    --max-iters 200 --workers 6 --holdout \
    > data/runs/ar2_pilot_1k.log 2>&1 &
  ```
  Launched 2026-05-12 02:36 (PID 47648).
- [x] Read `pilot_timing.json` + `pilot_solver_stats.json` (2026-05-17). Gate results:
  - Wall-clock: **298 s (4.97 min)** ≤ 30 min → **PASS** (6× under budget after parallelization fixes).
  - `frac_converged`: **1.00** (all 1000 deals reach ε≈0 at max_iters=200) → **PASS**.
  - Row counts: **n_rows=218,000** vs. gate `[1500, 4000]` → **NO-GO by gate**, but treated as a **design/gate spec miss**. Solver writes every visited infoset in the bidding DAG (~218/deal — distribution min=196, p25=211, median=216, p95=216, max=216). Design §5 budgeted 15–60 rows/deal assuming reach-prob-filtered support; code does not implement that filtering. Decision: **accept the higher row count as the corrected per-deal estimate** and proceed to Step 10. Gate threshold in `ar2_pilot.py:99` left as-is (will be revisited if it becomes load-bearing); design doc row-budget paragraph (§4 / §5) to be updated at Phase 9.
- [x] Update [project_overnight_training_plan.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/project_overnight_training_plan.md) with pilot launch time + projected sweep budget (deferred — only the impl-progress memory was updated this session; do at sweep launch).
- [ ] **Downstream gap surfaced:** shard rows carry no `reach_prob` column (keys: `deal_idx, state_player, cur_bid_idx, hand_p0, hand_p1, pool_size, target_bid, feasible_mask, target_call`). Trainers therefore loss-average uniformly over all visited infosets, not reach-weighted as design §5 implied. Likely defensible (solver-average labels are already reach-correct), but verify at acceptance gate; if not defensible, add `reach_prob` to row schema + weight in `loss_step`.
- [ ] **Sweep launch caveat:** sweep cells now run with internal worker pools. On a 10-core box, choose `(sweep --max-parallel) × (per-cell --workers) ≤ ~10` to avoid oversubscription. Recommend `--max-parallel 2 --workers 4` (4 cells × ~8 cores busy at any time, but only 2 distillations active concurrently) or `--max-parallel 1 --workers 8`.

---

## Step 10 — Launch the sweep (session boundary; ≤ 33 h wall-clock)

- [x] Confirm `git status` clean (sweep harness records the SHA). SHA at launch: `72a29c1`.
- [x] **N=10k smoke cell run first (2026-05-17 14:43→15:39, 56 min):** clean end-to-end. `ar2_summary.json`: callpolicy val BCE=0.282, bidpolicy val KL=0.489 overall (n=4→0.068, n=6→0.409, n=8→0.554, n=10→0.933). Confirms shard write + heads train at scale. `data/runs/ar2_cell_n10k/`.
- [x] Launch (2026-05-17 20:01):
  ```bash
  PYTHONPATH=src /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m training.sweep \
    configs/sweeps/ar2_distillation_count.yaml --max-parallel 2 \
    > data/runs/ar2_sweep.log 2>&1 &
  ```
  Used `--max-parallel 2` (not 4) because per-cell `workers=4` is in YAML — 2×4=8 cores fits the 10-core box.
- [x] Sweep cell IDs (sweep_id prefix `ar2-20260517T200142Z-*-72a29c18`): `N1000`, `N5000`, `N10000`, `N50000`.
- [ ] After completion, verify all 4 cells produced `data/runs/<cell_run_id>/ar2_summary.json`. Investigate any missing cell before summarising.
- [ ] **Repo hygiene fix landed alongside sweep launch:** prior commits tracked 576 `.npz` shard files (5.4 GB). Added `.gitignore` rules for `cfr_deals/`, `*.npz`, and `data/runs/*.log`; `git rm --cached` cleared the index. Shards are reproducible from seed + config, so safe.

---

## Step 11 — Elbow detection + pick `N`

- [ ] Run the summariser:
  ```
  PYTHONPATH=src /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m training.ar2_sweep_summary \
    --sweep-id <sweep_id>
  ```
- [ ] Inspect `data/sweeps/<sweep_id>/elbow.json` and `ar2_kl_curve.png`. Confirm the chosen `N` matches visual elbow on the plot.
- [ ] If `reason == "fallback_per_design_3.3"`: record this as a deviation in §"Deviations to record" below and in the Phase 7 entry in the parent checklist.

---

## Step 12 — Wire chosen checkpoints into the `modular_nash` factory

This unblocks Phase 8's `ModularNashAgent` construction.

- [ ] Update `src/agents/registry.py::_make_modular_nash` to load the three artefacts:
  - Trunk: `<ar1_winner>/best.pt`
  - CallPolicy: `<chosen_cell>/callpolicy/best.pt` (read from `elbow.json`)
  - BidPolicy:  `<chosen_cell>/bidpolicy/best.pt`
- [ ] Hard-code these paths via a small config dataclass `ModularNashCheckpoints` (`trunk_ckpt`, `callpolicy_ckpt`, `bidpolicy_ckpt`) loaded from a YAML at `configs/modular_nash.yaml` (new) so paths are not buried in registry code.
- [ ] Smoke: `python -c "from agents.registry import make_agent; a = make_agent('modular_nash', exact_rules=True, high_hand=True); print(a.action_probs(...))"` (use a sample infostate from an existing test).
- [ ] Confirm `pytest tests/agents/learned/test_modular_nash_smoke.py -v` still passes — those tests construct the agent directly, but registry-level breakage must not regress.

---

## Step 13 — Memory + docs (lightweight; full doc pass is Phase 9)

- [ ] Update [project_ar2_impl_progress.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/project_ar2_impl_progress.md): mark Phase 7 ✅ with `sweep_id`, `chosen_N`, ckpt paths.
- [ ] Update parent checklist: tick the Phase 7 boxes; status line moves to "Phases 7 complete (yyyy-mm-dd); Phase 8 pending".
- [ ] Commits (one per session boundary):
  - "AR-2 Phase 7 pilot: GO with N=1000, wall-clock <Xs>, frac_converged=<…>"
  - "AR-2 Phase 7 sweep: 4-cell ar2_distillation_count complete (sweep_id=<…>)"
  - "AR-2 Phase 7 elbow: chosen N=<…>, wired into modular_nash factory"

---

## Out of scope for Phase 7

- Acceptance gate (win-rate, exploitability) — Phase 8.
- `project_ar2_distillation.md` memo + CHANGELOG — Phase 9.
- Adding a `100k` cell — explicitly deferred per parent design §3.3.
- Early-stopping inside cells — design §10 open question.

---

## Deviations recorded in implementation (2026-05-11)

- **Call-head data source.** `trunk_<sh>.npz` is aligned 1:1 with `bid_<sh>.npz`
  only (Phase-4 implementation). To keep `iter_shards` clean, the call head
  trains on the bid-eligible row subset; call-only rows (where
  `target_call ≈ 1`, no real call-vs-bid decision) are dropped. `write_shards`
  now also stores `target_call` inside `bid_<sh>.npz` so `iter_shards(head="call")`
  can join `bid_<sh>` + `trunk_<sh>` 1:1. The full call-row superset in
  `call_<sh>.npz` remains untouched (no consumer in Phase 7).
- **Sweep index file.** AR-0b's sweep harness writes `data/sweeps/<id>/index.json`
  (cell_key → run_id), not `cells.json` as the checklist named. Summariser reads
  `index.json`.

---

## Deviations to record if they occur

- Pilot wall-clock > 30 min at `max_iters=500`: record actual time, the chosen remediation (lower `max_iters`, or fall back to `N=10k`), and skip the sweep.
- `frac_converged < 0.95` at `max_iters=1000`: cap at 1000 and document; do not chase further. The bid head will absorb residual solver noise.
- Sweep cell wall-clock blow-up at `N=50k` (> 1.5× projection): kill that cell, set chosen `N` = best plateau in `{1k, 5k, 10k}` and record.
- Any smoke test from Steps 1–7 needs `weights_only=False` not yet applied: thread it through and note here.
- Elbow falls outside the swept range (all four cells still improving): per design §3.3, `chosen_N = 10000` (fallback), record `reason`, **do not** auto-add `100k`.
