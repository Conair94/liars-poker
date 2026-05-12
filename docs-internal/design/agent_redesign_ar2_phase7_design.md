# AR-2 Phase 7 — Pilot run + distillation-count sweep

- **Status:** Draft (design-first gate; implementation in follow-up session)
- **Date:** 2026-05-11
- **Owner:** main
- **Parent design:** [agent_redesign_ar2.md](agent_redesign_ar2.md) §3, §6
- **Parent checklist:** [agent_redesign_ar2_checklist.md](agent_redesign_ar2_checklist.md) §Phase 7
- **Predecessors:** Phases 1–6 complete (2026-05-11). `run_distillation` end-to-end pipeline lives at [src/training/cfr_distillation.py](../../src/training/cfr_distillation.py); CallPolicy/BidPolicy `loss_step` + `build_train_state` ready; no head-training CLI exists yet.

Phase 7 turns the offline distillation pipeline into actual trained head
checkpoints across a 4-cell `N` sweep, picks the elbow per parent §3.2/§3.3,
and emits the `(LearnedHandModel[frozen], DistilledCallPolicy, DistilledBidPolicy)`
artefacts that Phase 8 will gate on.

This phase introduces **new files only** — no existing module is touched.

---

## 0. Context — what already exists

| Component | Location | Status |
|-----------|----------|--------|
| Deal sampling + solve + walk + shard | [src/training/cfr_distillation.py](../../src/training/cfr_distillation.py) `run_distillation` | Done (Phase 4) |
| Trunk-activation precompute | same module, `precompute_trunk` | Done (Phase 4) |
| Split JSON (80/10/10 by-deal) | same module, `write_split_json` | Done (Phase 4) |
| Head losses + train-state factories | `agents/learned/{callpolicy,bidpolicy}/trainer.py` | Done (Phase 4) |
| HandModel checkpoint pin (`b64-h256-n2`) | `data/runs/ar1_handmodel_arch/<best>/best.pt` | Done (AR-1) |
| Sweep harness | [src/training/sweep.py](../../src/training/sweep.py) | Done (AR-0b) |
| Sweep driver fix | [feedback_sweep_driver_fix.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_sweep_driver_fix.md) | Applies verbatim |

What is missing and Phase 7 introduces:

1. **Head training CLIs** — neither `callpolicy/trainer.py` nor `bidpolicy/trainer.py` has a `_main()`. Phase 7 adds one to each.
2. **Validation KL evaluator** — a one-shot script that loads a trained head + held-out shards and emits `KL(target ‖ pi)` per `n`.
3. **Pilot script** — a thin wrapper around `run_distillation(N=1000, …)` that records per-deal solver iters/eps so we can gate the full sweep on wall-clock.
4. **Sweep YAML** — `configs/sweeps/ar2_distillation_count.yaml`, one axis (`N`), four cells.
5. **Per-cell driver entry point** — a single `python -m training.ar2_pipeline` command the sweep can call with `{N}` interpolated; it runs distillation → head training → validation and writes the cell summary.

---

## 1. The pilot pre-flight (`N=1000`, ≤ 30 min CPU)

### Purpose

Three gates before launching the 4-cell sweep:

- **Wall-clock gate.** End-to-end `run_distillation(N=1000, …)` must finish in ≤ 30 minutes on a single CPU. Linear-in-`N` extrapolation puts the `N=50k` cell at ≤ 25 hours, which is the budget Phase 7 commits to.
- **Solver convergence gate.** ≥ 95% of deals reach `final_eps < 1e-3` at `max_iters=500`. If not, lift `max_iters` to 1000 before the sweep (one knob; no other re-tuning).
- **Row-count sanity.** Expected `~2.5N` distillation rows per head at `N=1000` (~2500 bid rows, ~2500 call rows). Wide deviation → bug.

### Inputs

| Symbol | Type | Source |
|--------|------|--------|
| `N` | `int` | 1000 |
| `seed` | `int` | 0 |
| `trunk_ckpt` | `str` | `data/runs/<ar1_winner>/best.pt` (b64-h256-n2 pinned) |
| `max_iters` | `int` | 500 |
| `run_id` | `str` | `ar2_pilot_1k` |

### Procedure

```
python -m training.ar2_pilot \
    --run-id ar2_pilot_1k \
    --N 1000 \
    --seed 0 \
    --trunk-ckpt data/runs/<ar1_winner>/best.pt \
    --max-iters 500
```

Internally:

1. Time-wrap `run_distillation(...)` — capture wall-clock to `pilot_timing.json`.
2. After distillation, iterate the sharded data once to compute the
   per-deal `iters_used` and `final_eps` histograms; write
   `pilot_solver_stats.json`.
3. Print a one-line **GO / NO-GO** verdict to stdout based on the three gates.

### Outputs

- `data/runs/ar2_pilot_1k/cfr_deals/{call,bid,trunk}_<shard>.npz` (shared shard layout from Phase 4).
- `data/runs/ar2_pilot_1k/split.json`.
- `data/runs/ar2_pilot_1k/pilot_timing.json` — `{wall_clock_seconds, n_deals, rows_per_deal, projected_50k_hours}`.
- `data/runs/ar2_pilot_1k/pilot_solver_stats.json` — `{iters_hist, eps_hist, frac_converged}`.

### Pass criteria

| Gate | Threshold | If failed |
|------|-----------|-----------|
| Wall-clock | ≤ 30 min | Reduce `max_iters`, re-run pilot. If still failing, fall back to `N=10k` as the final pick (parent §3.3) and skip the sweep. |
| Solver convergence | ≥ 95% deals at `eps < 1e-3` | Raise `max_iters` to 1000, re-run pilot. |
| Row counts | `1.5N ≤ n_rows ≤ 4N` per head | Investigate `walk_avg_strategy`; do not proceed. |

The pilot is a **NO-GO veto**, not a tuning loop. If both early gates pass, the sweep launches with whatever `max_iters` the pilot validated.

### File location

`src/training/ar2_pilot.py` (new; ~80 lines wrapping `run_distillation`).

---

## 2. Head training CLIs

Both heads share the same Phase 4 loss + train-state machinery; only the data
loader and per-step plumbing differ. Phase 7 adds parallel CLI modules.

### 2.1 Shared dataset loader

| Symbol | Type | Source |
|--------|------|--------|
| `run_id` | `str` | matches the distillation `run_id` |
| `split` | `str` | `"train"` / `"val"` / `"test"` (resolved via `split.json`) |

```python
# new helper inside src/training/cfr_distillation.py (kept in one place)
def iter_shards(
    run_id: str,
    split: str,
    *,
    head: Literal["call", "bid"],
    data_root: str | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Yields a dict per row: trunk_repr, q, standing_bid, pool_size, target.
    Skips rows whose deal_idx is not in the requested split bucket."""
```

The Phase 4 sharded `.npz` files already align 1:1 by row index across
`{call_<shard>.npz, bid_<shard>.npz, trunk_<shard>.npz}`. The loader joins
them lazily; one `.npz` is in memory at a time (~50 MB at `N=50k`).

### 2.2 CallPolicy CLI — `python -m agents.learned.callpolicy.trainer`

**New `_main()` added at the bottom of [callpolicy/trainer.py](../../src/agents/learned/callpolicy/trainer.py).**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--run-id` | `str` | required | Distillation `run_id` (the cell's data) |
| `--load-trunk` | `str` | required | AR-1 HandModel checkpoint |
| `--epochs` | `int` | 20 | One epoch = one pass over training shards |
| `--batch-size` | `int` | 4096 | |
| `--lr` | `float` | 1e-3 | Overrides config default |
| `--out-dir` | `str` | `data/runs/<run_id>/callpolicy/` | |
| `--device` | `str` | `cpu` | |
| `--seed` | `int` | 0 | |

Procedure per epoch:

1. Iterate `iter_shards(run_id, "train", head="call")` in shuffled batches.
2. For each batch, build `features = build_call_features(trunk_repr, q, standing_bid, pool_size)` (already in `callpolicy/network.py`) and `targets = avg_call_prob`. Call `loss_step(state, features, targets)`. Step optimizer.
3. After each epoch, evaluate on the `"val"` split:
   - `val_bce` (mean of per-row BCE)
   - Save best-`val_bce` checkpoint to `<out-dir>/best.pt` (atomic rename).

Final artefacts:

- `<out-dir>/best.pt` — `{state_dict, config_dict, val_bce, epoch}`.
- `<out-dir>/training_curve.json` — `[{epoch, train_loss, val_bce, wall_clock_s}, …]`.

### 2.3 BidPolicy CLI — `python -m agents.learned.bidpolicy.trainer`

Same skeleton as 2.2 with these differences:

| Flag | Default | Notes |
|------|---------|-------|
| `--epochs` | 30 | Bid head is wider (110-way softmax); needs more passes. |
| `--batch-size` | 2048 | Lower than CallPolicy to hold 110-d targets in cache. |
| `--out-dir` | `data/runs/<run_id>/bidpolicy/` | |

Per-batch tensors fed to `loss_step`:

| Tensor | Shape | Source |
|--------|-------|--------|
| `features` | `(B, 367)` | `concat([trunk_repr, q, n/25])` |
| `log_q` | `(B, NUM_BIDS)` | `log(q + 1e-12)` |
| `bid_mask` | `(B, NUM_BIDS)` | feasibility from the row |
| `targets` | `(B, NUM_BIDS)` | `avg_bid_dist` (rows sum to 1; zero on infeasible) |
| `pool_size` | `(B,)` | `n` per row |

Validation metric: `val_kl = mean over rows of KL(target ‖ pi)` (the same loss used for training, no entropy term — i.e. cross-entropy minus target-entropy constant; we report cross-entropy and KL side-by-side for clarity).

Final artefacts:

- `<out-dir>/best.pt` — `{state_dict, config_dict, val_kl, epoch}`.
- `<out-dir>/training_curve.json` — `[{epoch, train_loss, val_ce, val_kl, val_kl_per_n: {2:…, 4:…, 6:…, 8:…, 10:…}}, …]`.

The **per-`n` validation KL** is the metric Phase 7 plots vs `N` (§3 below). Compute it at the end of every epoch; saves a separate pass at sweep-summary time.

### 2.4 Why two CLIs and not one combined script

- Different epoch counts and batch sizes.
- Different validation metrics (BCE vs KL).
- Independent best-checkpoint selection — call-head accuracy at small `N` may peak before bid-head accuracy.
- Easier to retrain one head in isolation later (e.g. AR-3 fine-tune).

The Phase 7 sweep cell driver (§4) calls both back-to-back.

---

## 3. Validation: per-`n` KL curve and the elbow

### 3.1 What we plot

Parent §3.2 specifies: `KL(distilled ‖ cfr_plus_avg)` per `n`, against `N`.

For each sweep cell (`N ∈ {1k, 5k, 10k, 50k}`) and for each `n ∈ {4, 6, 8, 10}`, we report the BidPolicy `val_kl` on the **shared held-out set** (§3.2 below). The CallPolicy `val_bce` is reported the same way but does not gate elbow selection.

### 3.2 The shared held-out set

Per parent §3.2: **fixed 2 000 deals not in any train split, shared across cells.**

- One-shot generation in the pilot run (§1): the pilot writes
  `data/runs/ar2_holdout_2k/` using a *disjoint* seed (`seed=1000`,
  guarantees the deal stream doesn't overlap `seed=0` at any cell's `N`).
- All four sweep cells evaluate on this same 2k-deal set; the
  validation split inside each cell's own data is only used for
  early-stopping/best-checkpoint selection.
- The bidpolicy/callpolicy trainers accept `--external-val-run-id ar2_holdout_2k`
  to point validation at the shared set instead of (or in addition to)
  the cell's own val split.

### 3.3 Elbow rule (parent §3.2/§3.3)

Define `kl_n(N) = mean validation KL at pool-size n` for the chosen `N`.

Per-`n` slope per doubling: `slope_n(N) = (kl_n(N/2) - kl_n(N)) / kl_n(N/2)`.

**Elbow condition:** for **every** `n ∈ {4, 6, 8, 10}`, `slope_n(N) < 0.05` (less than 5% improvement per doubling).

**Selection rule:**

1. Find the smallest `N` in `{1k, 5k, 10k, 50k}` where the elbow condition holds. Pick that `N`.
2. If no `N` satisfies it: per parent §3.3, fall back to `N = 10k` and record a deviation. Do *not* automatically add a `100k` cell — that decision belongs to a follow-up session, not the sweep harness.

### 3.4 Plotting / summary artefact

`src/training/ar2_sweep_summary.py` (new, ~120 lines):

- Reads each cell's `bidpolicy/training_curve.json` (best epoch's `val_kl_per_n`).
- Emits `data/sweeps/<sweep_id>/ar2_kl_curve.json` and a matplotlib PNG.
- Computes the elbow per §3.3 and writes `data/sweeps/<sweep_id>/elbow.json`:
  ```
  {
    "chosen_N": 10000,
    "reason": "elbow_at_N=10k" | "fallback_per_design_3.3",
    "slopes": {"4": …, "6": …, "8": …, "10": …},
    "kl_per_n_per_N": {…}
  }
  ```

This is a post-hoc script invoked once after the sweep completes; it is **not** part of any sweep cell.

---

## 4. Sweep YAML and per-cell driver

### 4.1 `configs/sweeps/ar2_distillation_count.yaml`

```yaml
sweep_name: ar2_distillation_count
phase: ar2
production: false
random_seed: 0
device: cpu

# One-axis sweep: deal count N. Each cell runs distillation,
# trains both heads on the cell's data, and reports per-n
# validation KL on the shared 2k held-out set.
axes:
  - key: N
    values: [1000, 5000, 10000, 50000]

runner:
  command: python -m training.ar2_pipeline
  args:
    N: "{N}"
  fixed:
    trunk-ckpt:        data/runs/<ar1_winner>/best.pt
    holdout-run-id:    ar2_holdout_2k
    max-iters:         500          # validated by pilot
    seed:              0
    callpolicy-epochs: 20
    bidpolicy-epochs:  30
```

`<ar1_winner>` is substituted by the implementer at the point of writing — the path is fully resolved at sweep-launch time, not interpolated. Memo: see [project_ar1_handmodel.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/project_ar1_handmodel.md).

### 4.2 Per-cell driver — `src/training/ar2_pipeline.py`

Single entry point that the sweep invokes once per cell. New file (~120 lines).

CLI:

```
python -m training.ar2_pipeline \
    --run-id ar2_distillation_count__N=10000  \
    --N 10000 \
    --seed 0 \
    --trunk-ckpt data/runs/<ar1_winner>/best.pt \
    --holdout-run-id ar2_holdout_2k \
    --max-iters 500 \
    --callpolicy-epochs 20 \
    --bidpolicy-epochs 30 \
    --device cpu
```

`--run-id` is generated by the sweep harness from `{sweep_name}__N={N}` per AR-0b convention.

Procedure:

1. `run_distillation(N, seed, run_id, trunk_ckpt, max_iters)` → sharded data.
2. Subprocess `python -m agents.learned.callpolicy.trainer …` with cell's `run_id` and shared `--external-val-run-id`. Wait, capture exit code.
3. Subprocess `python -m agents.learned.bidpolicy.trainer …` likewise.
4. Read the two `training_curve.json`s, write a single per-cell
   `data/runs/<run_id>/ar2_summary.json`:
   ```
   {
     "N": 10000, "seed": 0,
     "distillation_wall_clock_s": …,
     "callpolicy_val_bce_best": …,
     "bidpolicy_val_kl_best": …,
     "bidpolicy_val_kl_per_n":  {"4": …, "6": …, "8": …, "10": …},
     "callpolicy_ckpt": "data/runs/<run_id>/callpolicy/best.pt",
     "bidpolicy_ckpt":  "data/runs/<run_id>/bidpolicy/best.pt"
   }
   ```

The sweep harness aggregates the four `ar2_summary.json`s; `ar2_sweep_summary.py` consumes them.

### 4.3 Why subprocess and not in-process

- AR-0b sweep harness expects one `command:` per cell. Wrapping the two head trainers inside `ar2_pipeline.py` keeps that contract.
- Subprocess isolation avoids torch CUDA/MPS context leakage between heads (irrelevant on CPU today, but cheap insurance).
- Each head's stdout/stderr lands in `data/sweeps/<sweep_id>/<cell>.log` naturally.

---

## 5. Run plan + budget

Wall-clock (CPU-single, projected from pilot):

| Cell | Distill | CallPolicy | BidPolicy | Total (h) |
|------|---------|------------|-----------|-----------|
| `N=1k`  | < 0.5 | 0.05 | 0.1 | < 1 |
| `N=5k`  | ~2.5  | 0.25 | 0.5 | ~3 |
| `N=10k` | ~5    | 0.5  | 1   | ~7 |
| `N=50k` | ~25   | 2.5  | 5   | ~33 |

Total ≤ 44 h serial. Sweep harness `--max-parallel 4` cuts it to the `N=50k` cell's ~33 h (CPU-bound). Realistic launch: kick off the sweep before EOD, check on it the next afternoon.

Memo: long runs follow [project_overnight_training_plan.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/project_overnight_training_plan.md) — update with the sweep launch time when Phase 7 starts.

---

## 6. Outputs Phase 8 consumes

| Artefact | Location |
|----------|----------|
| Winning `N` and rationale | `data/sweeps/<sweep_id>/elbow.json` |
| `DistilledCallPolicy` checkpoint | `data/runs/<chosen_cell_run_id>/callpolicy/best.pt` |
| `DistilledBidPolicy` checkpoint  | `data/runs/<chosen_cell_run_id>/bidpolicy/best.pt`  |
| Pinned trunk | unchanged from AR-1 |

Phase 8 builds `ModularNashAgent(trunk, call_head, bid_head)` from these three files and runs the win-rate + exploitability gates.

---

## 7. Tests added in Phase 7

These are **smoke** tests for the new CLIs and the pipeline glue. They do not duplicate Phase 4's loss/data tests.

| File | Purpose |
|------|---------|
| `tests/training/test_ar2_pilot_smoke.py` | `run_distillation(N=8, max_iters=20)` + verify the three output JSON files are well-formed; `frac_converged` ∈ [0, 1]. |
| `tests/agents/learned/test_callpolicy_cli_smoke.py` | Invoke `callpolicy.trainer._main(["--run-id", tmp_run_id, "--epochs", "1", "--batch-size", "8"])` against a tiny pre-staged shard; assert `best.pt` exists and loads. |
| `tests/agents/learned/test_bidpolicy_cli_smoke.py`  | Same for BidPolicy; additionally assert `val_kl_per_n` keys cover all `n` present in the synthetic data. |
| `tests/training/test_ar2_sweep_summary.py` | Synthetic 4-cell `ar2_summary.json` set with a hand-crafted KL curve → assert elbow at expected `N` per §3.3; second case with monotonic improvement → fallback path. |
| `tests/training/test_ar2_pipeline_smoke.py` | End-to-end `ar2_pipeline._main(...)` with `N=8`, 1 epoch each; assert `ar2_summary.json` exists and has both ckpt paths populated. |

All marked fast except `test_ar2_pipeline_smoke.py` (10–30 s; mark `@pytest.mark.slow` if runtime exceeds 30 s in CI).

---

## 8. File summary

| Path | New? | Purpose |
|------|------|---------|
| `src/training/ar2_pilot.py` | new | Pilot wrapper with timing + solver stats |
| `src/training/ar2_pipeline.py` | new | Per-cell driver (distill + train both heads + summary) |
| `src/training/ar2_sweep_summary.py` | new | Aggregator + elbow detector + KL-curve plot |
| `src/training/cfr_distillation.py` | edit | Add `iter_shards(run_id, split, head)` helper |
| `src/agents/learned/callpolicy/trainer.py` | edit | Add `_main()` + `--external-val-run-id` |
| `src/agents/learned/bidpolicy/trainer.py`  | edit | Add `_main()` + `--external-val-run-id` + per-`n` val KL |
| `configs/sweeps/ar2_distillation_count.yaml` | new | One-axis sweep over `N` |
| `tests/training/test_ar2_pilot_smoke.py` | new | Pilot smoke |
| `tests/training/test_ar2_pipeline_smoke.py` | new | End-to-end smoke |
| `tests/training/test_ar2_sweep_summary.py` | new | Elbow detector unit tests |
| `tests/agents/learned/test_callpolicy_cli_smoke.py` | new | CallPolicy CLI smoke |
| `tests/agents/learned/test_bidpolicy_cli_smoke.py` | new | BidPolicy CLI smoke |

Two files (`callpolicy/trainer.py`, `bidpolicy/trainer.py`) gain a `_main()`; both edits are additive (no Phase-4 behaviour change). `cfr_distillation.py` gains one helper.

---

## 9. Dependencies and environment

- Python: `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3` per [feedback_python_env_torch.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_python_env_torch.md).
- Device: CPU. MPS is 21× slower for the surrounding training loops per [feedback_mps_cpu_speed.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_mps_cpu_speed.md); the head trainers are MLP-only but we keep `--device cpu` default for consistency.
- Sweep driver invocation per [feedback_sweep_driver_fix.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_sweep_driver_fix.md): `PYTHONPATH=src python -m training.sweep configs/sweeps/ar2_distillation_count.yaml` (single `-m`).
- W&B: opt-in via `--wandb` flag on the per-cell driver; entity per [reference_wandb_entity.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/reference_wandb_entity.md). Not required for the sweep itself; JSON artefacts are sufficient.

---

## 10. Open questions (none blocking implementation)

- **Whether to also sweep `max_iters`.** Phase 7 fixes it via the pilot. If the pilot's `final_eps` histogram is heavy-tailed, AR-3 may want to revisit; not Phase 7's concern.
- **Whether to early-stop within a cell rather than fixed epochs.** Current design uses fixed epochs with best-checkpoint selection. Early-stop adds tuning surface for marginal wall-clock gain; revisit if a cell's `val_kl` plateaus before epoch 10.
- **5p / multi-round extensions** — AR-5; not Phase 7.

---

**Stopping here per the design-first gate** ([feedback_design_first_gate.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_design_first_gate.md)). Next session writes the Phase 7 implementation checklist; the session after that runs the pilot and launches the sweep.
