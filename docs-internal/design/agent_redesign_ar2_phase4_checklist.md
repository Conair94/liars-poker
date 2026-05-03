# AR-2 Phase 4 Checklist — Distillation pipeline + losses + entropy floor

- **Status:** Complete (2026-05-03) — design at [agent_redesign_ar2_phase4_design.md](agent_redesign_ar2_phase4_design.md)
- **Date:** 2026-05-03
- **Parent checklist entry:** [agent_redesign_ar2_checklist.md](agent_redesign_ar2_checklist.md) Phase 4
- **Predecessors in series:** Phase 3 commit `c6cb672`

Sequenced so each step's dependencies are already in place. Single end-of-phase commit at the end (Phase 4 is one logical unit; intermediate WIP commits are fine if a long step ends mid-session).

---

## Step 1 — Hand-size mixture sampler wrapper

- [x] Add `sample_deals_mixture(N, seed, mix={2: .25, 3: .25, 4: .25, 5: .25})` to `src/training/cfr_distillation.py` (per-seat hand_size keys, total pool n = 2 × hand_size).
- [x] Wraps existing [src/training/metrics/deal_sampler.py](../../src/training/metrics/deal_sampler.py) — calls `sample_deals` four times with deterministic per-bucket seeds (`seed + k*7919`); yields `(hands, n_total)` interleaved across buckets.
- [x] Test in `tests/training/test_cfr_distillation_smoke.py::test_mixture_sampler_counts` — N=400 yields exactly 100 per bucket; deterministic in seed.

## Step 2 — Walk avg-strategy → row generator

- [x] Implement `walk_avg_strategy(sol: SubgameSolution, hands) -> Iterator[Row]` in `cfr_distillation.py`. BFS from ROOT=(None, 0); emit one CallRow + one BidRow per visited state (per design §2.2); skip never-visited states (states absent from `sol.avg_call_prob`).
- [x] Build `Infostate` per row via existing `build_canonical_match_state` + `state_bridge` adapter.
- [x] Compute `target_bid` row: `sol.avg_bid_dist[s]` divided by `(1 - sol.avg_call_prob[s])` so it sums to 1 over feasible bids; if `1 - call_prob ≈ 0`, omit BidRow but still emit CallRow.
- [x] Test: smoke deal `n=4` yields ≥ 1 CallRow and ≥ 1 BidRow; all `target_bid` rows sum to 1 ± 1e-5.

## Step 3 — Sharded `.npz` writer

- [x] `write_shards(rows, run_id, shard_count=64)` in `cfr_distillation.py`:
  - [x] Group rows by `deal_idx % shard_count` into in-memory buffers.
  - [x] Flush each shard to `data/runs/<run_id>/cfr_deals/{call,bid}_<shard>.npz` with the schemas in design §2.3.
  - [x] Pad `hand_p0`/`hand_p1` to `max_hand=5` with `-1` sentinel.
- [x] Write `data/runs/<run_id>/cfr_deals/split.json` with deal_idx → split mapping (80/10/10 by hash).
- [x] Test: smoke run with N=32, shard_count=4 produces 1–4 shard files of each kind, all rows readable round-trip via `np.load`.

## Step 4 — Trunk activation precompute pass

- [x] `precompute_trunk(run_id, trunk_ckpt, device='cpu')` in `cfr_distillation.py`:
  - [x] Iterate shards in deal-order; reconstruct Infostate from each row; run frozen trunk forward (reuses [`_trunk_forward`](../../src/agents/learned/callpolicy/network.py#L160) helper) to produce 256-d `trunk_repr`.
  - [x] Run HandModel forward on the same Infostate to produce `q` (110-d posterior).
  - [x] Write `trunk_<shard>.npz` aligned 1:1 with `bid_<shard>.npz` row order.
- [x] Test: smoke run produces `trunk_*.npz` aligned with `bid_*.npz` (same row count per shard); `trunk_repr.shape == (R, 256)`.

## Step 5 — Top-level `run_distillation` entry point

- [x] `run_distillation(N, seed, run_id, trunk_ckpt, max_iters=500)` orchestrates:
  1. `sample_deals_mixture` → list of deals
  2. for each deal: `CFRPlusSubgameSolver(max_iters).solve(deal)` → `walk_avg_strategy` → buffer rows
  3. `write_shards`
  4. `precompute_trunk`
- [x] Add `if __name__ == "__main__":` CLI: `python -m training.cfr_distillation --N 32 --seed 0 --run-id phase4_smoke --trunk-ckpt <path>`.

## Step 6 — CallPolicy loss body

- [x] Add `CallPolicyNet._raw_logits(x)` returning `fc2(relu(ln(fc1(x))))` (no sigmoid) so the loss can use `binary_cross_entropy_with_logits`.
- [x] Replace `loss_step` `NotImplementedError` in [src/agents/learned/callpolicy/trainer.py](../../src/agents/learned/callpolicy/trainer.py) with the BCE-with-logits form per design §3.1.
- [x] Test `tests/agents/learned/test_phase4_losses.py::test_callpolicy_loss_decreases` — synthetic 256-row batch, 50 optimizer steps, loss strictly < initial loss.

## Step 7 — BidPolicy loss body

- [x] Replace `loss_step` in [src/agents/learned/bidpolicy/trainer.py](../../src/agents/learned/bidpolicy/trainer.py) with KL+entropy form per design §3.2.
- [x] Test `test_bidpolicy_loss_decreases` — synthetic batch (random feasible masks, sparse target distributions), 50 optimizer steps, loss strictly < initial loss.
- [x] Test `test_bidpolicy_no_nan_on_infeasible` — every infeasible action gets target=0 and produces 0 loss contribution; no NaN in gradients.

## Step 8 — Trunk-freeze invariance test (§7.5)

- [x] `test_phase4_losses.py::test_trunk_freeze_invariance` — build train state, snapshot trunk param L2 norms, run 10 optimizer steps on synthetic batch, assert L2 norms unchanged byte-equivalent (compare via `torch.equal`, not `allclose` — strict).

## Step 9 — Inference-time entropy floor

- [x] Add `_apply_entropy_floor(pi, feasible_mask, n, floor_frac)` helper in `src/agents/learned/bidpolicy/network.py` per design §4.1 (closed-form bisection over α).
- [x] Wire into `DistilledBidPolicy.bid_dist`: if `info.pool_size in self.net.config.floor_frac`, replace `pi_bids` with floored version before recomputing entropy + support.
- [x] Test `tests/agents/learned/test_entropy_floor.py`:
  - `test_floor_noop_above_floor` — pi already above floor → returned pi byte-equivalent.
  - `test_floor_raises_to_target` — pi peaked → returned `H(pi') ≥ H_floor(2) - 1e-4`, sums to 1.
  - `test_floor_at_n_5_is_noop` — `floor_frac` empty for n=5 → no transformation applied.
  - `test_floor_property_n2` — 100 random pi vectors at n=2: all satisfy floor.

## Step 10 — End-to-end smoke + KL-reduction test (§7.6)

- [x] `tests/training/test_cfr_distillation_smoke.py::test_pipeline_end_to_end` (slow-marked):
  - Run `run_distillation(N=32, seed=0, run_id='phase4_smoke', trunk_ckpt=<AR-1 winner>)`.
  - Load `bid_*.npz` + `trunk_*.npz`; build a TensorDataset.
  - Compute init validation KL; train 5 epochs with `BidPolicy.loss_step`; compute final KL.
  - Assert `final_kl < 0.5 * init_kl` per parent §7.6.
- [x] Verify slow mark properly registered per [feedback_pytest_slow_mark.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_pytest_slow_mark.md).

## Step 11 — Doc updates + commit

- [x] Mark Phase 4 boxes complete in [agent_redesign_ar2_checklist.md](agent_redesign_ar2_checklist.md).
- [x] Note any deviations from the design in this checklist + propagate to [agent_redesign_ar2_phase4_design.md](agent_redesign_ar2_phase4_design.md).
- [x] Update memory `project_ar2_impl_progress.md` — Phase 4 completion date, smoke run_id, any deviations.
- [x] Commit with message `AR-2 Phase 4: CFR+ distillation pipeline + losses + entropy floor`.

---

## Out of scope (do not touch)

- ModularNashAgent wiring — Phase 5.
- Pilot run + sweep — Phase 7.
- Solver tuning, per-deal ε logging — Phase 7.
- Trunk-activation cache reuse across run_ids — non-blocking optimization.
