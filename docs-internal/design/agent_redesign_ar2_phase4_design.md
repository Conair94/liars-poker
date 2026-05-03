# AR-2 Phase 4 Design — CFR+ distillation pipeline

- **Status:** Draft (design-first gate; implementation lands in this same session per user instruction)
- **Date:** 2026-05-03
- **Scope:** AR-2 Phase 4 only — distillation data pipeline + loss bodies + inference-time entropy floor
- **Parent design:** [agent_redesign_ar2.md](agent_redesign_ar2.md) §5–§6
- **Parent checklist:** [agent_redesign_ar2_checklist.md](agent_redesign_ar2_checklist.md) Phase 4
- **Predecessors in this session series:** Phase 1 (`CFRPlusSubgameSolver`, c0952b5/`fb17d03`), Phase 2 (HH gate, `ec041bb`), Phase 3 (network packages + trainer stubs, `c6cb672`)

This sub-design fixes the implementation-level questions the parent left under-specified for Phase 4. It does not re-derive the loss form (parent §5.2/§5.3) or the deal-budget sweep (parent §3 — that's Phase 7).

## 1. What Phase 4 produces

Concrete deliverables, in order of dependency:

1. **`src/training/cfr_distillation.py`** — orchestrates: deal sampling → solver call → walk avg-strategy → write sharded `.npz` → trunk-activation precompute pass.
2. **Filled `loss_step` in [src/agents/learned/callpolicy/trainer.py](../../src/agents/learned/callpolicy/trainer.py)** — BCE per parent §5.2.
3. **Filled `loss_step` in [src/agents/learned/bidpolicy/trainer.py](../../src/agents/learned/bidpolicy/trainer.py)** — forward KL + entropy regularizer per parent §5.3.
4. **Inference-time entropy floor** — modify `DistilledBidPolicy.bid_dist` to renormalize `pi` against `H_floor(n)` per parent §5.3.
5. **Phase 4 tests** — distillation reduces KL (§7.6), trunk-freeze invariance (§7.5), entropy floor at `n=2` (§7.2 — slow). The other §7 tests are owned by Phase 5/6.

Phase 4 does **not** wire `ModularNashAgent` (Phase 5) and does **not** run the sweep (Phase 7). It produces a **callable end-to-end pipeline** demonstrated by a tiny smoke run (~32 deals).

## 2. Distillation data pipeline

### 2.1 `sample_deals_mixture` — wrapper around the existing sampler

Existing [src/training/metrics/deal_sampler.py:91](../../src/training/metrics/deal_sampler.py) takes a single `hand_size`. Parent §3.1 needs a 25/25/25/25 mix over `n ∈ {4, 6, 8, 10}` total pool size — i.e. per-seat hand_size `∈ {2, 3, 4, 5}`.

Decision: **don't modify `sample_deals`**. Add a thin wrapper `sample_deals_mixture(N, seed, mix)` in `cfr_distillation.py` that allocates `N · mix[k]` deals per seat-size, calls `sample_deals` four times with deterministic per-bucket seeds (`seed + k*7919`), and yields `(deal, n_total)` pairs in interleaved order. Stratification (weak/mid/strong on P0's hand) stays on per call.

### 2.2 Per-deal walk over avg-strategy support

Given `SubgameSolution` from `CFRPlusSubgameSolver.solve(hands)`:

```python
def walk_avg_strategy(sol: SubgameSolution, hands) -> Iterator[Row]:
    # BFS from ROOT=(None, 0).
    # For each visited decision state s = (cur_bid_idx, current_player):
    #   - Emit one CallRow:  (s, hands, target = sol.avg_call_prob[s])
    #   - Emit one BidRow:   (s, hands, target = sol.avg_bid_dist[s])     # rows sum to 1 minus call_prob
    #   - For each successor bid b with sol.avg_bid_dist[s][b] > 0
    #     and not hh_gate_at(b): enqueue (b, 1 - current_player)
    # Yields each state once (BFS visited set).
```

We **emit both heads' rows from the same state**. Reach probability is *not* used to weight rows — every visited state is one row, equally weighted. This matches DeepStack-style distillation and avoids the variance from importance-sampling reaches that early CFR distillation papers found problematic. Reach-weighted distillation is a Phase 7+ optimization if the elbow analysis demands it.

#### 2.2.1 Forced-HH labeling

Per parent §5.1, at the *root* (and any node) where `hh_gate_at(cur_bid_idx)` would fire on the *true pool*, that node is a forced HH terminal — it doesn't appear as a decision state in `SubgameSolution`. The walker therefore never visits it. Nothing extra to do; the gate's already baked into the solver's state enumeration ([src/training/cfr/subgame_solver.py:268-277](../../src/training/cfr/subgame_solver.py#L268-L277)).

The CallPolicy / BidPolicy heads are *only* trained on `hh_fired=False` infostates by construction. At inference, `should_declare_hh` is dispatched first (Phase 5) so the heads never see HH-fired states.

### 2.3 Row schema (sharded `.npz`)

Each shard file is one of three flavors. Sharding is `deal_idx % 64` per parent §6.

**`call_<shard>.npz`** (one row per visited state per deal):
| field           | dtype     | shape          | notes |
|-----------------|-----------|----------------|-------|
| `deal_idx`      | int32     | (R,)           | for split bookkeeping |
| `state_player`  | int8      | (R,)           | current_player (0 or 1) |
| `cur_bid_idx`   | int16     | (R,)           | -1 sentinel for None (root) |
| `hand_p0`       | int8      | (R, max_hand)  | -1-padded; max_hand=5 |
| `hand_p1`       | int8      | (R, max_hand)  | -1-padded |
| `pool_size`     | int16     | (R,)           | n |
| `target_call`   | float32   | (R,)           | `avg_call_prob[s]` ∈ [0,1] |

**`bid_<shard>.npz`** — same per-row identity columns plus:
| field            | dtype   | shape          |
|------------------|---------|----------------|
| `target_bid`     | float32 | (R, NUM_BIDS)  | normalized to row-sum 1 over feasible bids (after dropping the call-mass) |
| `feasible_mask`  | bool    | (R, NUM_BIDS)  | from `Infostate.feasible_mask[:NUM_BIDS]` |

We materialize `Infostate` from `(hands, cur_bid_idx, current_player)` via `build_canonical_match_state` + the existing `state_bridge` adapters (already used in Phase 3 trainers) at row-emission time. This avoids storing a serialized Infostate in the `.npz`.

**`trunk_<shard>.npz`** (precompute pass, written *after* call/bid shards):
| field         | dtype   | shape          |
|---------------|---------|----------------|
| `trunk_repr`  | float32 | (R, 256)       |
| `q`           | float32 | (R, NUM_BIDS)  | from HandModel forward on the same Infostate |

Rows in `trunk_<shard>.npz` are aligned 1:1 with `bid_<shard>.npz` (same R, same order). CallPolicy uses the same `trunk_repr` + `q` — we don't duplicate them.

### 2.4 Split by deal, never by row

Parent §6 requires 80/10/10 by *deal*. Implementation: a fixed split file `data/runs/<run_id>/cfr_deals/split.json` listing `{train: [deal_idx, ...], val: [...], test: [...]}` written once at sampling time (`hash(deal_idx + run_id) % 10 ∈ {0..7}/8/9`). DataLoader filters rows by `deal_idx` membership at load time.

### 2.5 Failure modes / budget caps

- **CFR+ doesn't converge on a deal.** `compute_eps=False` by default in distillation (the iter cap is the budget). Solver always returns *some* avg-strategy after `max_iters` iters. Quality is tracked by the §7.6 KL test, not per-deal ε. The pilot run (Phase 7, separate session) will introduce per-deal ε logging; Phase 4 doesn't need it.
- **`max_iters` default for distillation:** 500 iters (matches Phase 1 default). On a 32-deal smoke run at n=10 this is ~1–2 s/deal on CPU — acceptable for the smoke test.
- **Empty avg_bid_dist row.** Some states are never visited (call-only at the root if call_prob == 1.0). We skip them — no row emitted.
- **Memory.** A 32-deal smoke produces ~32 × 10 ≈ 320 rows total — single shard, well under 1 MB.

## 3. Loss bodies

### 3.1 CallPolicy — BCE

```python
def loss_step(state, features, targets):  # features: (B, 478), targets: (B,) ∈ [0,1]
    logits = state.net._raw_logits(features)  # need a sigmoid-free entry point
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
```

We need a `_raw_logits` accessor on `CallPolicyNet` that returns `fc2(relu(ln(fc1(x))))` *without* the sigmoid, so we can use `binary_cross_entropy_with_logits` (numerically stable) instead of `binary_cross_entropy` on the squashed output. Add as a method; `forward` continues to return the sigmoid for inference.

`targets` are soft labels in [0, 1]. PyTorch's BCE-with-logits accepts soft targets directly — no quantization.

### 3.2 BidPolicy — forward KL + entropy regularizer

```python
def loss_step(state, features, log_q, bid_mask, targets, pool_size):
    # features: (B, 367), log_q: (B, NUM_BIDS), bid_mask: (B, NUM_BIDS) bool,
    # targets: (B, NUM_BIDS) probs (rows sum to 1 over feasible), pool_size: (B,) int.
    masked_logits = state.net(features, log_q, bid_mask)        # (B, NUM_BIDS), -inf on infeasible
    log_pi        = F.log_softmax(masked_logits, dim=-1)        # -inf on infeasible
    # KL(target || pi) = sum_a target * (log target - log pi); the log-target term is
    # constant w.r.t. parameters → drop it (cross-entropy form).
    ce  = -(targets * log_pi).sum(dim=-1)                       # (B,)
    pi  = log_pi.exp()
    H   = -(pi * log_pi.clamp_min(-30.0)).sum(dim=-1)           # (B,) — clamp avoids -inf*0=NaN
    beta = state.net.config.beta_max * (1.0 - pool_size.float() / 5.0).clamp_min(0.0)
    return (ce - beta * H).mean()
```

Notes:
- `targets[..., a] = 0` whenever `bid_mask[..., a] == False` (the solver respects the same mask). The `0 * log_pi` term contributes 0 even when `log_pi == -inf`, which Torch handles correctly because masked entries of `targets` are `0.0` exactly (not approximate).
- The clamp on `log_pi` for the entropy term is purely defensive: `pi==0` rows may have `log_pi==-inf`; `0 * -inf` is NaN in IEEE-754. Clamping `log_pi` to ≥ -30 makes the contribution effectively zero without producing NaN gradients.
- Per-deal pool size `n` enters via `beta(n) = beta_max * max(0, 1 - n/5)`. At n=5 → β=0; at n=10 → β=0; at n=2 → β=0.6·beta_max. Matches parent §5.3.

### 3.3 Where the warm-start sits

Already wired in Phase 3 — `BidPolicyNet.forward` adds `log_q` to its raw `hidden` output before masking. The trainer doesn't need to do anything extra; the loss above closes over `BidPolicyNet.forward` and the warm-start is automatic.

## 4. Inference-time entropy floor (parent §5.3, finalized)

Parent §5.3 specifies: at inference, add `α · uniform_over_feasible` to `pi` and renormalize, where α is the smallest constant such that `H(pi') ≥ H_floor(n) = floor_frac(n) · H(uniform_over_feasible)`.

### 4.1 Closed-form α

Let `pi` be the network's softmax output (already feasibility-masked, sums to 1). Let `u = feasible_mask / |feasible|` (uniform over feasible). Define the mixture:

```
pi'(α) = (1 - α) · pi + α · u    for α ∈ [0, 1]
```

This is a probability vector for any α ∈ [0, 1] (no renormalization needed — both `pi` and `u` already sum to 1, and the mixture is a convex combination). `H(pi'(α))` is concave and monotone non-decreasing on `α ∈ [0, α*]` where α* is the entropy-maximizing point (which equals 1 for this convex mixture, since pi'(1) = u maximizes entropy over the feasible support).

So the smallest α achieving `H(pi'(α)) ≥ H_floor(n)` is:
- `α = 0` if `H(pi) ≥ H_floor(n)` already.
- Else: 1-D bisection over `α ∈ (0, 1]` until `|H - H_floor| < 1e-4`. Bisection converges in ~14 iters; cost is negligible vs. trunk forward.

### 4.2 When to apply

Only when `pool_size <= 4` (`floor_frac` is empty for n ≥ 5). Implementation: in `DistilledBidPolicy.bid_dist`, after the softmax, if `info.pool_size in self.net.config.floor_frac`, apply the floor; else return the network output unchanged.

### 4.3 Entropy logged on the returned `BidDistribution`

The contract field `entropy` is currently computed from the network's pi. After the floor is applied, recompute `entropy` and `support_size` on the floored distribution so callers see the actual sampled-from distribution.

## 5. Tests added in Phase 4

| Test                                | Cost  | File                                                              |
|-------------------------------------|-------|-------------------------------------------------------------------|
| Distillation pipeline smoke (32 deals end-to-end produces non-empty shards) | <30s  | `tests/training/test_cfr_distillation_smoke.py` |
| CallPolicy BCE step decreases loss on synthetic batch | <5s   | `tests/agents/learned/test_phase4_losses.py`                      |
| BidPolicy KL+entropy step decreases loss on synthetic batch | <5s   | same                                                              |
| Trunk-freeze invariance (§7.5) — L2 norm unchanged after one epoch | <30s  | same                                                              |
| Distillation reduces KL ≥ 50% (§7.6) on a single 32-deal run | ~1min | `tests/training/test_cfr_distillation_smoke.py` (slow-marked)     |
| Entropy floor: 100 random `pi` at n=2 satisfy `H(pi') ≥ H_floor(2)` | <5s   | `tests/agents/learned/test_entropy_floor.py`                      |

The full 1000-deal §7.2 property test stays deferred to Phase 6 (too slow for the Phase 4 commit).

## 6. Out of scope for Phase 4

- `ModularNashAgent` wiring — Phase 5.
- Pilot run + sweep — Phase 7.
- Acceptance gate evaluation — Phase 8.
- Trunk-activation cache *re-use* across multiple training runs (Phase 4 writes the cache once per `run_id`; sharing across runs would need a content-addressed cache layer that's an unrelated optimization).

## 7. Open questions for Phase 4 implementation

1. **Where do `Infostate`s come from at distillation time?** We have `build_canonical_match_state(hands, cur_bid_idx, current_player) -> MatchState`. The state→Infostate adapter lives in `src/agents/state_bridge.py` (used by Phase 3 — verified by the smoke tests). Implementation imports that adapter; no new bridge needed.

2. **Run-id convention for the smoke test.** Use `run_id="phase4_smoke"` under `data/runs/phase4_smoke/cfr_deals/`. Real distillation runs (Phase 7) will use the AR-0b sweep harness's run-id allocator.

3. **Should the distillation script also train?** No — keep `cfr_distillation.py` purely a *data producer*. Training is invoked separately via the trainers (which the Phase 4 smoke test exercises via a thin adapter that reads the shards into a TensorDataset). This separation lets Phase 7's sweep parallelize sampling and reuse the dataset across multiple training-config cells.

---

**Stopping the design here.** Implementation proceeds against the Phase 4 checklist next.
