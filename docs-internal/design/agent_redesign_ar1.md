# AR-1 Sub-design — HandModel architecture and Phase-A pretrain

- **Status:** Draft (design-first gate; implementation lands in a follow-up session)
- **Date:** 2026-04-29
- **Owner:** main
- **Parent plan:** [agent_redesign_plan.md](agent_redesign_plan.md) §AR-1
- **Parent design:** [agent_redesign.md](agent_redesign.md) §4.1, §6 Phase A, §8.2
- **Predecessor commit:** AR-0a + AR-0b at `fb17d03`

This sub-design fixes the AR-1 open questions the plan flagged: network
shape, bid-history encoder, tokenization, dataset size and split,
calibration metric, early-stop criterion. Once landed, the AR-1
implementation session imports the AR-0a contracts (`HandBelief`,
`HandModel` Protocol, `Infostate`) and writes only the new package
`src/agents/learned/handmodel/` plus its sweep config.

## Goals

- Produce one trained `LearnedHandModel` checkpoint that strictly
  Pareto-dominates the analytic conditional table on Brier across
  `n ∈ {5..25}` on a held-out set, while preserving the AR-0a mask
  invariants (`q[~feasible].sum() < 1e-6`).
- Ship `AnalyticHandModel` and `HeuristicHandModel` adapters in the
  same package so the §9.4 ablation matrix is non-empty on day one.
- Keep training Phase-A-only (supervised cross-entropy, frozen rollout
  policy). No CFR, no R-NaD, no auxiliary heads. Phase A is a *prior*,
  not the final agent.
- Reuse the AR-0b sweep harness for architectural search; do not invent
  a new training driver.

## Non-goals

- CallPolicy / BidPolicy heads (AR-2).
- Bayesian-consistent training against an evolving opponent (parent
  §4.1 option 2; deferred to Phase C as the auxiliary loss).
- Trunk-sharing across heads. AR-1 ships a stand-alone HandModel; the
  AR-2 design will decide whether to factor a shared trunk out.
- 5-player generalisation (AR-5). The seat-relative encoding is chosen
  here so it lifts cleanly, but only 1v1 is trained and evaluated.

---

## 1. Target and label

The output space is the existing 110-bid lattice (`NUM_BIDS = 110`). The
label for a decision at infostate `info` is the bid index of the
**pool's best 5-card hand** at the end of the deal — i.e.
`bid_to_index((pool_best_type, pool_best_primary_rank))`. This is what
`ExactRulesConditional` already targets implicitly and what
`extended_conditional_exact_probs.json` tabulates.

Single label per decision (not per round): we record one
`(infostate, true_pool_index)` row at every decision point in the
rollout, but the label is fixed across all decisions in the same deal
(the pool doesn't change). A turn-7 decision and a turn-3 decision in
the same deal share a label; what differs is the infostate features
(more bid history at turn 7).

This is supervised classification with `NUM_BIDS = 110` classes and a
hard mask: at infostate `info`, classes `i` with `info.feasible_mask[i]
== False` (restricted to bid actions, indices `0..109`) get logit
`-inf` before softmax. Cross-entropy is computed on the masked
distribution. The label is **always** in the feasible set — every
true-pool index is feasible at any `n ≥ 1`.

## 2. Architecture

### 2.1 Encoders

Three encoders, concatenated into a flat trunk input. None reuse the
existing `LiarsPokerNet` directly — that net has a value head and a
warm-start lookup we explicitly do not want for AR-1 — but the
patterns are lifted:

| Encoder | Input | Output dim | Notes |
| --- | --- | --- | --- |
| Card encoder | `info.own_hand` (1..5 card ids) | `card_emb_dim = 32` | `nn.Embedding(52, 32)` summed (DeepSet). Permutation-invariant over cards. Empty hand → zero vector (only happens during 5p with eliminated seats; not in 1v1). |
| Bid-history encoder | sequence of `(seat_offset, bid_idx)` tokens | `bid_hist_dim = 64` | See §2.2 — small Transformer with one layer over the full sequence. |
| Scalar encoder | 5 floats: `n/25`, `own_hand_size/5`, `avg_opp_size/5`, `is_first_bidder`, `round_position` | 5 | No projection; concatenated raw. (Removed `round_index` from the legacy net — single-round only per parent §2 non-goals.) |

Trunk input dim: `32 + 64 + 5 = 101`. (No warm-start features; AR-1
must learn the conditional on its own to test whether bid-history
conditioning buys headroom over the analytic table.)

### 2.2 Bid-history encoder choice

**Choice: 1-layer Transformer encoder, 4 heads, model dim 32, FFN dim
64, sequence length 16, mean-pooled.** Reasons:

- The bid sequence has *order* (a Pair-of-Aces opener carries different
  information than a Pair-of-Aces response to a Trips bid) and
  *content over a small vocabulary* (`NUM_BIDS + 2` tokens). This is
  the exact regime where small Transformers beat RNNs on data
  efficiency and stability.
- Permutation-equivariance over **bid positions within the same seat**
  is not a property we want — order matters. So `nn.Embedding` per
  bid-index, learned positional embedding per slot, learned
  seat-offset embedding, summed.
- One layer is sufficient: the longest 1v1 round has ~10 bids; deeper
  attention is overkill at this length. Width is preferred over depth
  here (cheap parameters; minor variance reduction).
- Mean-pool over non-pad positions (not `[CLS]`): cheaper, no special
  token to break invariants under sequence-length variation.

We will **not** ship an RNN variant. The sweep can compare the
Transformer encoder against `bid_hist_dim = 0` (i.e. no bid-history
conditioning at all) — that is the directly informative ablation
because it isolates what the bid-history is buying.

### 2.3 Tokenisation

Each prior bid in `info.bid_history` is mapped to two integer tokens
fed into separate embedding tables, summed:

- **Bid token.** `bid_idx ∈ {0..NUM_BIDS-1}`, plus `CALL_TOKEN =
  NUM_BIDS` (parent §4 reserves this; though calls only end rounds, we
  include the token for forward compatibility with multi-round work)
  and `PAD_TOKEN = NUM_BIDS + 1`. Vocabulary size `NUM_BIDS + 2`.
  `bid_emb_dim = 32`.
- **Seat-offset token.** `seat_offset = (bidder_seat - own_seat) mod
  num_seats` ∈ `{0..4}` (1v1 uses only `{0, 1}`; 5p uses `{0..4}`).
  Vocabulary size 5. Always relative to the actor — never absolute.
  This is the parent §3 / §7.3 generalisation lever: a 1v1-trained
  net runs unchanged at 5p inference, only with more non-zero offsets
  in the input. `seat_emb_dim = 8`.
- **Position token.** Slot index `0..bid_hist_len - 1`. Learned
  positional embedding, dim 32. Right-aligned (most recent bid at the
  highest position) and left-padded — matches `LiarsPokerNet`'s
  convention.

The three embeddings are summed (not concatenated) to `model_dim = 32`,
then run through the Transformer encoder.

`bid_hist_len = 16` covers >99% of observed 1v1 rounds (longest
recorded round in 2026-04 benchmarks: 11 bids); truncation of the
oldest tokens is acceptable for the rare overrun.

### 2.4 Trunk and head

- Trunk: 2 × `Linear → LayerNorm → ReLU`, hidden dim 128. Orthogonal
  init, gain 0.5, matches `LiarsPokerNet`.
- Head: `Linear(128 → NUM_BIDS = 110)`. Logits are masked to
  `info.feasible_mask[:NUM_BIDS]` *before* softmax. Mask values: `0`
  for feasible, `-inf` for infeasible (added to logits — same pattern
  as `_mask_logits` in [src/agents/learned/rnad/network.py:363](../../src/agents/learned/rnad/network.py#L363)).

Total parameter count target: under 200k. (Reference: `LiarsPokerNet`
at default config is ~250k including value head.) Small models train
faster, sweep faster, and the AR-1 task — pool-distribution inference
under public information — is not high-capacity. The architectural
sweep can push capacity up if the floor model under-fits.

### 2.5 Sweep axes for the architectural search

The mandatory sweep config `configs/sweeps/ar1_handmodel_arch.yaml`
varies (axis-product, 12 cells):

- `hidden_dim ∈ {64, 128, 256}`
- `num_trunk_layers ∈ {1, 2}`
- `bid_hist_dim ∈ {0, 64}` — the directly informative ablation

All other knobs (dropout, learning rate, batch size, epochs) are fixed
defaults; the sweep is **architecture only**, not hyperparameter
search. Hyperparameter tuning, if needed, is a follow-up sweep against
the winning architecture.

## 3. Dataset

### 3.1 Source policy

**`ExactRulesAdaptive` self-play, 1v1, exact rules + HH enabled.** This
is the strongest current heuristic ladder agent
([memory: project_benchmark_results.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/project_benchmark_results.md))
and produces realistic bid-history distributions. Random play would
under-represent late-round infostates; CFR-derived play would beg the
question (we'd be evaluating HandModel on the exact distribution
AR-2's CFR+ produces).

### 3.2 Hand-size mixture

25% each of `(n_self, n_opp) ∈ {(2,2), (3,3), (4,4), (5,5)}` — i.e.
both seats have the same hand size, four equally weighted pool sizes.
Asymmetric sizes (`n_self != n_opp`) are deferred to AR-5 where
elimination matters; for AR-1 this is single-round 1v1 with both
players starting full.

### 3.3 Volume

**1.0M decisions total** in the training set, 100k in val, 100k in
held-out test. (The plan's "~10⁶ decisions" budget.) Roughly 30 bids
per game across the four pool sizes ≈ 35k games for training, ~3.5k
each for val and test. CPU rollout cost: ~10 minutes per pool size at
1k games/min.

The label distribution is unbalanced — high pool sizes produce more
decision points per game *and* shift the pool-best distribution. We
report Brier per-`n`, never pooled, so the imbalance is visible rather
than hidden.

### 3.4 Split strategy

Split **by deal**, not by decision. All decisions from the same deal
go to the same split. Otherwise the model trivially memorises the
deal's pool composition by seeing earlier decisions in the deal during
training — they share the label by construction. Splits are seeded:
`train` = first 35k seeded deals per `n`, `val` = next 3.5k, `test` =
next 3.5k. Reproducible from the seed alone.

### 3.5 On-disk format

`data/runs/<run_id>/rollouts/{train,val,test}/<n>.jsonl.gz` — one row
per decision:

```json
{"infostate": { ... §8.1 JSON form ... }, "label": 42, "deal_id": 17231}
```

Re-uses the AR-0a `Infostate` JSON serialiser from `contracts_io.py`.
`deal_id` is kept on the row so a later sanity audit can confirm the
split discipline.

## 4. Training

### 4.1 Loss

Masked cross-entropy. Implementation:

```python
logits = head(trunk_out)                   # (B, NUM_BIDS)
logits = logits.masked_fill(~feasible, -INF)
loss   = F.cross_entropy(logits, labels)   # standard log-softmax + NLL
```

No label smoothing. The label space is small (110), the labels are
exact (no annotation noise), and label smoothing distorts the
calibration metric we are explicitly optimising.

### 4.2 Optimisation defaults

- Adam, lr `1e-3`, betas default, weight decay 0.
- Batch size 512.
- Linear warmup over the first 1000 steps; constant after.
- Max 30 epochs; early stop (§4.3).
- Device: CPU per
  [feedback_mps_cpu_speed.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_mps_cpu_speed.md).
  HandModel pretrain is supervised + batched, so MPS may actually be
  competitive here — but we hold the project default until measured;
  flagged as a Phase-A follow-up.

### 4.3 Early stop

**Stop when val loss has not improved by ≥ 0.5% over the last 3
epochs.** Min epochs: 5 (avoid stopping during warmup-driven noise).
Max epochs: 30. Save the best-val checkpoint (not the latest).

The 0.5% threshold is loose because the absolute loss is small (well
under 3.0 nats given mask + label distribution); tighter thresholds
yield false-positive plateaus on noisy val curves.

## 5. Calibration metric (the AR-1 acceptance metric)

### 5.1 Metric

**Per-`n` Brier score on the masked posterior**, averaged over the
held-out test set:

```
Brier(n) = mean over (info, label) with info.pool_size == n of
           sum_i (q_i - 1[i == label])^2
```

Reasons for Brier over NLL:

- Brier is bounded `[0, 2]`; NLL has no upper bound and is dominated by
  rare hard cases. We want a metric that summarises the typical case.
- Brier is a proper scoring rule (calibrated; rewards honest
  probabilities) and decomposes cleanly into reliability + resolution
  + uncertainty. Reliability is precisely what "well-calibrated `q`"
  means in the parent design.
- Brier on the masked distribution is well-defined; NLL on a masked
  distribution where the label probability is near 0 explodes.

NLL is reported alongside (it is what we minimise during training) but
acceptance is on Brier.

### 5.2 Per-`n` reporting, not pooled

`n ∈ {5..25}` — pool sizes that occur in 1v1 play under the §3.2
mixture. We log Brier separately for each `n` and require strict
Pareto-dominance: `LearnedHandModel.Brier(n) ≤ AnalyticHandModel.Brier(n)`
**for every `n`**, by at least the val-noise floor measured in the
sweep (one standard deviation across the 12 architecture cells).

Pooled Brier hides the historical failure mode where a learned model
beats analytic at `n = 25` (where bid-history is informative) but
regresses at `n = 5` (where the analytic table is already exact and
small data hurts).

### 5.3 Held-out set construction

Test set is the §3.4 100k-decision held-out split, never seen during
training, val, or sweep selection. The sweep's winning config is
chosen on **val** Brier; test Brier is computed *once*, on the
selected config, and is the number that goes into the AR-1 acceptance
report.

## 6. Adapters: Analytic and Heuristic flavours

Both ship in `src/agents/learned/handmodel/baselines.py`, both
implement the AR-0a `HandModel` Protocol.

### 6.1 `AnalyticHandModel`

Wraps `WarmStartLookup.get_features` and exposes the conditional row
as a `HandBelief`. Implementation:

```python
def belief(self, info: Infostate) -> HandBelief:
    _, c_vec, _ = self._lookup.get_features(info.own_hand, info.pool_size)
    feasible = np.asarray(info.feasible_mask[:NUM_BIDS], dtype=bool)
    q = c_vec.astype(np.float32) * feasible
    q /= max(q.sum(), 1e-12)
    q_logits = np.where(feasible, np.log(q + 1e-12), -np.inf).astype(np.float32)
    return HandBelief(q=q, q_logits=q_logits, feasible_mask=feasible, n=info.pool_size)
```

Bid-history is ignored by construction — this is the analytic floor.
Used as the Brier baseline in §5 and as a `--handmodel analytic` cell
in the §9.4 ablation matrix.

### 6.2 `HeuristicHandModel`

Extracts `_compute_adj_exact` from
[ExactRulesOpponentModelAgent](../../src/agents/registry.py) behind
the `HandModel` Protocol. Bid-history *is* used (this is the heuristic
ladder's opponent-bump logic). Useful as a third point in the
calibration plot to confirm the heuristic bumps actually help (or
don't) in Brier terms.

Both adapters are pure-Python and need no checkpoint — they
participate in sweeps via a registry entry, not a `load_component`
call.

## 7. Tests

Under `tests/agents/learned/handmodel/`:

1. **Mask invariant** (mandatory).
   Property test: 1000 random `Infostate`s built from random
   `MatchState`s. For each, `LearnedHandModel.belief(info).q[~feasible]`
   sums to less than `1e-6` and `q.sum()` is within `1e-5` of 1.0.
2. **Calibration vs. analytic** (acceptance test).
   Per-`n` Brier on a fixed 10k-decision sample of the test set.
   `LearnedHandModel.Brier(n) ≤ AnalyticHandModel.Brier(n) - noise_floor`
   for every `n`. Marked `@pytest.mark.slow` per
   [feedback_pytest_slow_mark.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_pytest_slow_mark.md).
3. **Batching parity.** `belief_batch([i1, i2, ...])` returns
   `HandBelief`s elementwise equal to `[belief(i1), belief(i2), ...]`
   within `1e-6`. Catches the historical "batched inference takes a
   different code path and silently disagrees with looped" bug.
4. **Adapter parity.** `AnalyticHandModel` produces the same `q` as
   the existing `ExactRulesConditional` agent's internal probability
   on a 100-deal random sample (modulo mask normalisation). Catches
   `WarmStartLookup` API drift.

## 8. Acceptance gate (mirrors the plan's AR-1 gate)

- **Mask invariant** holds on a 10⁵-sample property test (raised from
  the plan's 10³ — it's free, and the bug class is silent).
- **Pareto-dominance** of `AnalyticHandModel` on per-`n` Brier across
  `n ∈ {5..25}` on the held-out test set, by at least the val-noise
  floor.
- The architectural sweep produces a single recommended config
  written to `configs/agents/handmodel_v1.yaml` with its `run_id` and
  `git_sha` pinned (per AR-0b reproducibility floor).
- `MEMORY.md` has a `project_ar1_handmodel.md` entry.
- CHANGELOG entry.

## 9. Open questions (deferred to follow-up sub-design or AR-2)

1. **Trunk sharing across heads.** AR-1 ships a stand-alone HandModel.
   Whether AR-2 mounts CallPolicy / BidPolicy on the same trunk via
   the parent §7.8 `--load-trunk` flag, or on independent trunks, is
   an AR-2 question. AR-1 keeps the trunk-output API
   (`forward_trunk(info) -> Tensor[hidden_dim]`) public so it can be
   reused either way.
2. **Auxiliary HandModel loss in Phase C.** Parent §6 Phase C wants an
   auxiliary HandModel loss against fresh self-play pool outcomes.
   That loss reuses this same head; the AR-3 design will decide
   freeze schedule and weighting.
3. **MPS/CPU bench for the supervised loop.** The CPU-default
   memory entry was measured under R-NaD's sequential rollout
   workload. AR-1 is batched supervised — possibly different. Add a
   one-cell sweep `device ∈ {cpu, mps}` *after* the architectural
   sweep lands; do not block AR-1 on it.
4. **Bid-history truncation policy.** `bid_hist_len = 16` covers >99%
   of 1v1 rounds. Whether to right-truncate or fold older bids into a
   summary token is open; AR-5 may force the question if 5p rounds
   exceed 16 bids more often. For AR-1, simple right-truncation.

## 10. Done definition for AR-1

This sub-design is *fully executed* when the plan's AR-1 checkboxes
are all `[x]`, the acceptance gate (§8) passes, and the AR-2
sub-design can open with "AR-1 produced
`configs/agents/handmodel_v1.yaml`; freeze that checkpoint as the
chance prior."
