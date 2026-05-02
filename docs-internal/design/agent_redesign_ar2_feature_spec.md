# AR-2 Phase 3 — Feature Spec for CallPolicy / BidPolicy heads

- **Status:** fixed for Phase 3 implementation
- **Date:** 2026-05-02
- **Parent design:** [agent_redesign_ar2.md](agent_redesign_ar2.md) §4.1, §4.2, §4.3
- **Scope:** the exact byte layout of the input vectors fed into `CallPolicyNet` and `BidPolicyNet`, plus the resolution of "what is `trunk_repr(info)`" in code.

This note exists so Phase 3 (network packages) and Phase 4 (distillation pipeline) build features in the same order with no drift. It does **not** add design surface — it just pins down the slicing the design doc described prosaically.

---

## 1. Trunk representation — exact source

`trunk_repr(info)` is the **pre-head 256-d activation** of the AR-1 winner [LearnedHandModelNet](../../src/agents/learned/handmodel/network.py) — i.e. the output of `self.trunk(x)` immediately before `self.head` at line 230–231:

```python
x = self.trunk(x)          # ← this is trunk_repr; shape (B, hidden_dim)
logits = self.head(x)
```

For the pinned winner `b64-h256-n2`, `hidden_dim == 256`. Phase 3 hard-asserts this at head construction; a different HandModel checkpoint with a different hidden width fails fast rather than silently broadcasting.

**Extraction API (added in Phase 3):** a new method `LearnedHandModelNet.trunk_forward(...)` that mirrors the existing `forward` signature but returns `x` after the trunk and skips the head. The existing `forward` is refactored to call `trunk_forward` and then `self.head` so behavior is byte-identical (the AR-1 unit tests guard this).

**Why a method on the net rather than a forward hook:** hooks are stateful + global, fragile under DataLoader workers. A method is explicit, testable, and the same code path is used at training cache-build time and inference.

**Frozen for AR-2:** all trunk params (and HandModel's bid head) have `requires_grad=False`. Only the new heads' weights enter the optimizer's parameter list. Tested in Phase 6 §7.5 (trunk-freeze invariance).

---

## 2. CallPolicy input layout — 478 d

Input vector `x_call ∈ ℝ⁴⁷⁸` indexed `[0..477]`:

| Slice | Width | Source | Notes |
| --- | --- | --- | --- |
| `[0  : 256)` | 256 | `trunk_repr(info)` (float32) | from `LearnedHandModelNet.trunk_forward` |
| `[256: 366)` | 110 | `belief.q` (float32, sums to 1) | `NUM_BIDS = 110` |
| `[366: 476)` | 110 | `one_hot(info.standing_bid, NUM_BIDS)` | zeros if `standing_bid is None` |
| `[476]` | 1 | `info.pool_size / 25` | matches HandModel's `n/25` scalar |
| `[477]` | 1 | `belief.q[info.standing_bid]` | 0.0 if `standing_bid is None` — sufficient statistic for the bluff vs honest ±1 EV decision |

Output: scalar `p_call ∈ [0, 1]` via `Linear(478→64) → LayerNorm → ReLU → Linear(64→1) → sigmoid`.

**Edge cases:**

- `standing_bid is None` → both the one-hot block and the `q[bid]` scalar are zero. CallPolicy will never be queried at an opener's first move (CALL is illegal there), but the layout must still be defined for safe batched inference.
- `q` is already feasibility-masked + L1-normalized inside `LearnedHandModel.belief_batch` (lines 270–274). Heads receive a clean simplex.

---

## 3. BidPolicy input layout — 367 d

Input vector `x_bid ∈ ℝ³⁶⁷` indexed `[0..366]`:

| Slice | Width | Source | Notes |
| --- | --- | --- | --- |
| `[0  : 256)` | 256 | `trunk_repr(info)` (float32) | identical slice to CallPolicy |
| `[256: 366)` | 110 | `belief.q` (float32, sums to 1) | identical slice to CallPolicy |
| `[366]` | 1 | `info.pool_size / 25` | identical scalar to CallPolicy |

Output: distribution over `NUM_BIDS = 110` actions:

```text
hidden  = Linear(367 → 128) → LayerNorm → ReLU → Linear(128 → 110)
logits  = hidden + log(belief.q + 1e-12)              # warm-start
masked  = logits.masked_fill(~bid_mask, -inf)         # where:
                                                      #   bid_mask = info.feasible_mask[:NUM_BIDS]
pi      = softmax(masked, dim=-1)
```

`info.feasible_mask` (per [contracts.py:75-77](../../src/agents/contracts.py#L75-L77)) is already the joint legal-∩-feasible mask over `NUM_ACTIONS = 112`. The bid-only slice `[:NUM_BIDS]` excludes CALL and HH automatically, since those are at indices 110 and 111. HH and CALL live on the `BidPolicy ⊥ CallPolicy ⊥ HHGate` boundary, not inside the bid distribution.

**Final-layer init:** `nn.init.orthogonal_(linear_2.weight, gain=0.01)` and `zeros_(bias)`. With `hidden ≈ 0`, `logits ≈ log q + 0`, so the pre-training policy is `softmax(log q) = q / Z(mask) ≈ q` after masking — i.e. "bid the most likely true hand," matching parent §4.3.

---

## 4. Numerical conventions

- **dtype:** float32 throughout. The AR-1 trunk emits float32; we don't promote.
- **Endianness / ordering:** all arrays are C-contiguous. The concat order in §2 and §3 is the source of truth; tests verify by construction.
- **Trunk-frozen guarantee:** Phase 3 head packages call `for p in trunk.parameters(): p.requires_grad_(False)` at construction *and* exclude trunk params from the optimizer. Both belt and braces because `requires_grad=False` alone wouldn't catch an Adam state with a reference held over a checkpoint round-trip.

---

## 5. Trunk activations: when computed, when cached

Per design §4.1 and the user-confirmed Phase 3 plan:

- **Phase 3:** uncached forward — every head call re-runs the trunk. Acceptable because Phase 3 ships only smoke-test code, not the distillation loop.
- **Phase 4:** the distillation pipeline (`src/training/cfr_distillation.py`) precomputes a `(num_infostates, 256)` trunk-activation tensor once per shard and writes `trunk_<shard>.npz` alongside `call_<shard>.npz` / `bid_<shard>.npz`. The trainer dataloader concatenates the three slices on the fly using the layout above.
- **Inference (post-AR-2 acceptance):** `ModularNashAgent` calls `trunk_forward` once per `Infostate` it sees in a match. No persistent cache — matches are short, and the model is small enough that one extra forward per decision is cheap.

---

## 6. What is *not* in this spec

- The full `LearnedHandModelNet.trunk_forward` implementation — that's a 5-line refactor in Phase 3.
- The `HHGate` is not a feature; it runs **before** the heads and short-circuits to a degenerate HH distribution. Heads are conditioned on `hh_fired=False`.
- Loss functions, entropy regularization, inference-time entropy floor — those are Phase 4 / Phase 5 concerns and live in design §5.2 / §5.3.
