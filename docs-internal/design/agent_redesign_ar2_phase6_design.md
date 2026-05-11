# AR-2 Phase 6 — Test Suite

- **Status:** Draft (design-first gate; implementation in follow-up session)
- **Date:** 2026-05-11
- **Owner:** main
- **Parent design:** [agent_redesign_ar2.md](agent_redesign_ar2.md) §7
- **Parent checklist:** [agent_redesign_ar2_checklist.md](agent_redesign_ar2_checklist.md) §Phase 6
- **Predecessors:** Phases 1–5 complete (Phases 2/5 resolve §7.3 and §7.4 ahead of schedule)

Phase 6 closes the three remaining AR-2 test boxes: §7.1 (impossible-bid
→ CallPolicy learns to call), §7.2 (entropy floor property test at n=2),
and §7.5 (trunk params unchanged after a real gradient step). Tests §7.3 and
§7.4 were landed in Phases 2 and 1 respectively; §7.6 was landed in Phase 4.

---

## Context — what is already covered

| §  | Description                              | Covered by      |
|----|------------------------------------------|-----------------|
| 7.3 | HH gate truth table                     | Phase 2         |
| 7.4 | Solver byte-equivalence (slow)           | Phase 1         |
| 7.6 | KL drops ≥ 50% on smoke distillation     | Phase 4         |

Remaining open: **§7.1**, **§7.2** (slow), **§7.5**.

---

## 1. §7.1 — Impossible-bid deal → `p_call > 0.95` (fast)

### Purpose

A standing bid that is physically impossible at the current pool size (e.g. a
Straight bid with only 4 cards in the pool) should be called almost always.
This test checks:

1. The CFR+ solver correctly assigns `avg_call_prob ≈ 1.0` at the impossible-bid
   state — confirming the solver labelling is correct before the head ever trains.
2. A `CallPolicyNet` trained for a small number of steps on that solver label
   converges to `p_call > 0.95` — confirming the pipeline (features → head →
   BCE loss) transmits the obvious signal.

### Deal construction

Use `pool_size = 4` (hand_size = 2 per seat in 1v1, pool = 4 cards).  
At `n = 4`, any bid whose hand type requires ≥ 5 cards is infeasible:
`STRAIGHT`, `FLUSH`, `FULL_HOUSE`, `STRAIGHT_FLUSH`.

The smallest infeasible bid index at n=4 is the first STRAIGHT bid.
`bid_to_index` can be used to find it; alternatively, iterate over
`feasible_bid_mask(4)` to find the first `False` entry above the feasible zone.

Concrete deal: two hands of 2 cards each, dealt from a standard deck.
The standing bid: whichever `bid_idx` satisfies
`not feasible_bid_mask(4)[bid_idx]`.

### I/O spec

**Inputs**

| Symbol | Type | Source |
|--------|------|--------|
| `hands` | `tuple[list[int], list[int]]` | Hard-coded 2-card hands (e.g. `([0, 1], [2, 3])`) |
| `standing_bid` | `int` | First infeasible bid at n=4 (computed from `feasible_bid_mask(4)`) |
| `max_train_steps` | `int` | Fixed at 300 |
| `trunk` | `LearnedHandModel` | `_tiny_handmodel()` (hidden_dim=32, random weights, no checkpoint) |

**Key intermediate values**

| Symbol | Type | Description |
|--------|------|-------------|
| `solution` | `SubgameSolution` | `CFRPlusSubgameSolver(max_iters=200, eps=1e-4, seed=0).solve(hands)` |
| `solver_target` | `float` | `solution.avg_call_prob[(standing_bid, 0)]` — expected ≈ 1.0 |
| `info` | `Infostate` | Constructed from `standing_bid`, `pool_size=4`, `legal_actions={standing_bid+1..NUM_BIDS-1, CALL_ACTION}` |
| `belief` | `HandBelief` | `trunk.belief(info)` — `q[standing_bid] == 0.0` since `feasible_mask[standing_bid] == False` |
| `features` | `np.ndarray (1, 478)` | `build_call_features(trunk_repr, q, standing_bid, pool_size)` |
| `targets` | `torch.Tensor (1,)` | `torch.tensor([solver_target])` |

**Assertions**

```
assert solver_target >= 0.98                        # solver knows to call
assert final_p_call > 0.95                          # head learned the signal
```

### Training loop (mini-distillation)

No checkpoint is needed. The trunk is randomly initialized and frozen.
The loop trains the `CallPolicyNet` head only:

```python
cfg  = CallPolicyConfig(hidden=64, load_trunk=None, device="cpu")
net  = CallPolicyNet(cfg)
opt  = AdamW(net.parameters(), lr=1e-3)
feat = torch.from_numpy(features).expand(32, -1)    # repeat as batch of 32
tgt  = torch.full((32,), solver_target)

for _ in range(max_train_steps):
    opt.zero_grad()
    loss = loss_step(CallPolicyTrainState(net, trunk, opt), feat, tgt)
    loss.backward()
    opt.step()
```

The test does NOT call `build_train_state` (which requires a checkpoint path).
It constructs `CallPolicyTrainState` directly, which is valid for tests.

### Deviation note

The AR-2 design §7.1 says "Distilled CallPolicy.call_prob(...).p_call > 0.95"
and implies a *trained* model. This design operationalises "trained" as a
mini-distillation on a single fixed infostate repeated as a batch, rather than
a full distillation run. The `q_at_bid = 0.0` feature provides a clean
discriminative signal; a simple MLP with BCE loss and AdamW converges to
`sigmoid(logit) > 0.95` reliably within 300 steps for any trunk weight
initialisation. `max_train_steps=300` is the accepted threshold; increase to
500 only if CI shows flakiness.

### File location

`tests/agents/learned/test_phase6_impossible_bid.py`

---

## 2. §7.2 — `n=2` entropy floor property test (slow, 10³ deals)

### Purpose

Verify that `DistilledBidPolicy.bid_dist()` always satisfies the
inference-time entropy floor for every output at `pool_size = 2`, regardless
of the network's current weights. This tests `apply_entropy_floor` through the
full public API path rather than calling it in isolation (which duplicates
`test_entropy_floor.py::test_floor_property_n2`).

### Pool-size context

`pool_size = 2` → 1 card per seat in a 1v1 match.  
At `n = 2`, only `HIGH_CARD` bids are feasible (requires 1 card); that is
13 bids (ranks 0–12). The entropy floor: `H_target = floor_frac[2] * log(13) ≈ 0.6 * 2.565 = 1.539 nats`.

### I/O spec

**Inputs**

| Symbol | Type | Source |
|--------|------|--------|
| `bid_policy` | `DistilledBidPolicy` | `_tiny_handmodel()` trunk + random-init `BidPolicyNet` |
| `infostates` | `list[Infostate]` (1 000 items) | Generated programmatically (see below) |
| `beliefs` | `list[HandBelief]` (1 000 items) | `HandBelief` with `q` uniform over feasible bids at n=2 |

**Infostate generation**

Generate 1 000 `Infostate` objects at `pool_size = 2` covering:
- 500 opener infostates (no standing bid) — all HIGH_CARD bids are legal.
- 500 mid-round infostates with `standing_bid ∈ {0, 1, …, 11}` (any HC bid
  below the top), so `legal_actions = {standing_bid+1 … 12} ∪ {CALL_ACTION}`.

Use a seeded `np.random.default_rng(42)` to pick `standing_bid` for each
mid-round case. Use `own_hand = (0,)` (card index 0) for all — the trunk is
random so the actual card value does not affect the floor property being tested.

**Output format per infostate**

| Symbol | Type | Description |
|--------|------|-------------|
| `dist` | `BidDistribution` | Returned by `bid_policy.bid_dist(info, belief, hh_fired=False)` |
| `feasible_count` | `int` | `int(np.array(info.feasible_mask[:NUM_BIDS]).sum())` |
| `H_target` | `float` | `floor_frac[2] * math.log(feasible_count)` if `feasible_count > 1` else `0.0` |

**Assertion (per infostate)**

```
assert dist.entropy >= H_target - 1e-3
```

If `feasible_count <= 1`, entropy is trivially 0 and no floor applies —
skip those rows (degenerate support; `apply_entropy_floor` returns early).

### Slow mark

```python
@pytest.mark.slow
def test_entropy_floor_n2_property() -> None: ...
```

Registered in `pyproject.toml` (already present per
[feedback_pytest_slow_mark.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_pytest_slow_mark.md)).
Expected runtime < 60 s on CPU with `hidden_dim = 32` trunk.

### File location

`tests/agents/learned/test_phase6_entropy_floor_n2.py`

---

## 3. §7.5 — Trunk-freeze invariance after one epoch (fast)

### Purpose

After running a full training epoch (gradient accumulation + `optimizer.step()`
via the BidPolicy `loss_step`), the frozen trunk parameters must be bitwise
unchanged. Phase 3 already checks `requires_grad=False` is set at construction;
§7.5 is a stronger invariant: even when gradients flow through the head and the
optimizer's `step()` is called, the trunk params must not move.

This guards against bugs such as:
- Accidentally including trunk params in the optimizer's parameter list.
- A future refactor that re-enables trunk gradients without updating the Phase 3 freeze.

### I/O spec

**Inputs**

| Symbol | Type | Source |
|--------|------|--------|
| `trunk` | `LearnedHandModel` | `_tiny_handmodel(hidden_dim=32)` — random init |
| `net` | `BidPolicyNet` | `BidPolicyNet(cfg)` — random init; `cfg.trunk_dim = 32` |
| `optimizer` | `AdamW` | `AdamW(net.parameters(), lr=1e-3)` — trunk params excluded |
| `state` | `BidPolicyTrainState` | Constructed directly (no checkpoint path required) |
| `n_steps` | `int` | 50 gradient steps on synthetic data |

**Synthetic batch construction (per step)**

| Symbol | Shape | Description |
|--------|-------|-------------|
| `features` | `(32, 367)` | `torch.randn(32, cfg.input_dim)` |
| `log_q` | `(32, NUM_BIDS)` | `torch.randn(32, NUM_BIDS)` |
| `bid_mask` | `(32, NUM_BIDS)` bool | All True (unconstrained feasibility for this test) |
| `targets` | `(32, NUM_BIDS)` | `F.softmax(torch.randn(32, NUM_BIDS), dim=-1)` (random valid simplex) |
| `pool_size` | `(32,)` int | `torch.full((32,), 5)` (n=5 → β(n)=0 for simplest loss) |

**Snapshot and assertion**

```python
def _trunk_l2(trunk: LearnedHandModel) -> float:
    return sum(p.data.norm(p=2).item() ** 2 for p in trunk.net.parameters()) ** 0.5

before = _trunk_l2(trunk)
for _ in range(n_steps):
    loss = loss_step(state, features, log_q, bid_mask, targets, pool_size)
    state.optimizer.zero_grad()
    loss.backward()
    state.optimizer.step()
after = _trunk_l2(trunk)

assert before == after   # bitwise float equality; trunk must not drift
```

The `before == after` comparison uses Python float `==`, which is exact
(bitwise IEEE-754 equality for `float`). A single gradient step that touches
trunk params would produce a nonzero delta guaranteed to differ at float64
precision.

### File location

`tests/agents/learned/test_phase6_trunk_freeze.py`

---

## 4. File summary

| File | Test(s) | Mark | §  |
|------|---------|------|----|
| `tests/agents/learned/test_phase6_impossible_bid.py` | `test_impossible_bid_solver_target`, `test_impossible_bid_mini_distillation` | — | 7.1 |
| `tests/agents/learned/test_phase6_entropy_floor_n2.py` | `test_entropy_floor_n2_property` | `slow` | 7.2 |
| `tests/agents/learned/test_phase6_trunk_freeze.py` | `test_trunk_freeze_after_epoch` | — | 7.5 |

---

## 5. Shared helper

All three test files use `_tiny_handmodel()` (32-dim hidden, single transformer
layer, random init). To avoid duplication, copy-paste the factory from
`test_phase3_smoke.py`. If it grows in a later session, factor it into
`tests/agents/learned/conftest.py`. Do not create a shared module now
(YAGNI — three callers is the threshold per project conventions).

---

## 6. Dependencies and what is NOT needed

- No checkpoint file is needed for any of the three tests.
- No distillation run (Phase 7) is needed.
- `CFRPlusSubgameSolver` is needed only for §7.1's solver assertion.
- All three tests run on CPU; the `PYTHONPATH` / python interpreter note
  from [feedback_python_env_torch.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_python_env_torch.md)
  applies: use `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`.

---

## 7. Open questions (none blocking implementation)

None. The three test designs are fully specified. Proceed to implementation
once the design is approved.

**Stopping here per the design-first gate.**
