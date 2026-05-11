# AR-2 Phase 6 — Test Suite Checklist

- **Status:** Pending (design complete 2026-05-11)
- **Date:** 2026-05-11
- **Parent design:** [agent_redesign_ar2_phase6_design.md](agent_redesign_ar2_phase6_design.md)
- **Parent checklist:** [agent_redesign_ar2_checklist.md](agent_redesign_ar2_checklist.md) §Phase 6

---

## Step 1 — §7.1: Impossible-bid solver target

**File:** `tests/agents/learned/test_phase6_impossible_bid.py`

- [ ] Add `_find_first_infeasible_bid(pool_size: int) -> int` helper: iterate
  `feasible_bid_mask(pool_size)`, return first `False` index.
- [ ] Construct a 2-card-per-seat deal `([0, 1], [2, 3])` and `standing_bid =
  _find_first_infeasible_bid(4)`.
- [ ] Build `Infostate` for that state: `pool_size=4`, `standing_bid`,
  `legal_actions = {i : i > standing_bid, i < NUM_BIDS} ∪ {CALL_ACTION}`,
  `feasible_mask` from `feasible_action_mask(4)`.
- [ ] `test_impossible_bid_solver_target`: run
  `CFRPlusSubgameSolver(max_iters=500, eps=1e-4, seed=0).solve(hands)`.
  Assert `solution.avg_call_prob[(standing_bid, 0)] >= 0.98`.

## Step 2 — §7.1: Mini-distillation → `p_call > 0.95`

- [ ] `test_impossible_bid_mini_distillation`:
  - Construct `_tiny_handmodel(hidden_dim=32)` (copy helper from `test_phase3_smoke.py`).
  - Build `HandBelief`: `q` uniform over feasible bids at n=4 via `feasible_bid_mask(4)`;
    verify `q[standing_bid] == 0.0` (infeasible bid has zero mass).
  - Compute `trunk_repr` (shape (1, 32)) via `_trunk_forward(trunk, [info], device="cpu")`.
  - Call `build_call_features(trunk_repr, q[None], np.array([standing_bid]), np.array([4]))`.
  - Construct `CallPolicyNet(cfg)` (hidden=64; `cfg.trunk_dim=32`).
  - Build `CallPolicyTrainState(net, trunk, optimizer)` directly (no checkpoint).
  - Train 300 steps: repeat the single feature row as batch=32 against `target = solver_target`.
  - Assert `DistilledCallPolicy(net, trunk).call_prob(info, belief).p_call > 0.95`.
- [ ] Both tests pass; total runtime < 30 s.

## Step 3 — §7.2: Entropy floor property test at n=2

**File:** `tests/agents/learned/test_phase6_entropy_floor_n2.py`

- [ ] Add `@pytest.mark.slow` decorator.
- [ ] Construct `DistilledBidPolicy` from `_tiny_handmodel(hidden_dim=32)` +
  `BidPolicyNet(cfg)` (cfg.trunk_dim=32, cfg.hidden=64).
- [ ] Generate 1 000 `Infostate` objects at `pool_size = 2`:
  - 500 opener infostates (`standing_bid = None`): all HC bids (0–12) legal.
  - 500 mid-round infostates: `standing_bid = rng.integers(0, 12)` (HC bids only);
    `legal_actions = {i : standing_bid < i < NUM_BIDS} ∪ {CALL_ACTION}`.
  - All with `own_hand=(0,)`, `hand_sizes=(1, 1)`, `exact_rules=True`, `high_hand=True`.
- [ ] Generate matching `HandBelief` objects: `q` uniform over
  `feasible_bid_mask(2)` (13 HC bids), same for all 1 000.
- [ ] For each (info, belief): call `bid_dist = bid_policy.bid_dist(info, belief, hh_fired=False)`.
  Assert `bid_dist.entropy >= floor_frac[2] * math.log(feasible_count) - 1e-3`
  (skip if `feasible_count <= 1`).
- [ ] Test passes; all 1 000 infostates satisfy the floor.

## Step 4 — §7.5: Trunk-freeze invariance after one epoch

**File:** `tests/agents/learned/test_phase6_trunk_freeze.py`

- [ ] Construct `_tiny_handmodel(hidden_dim=32)`.
- [ ] Construct `BidPolicyNet(cfg)` (cfg.trunk_dim=32, cfg.hidden=64) and
  `AdamW(net.parameters(), lr=1e-3)`.
- [ ] Build `BidPolicyTrainState(net=net, trunk=trunk, optimizer=opt, step=0)` directly.
- [ ] Snapshot `before = _trunk_l2(trunk)`.
- [ ] Run 50 training steps with `loss_step(state, feat, log_q, bid_mask, targets, pool_size)`:
  - `feat = torch.randn(32, cfg.input_dim)`
  - `log_q = torch.randn(32, NUM_BIDS)`
  - `bid_mask = torch.ones(32, NUM_BIDS, dtype=torch.bool)`
  - `targets = F.softmax(torch.randn(32, NUM_BIDS), dim=-1)`
  - `pool_size = torch.full((32,), 5, dtype=torch.long)` (β=0 at n≥5)
  - Call `optimizer.zero_grad(); loss.backward(); optimizer.step()` each step.
- [ ] Snapshot `after = _trunk_l2(trunk)`.
- [ ] Assert `before == after` (Python float exact equality).
- [ ] Test passes; runtime < 10 s.

## Step 5 — Full suite run

- [ ] Run all three new test files:
  ```
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest \
    tests/agents/learned/test_phase6_impossible_bid.py \
    tests/agents/learned/test_phase6_trunk_freeze.py \
    -v
  ```
- [ ] Run slow test separately (background + Monitor per
  [feedback_cfr_test_suite_runtime.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_cfr_test_suite_runtime.md)):
  ```
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest \
    tests/agents/learned/test_phase6_entropy_floor_n2.py \
    -v -m slow
  ```
- [ ] Run existing regression suite (fast subset):
  ```
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest \
    tests/agents/ tests/training/ -v --ignore=tests/agents/heuristic/test_cfr_1v1.py
  ```
- [ ] All tests green (existing baseline: 1 pre-existing R-NaD import failure is
  acceptable; any new failure is a regression).

## Step 6 — Checklist update and commit

- [ ] Mark Phase 6 complete in [agent_redesign_ar2_checklist.md](agent_redesign_ar2_checklist.md).
- [ ] Commit: "AR-2 Phase 6: §7.1/7.2/7.5 test suite (impossible-bid, entropy floor n=2, trunk freeze)".

---

## Deviations to record if they occur

- If §7.1 mini-distillation requires > 300 steps, increase to 500 and record here.
- If §7.2 slow test takes > 2 min with hidden_dim=32, shrink to hidden_dim=16 and record.
- If trunk L2 comparison requires `pytest.approx` (float noise from `norm()` itself),
  switch to bitwise param-by-param equality and record.
