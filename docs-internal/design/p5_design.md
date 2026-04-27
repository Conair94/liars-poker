# P5 Design Doc

- **Status:** Draft
- **Date:** 2026-04-26
- **Owner:** main
- **Predecessor:** P4 (commit `2076a5b`)

## Goals

P5 closes the three follow-ups left open at the end of P4:

1. **HH in the OpenSpiel adapter** — wire the High Hand declaration action
   into `python_liars_poker_exact` so that any agent with nonzero HH policy
   mass can be projected onto the adapter without losing actions. This is a
   hard prerequisite for #2 and for any future training run that goes through
   the adapter, because the project-wide decision (2026-04-26) is that all
   future games use HH as a standard rule.
2. **Modular agent / policy interface + per-agent exploitability** — define
   a stable contract that registry agents can implement so they can be
   projected onto the Kuhn small-game variant, then populate the
   `exploitability_a` / `exploitability_b` fields in `benchmark.py
   --exploitability` (currently `null`).
3. **Pick up the three reflect rules deferred from P3** (see
   `project_p3_decision_logging.md`).

This doc treats #1 as a P4-completion task and #2/#3 as separate sessions.
Only #1 is in scope for the *current* session; #2 and #3 get scoped here so we
can sequence them, but their implementation is gated on a follow-up design
review.

## Non-goals

- Multi-round adapter / full-match exploitability. Still P6.
- Neural-agent observation tensors on the 52-card adapter. Not needed for
  HH wiring or for Kuhn projection.
- Five-Kings rules in the adapter. Out of scope.

---

## #1 — HH in the OpenSpiel adapter (this session)

### Current state

`src/interop/openspiel_adapter.py` declares:

```python
_FULL_NUM_ACTIONS = NUM_BIDS + 1   # 111  (bids 0..109 + CALL=110, HH disabled)
_FULL_CALL = NUM_BIDS              # 110
```

The project engine uses `NUM_ACTIONS = NUM_BIDS + 2 = 112`, with
`HH_ACTION = 111`. The adapter is therefore one action short of the engine
wire format.

### Design

- Bump the adapter to match the engine action layout exactly:
  - `_FULL_NUM_ACTIONS = NUM_BIDS + 2  # 112`
  - `_FULL_CALL = NUM_BIDS             # 110`  (unchanged)
  - `_FULL_HH   = NUM_BIDS + 1         # 111`  (new, matches engine `HH_ACTION`)
- `_legal_actions` adds `_FULL_HH` whenever a bid stands (i.e., same gating
  as `_FULL_CALL`):
  ```python
  return list(range(cur_idx + 1, NUM_BIDS)) + [_FULL_CALL, _FULL_HH]
  ```
- `_apply_action` adds an `action == _FULL_HH` branch that invokes a new
  `_resolve_high_hand` method.
- `_resolve_high_hand` mirrors the **single-round** projection of
  `MatchState._resolve_high_hand` from `src/game/engine.py:361`:
  - Pool = all dealt cards (single-round adapter has no eliminations).
  - Compute `(pool_type, pool_primary)` via
    `_evaluate_ranked` + `normalize_hand_type` (already imported).
  - `correct = (pool_type == bid.hand_type and pool_primary == bid.primary_rank)`
  - On `correct`: declarer (the player who chose HH) wins, bidder loses.
  - On incorrect: bidder wins, declarer loses.
  - Reward is ±1, zero-sum (no card-count penalty in single-round).
- `_action_to_string`: handle the HH action with the literal `"HH"`.
- `max_game_length` stays at `NUM_BIDS + 1` — HH ends the game like CALL,
  so adding it as an alternative terminator does not lengthen the worst case.
- `_FULL_GAME_INFO.num_distinct_actions` updates to `_FULL_NUM_ACTIONS`.

### Why HH semantics differ from CALL in the adapter

- **CALL** says *"no 5-card subset of the pool matches the standing bid."*
  Caller wins iff `_has_exact_hand(pool, bid)` is False.
- **HH** says *"the pool's single best 5-card hand is **exactly** the
  standing bid — no stronger hand exists."* Declarer wins iff
  `pool_best == bid`. These disagree when the pool contains the bid hand
  *and* something stronger.

That asymmetry is why we cannot share `_resolve_call`'s logic for HH.

### Tests

- Extend `tests/interop/test_openspiel_roundtrip.py` so the round-trip suite
  runs both `high_hand=False` (existing) **and** `high_hand=True` paths
  against the project engine. Parity assertions:
  - `legal_actions(state)` matches engine `legal_actions(seat)` over the
    1000-game replay.
  - Terminal returns match for both CALL- and HH-terminated games.
- Add a focused unit test that constructs a hand-crafted state (pool with a
  known best, a standing HC bid that does/does-not match the pool best) and
  verifies the HH resolver returns the expected `±1`.
- The Kuhn variant is unchanged — HH would be vacuous on a 1-card pool.

### Risk and rollback

- Bumping `num_distinct_actions` from 111 to 112 is a wire-format change;
  any consumer hard-coding 111 will break. Today the only consumers are the
  P4 round-trip tests and the exploitability oracle, both of which we
  control. Kuhn variant is unaffected (it has its own action space = 4).
- Rollback = revert the adapter file + ADR amendment; no on-disk artefacts
  depend on the new layout.

### ADR

Amend ADR-005 in place (it is `Accepted` with no superseding ADR yet, and
the HH disabling is explicitly called out as a TODO(P5) — that TODO is now
done). Update the `HH action` row of the `python_liars_poker_exact` table,
update `num_distinct_actions` and the action-layout row, and append a short
"2026-04-26 amendment: HH wired" note under "Decision".

### Checklist (#1)

- [ ] Add `_FULL_HH`, bump `_FULL_NUM_ACTIONS` and `_FULL_GAME_INFO`.
- [ ] Update `_legal_actions` to include `_FULL_HH` when a bid stands.
- [ ] Implement `_resolve_high_hand` in `LiarsPokerExactState`.
- [ ] Wire `_FULL_HH` into `_apply_action` and `_action_to_string`.
- [ ] Add HH unit test + extend round-trip test to `high_hand=True`.
- [ ] Run full `tests/interop/` suite; verify pass.
- [ ] Amend `docs-internal/design/adr/005-openspiel-game-id-and-encoding.md`.
- [ ] Update memory: mark `project_hh_must_be_enabled.md` as resolved;
      update `project_p4_openspiel.md` to note HH now wired.
- [ ] Update CHANGELOG with a P5-1 entry.

---

## #2 — Modular agent interface + per-agent exploitability (separate session)

Out of scope for the current session. Sketch only:

- Define a `policy.action_probs(info_state) -> dict[int, float]` contract
  that every registry agent implements.
- Add a Kuhn projection: a function that takes any project agent and
  produces an `open_spiel.python.policy.Policy` over the
  `python_liars_poker_kuhn` infosets, filling unreachable infosets with the
  uniform-legal default.
- Wire `exploitability_a` / `exploitability_b` in `benchmark.py
  --exploitability` to call OpenSpiel's `nash_conv` against the projected
  policy.
- **Open question (must resolve before implementation):** how to project a
  52-card-trained agent onto a 3-card Kuhn variant *meaningfully*. Naive
  uniform-fill on missing infosets gives a number but it isn't
  exploitability of *the agent* — it's exploitability of the projection.
  Need to decide whether the metric reported is (a) projection
  exploitability (cheap, possibly misleading) or (b) some kind of
  abstraction-aware lower bound.

A separate design doc will resolve (a) vs (b) before any code lands.

---

## #3 — Reflect rules deferred from P3 (separate session)

Out of scope for the current session. Three rules listed in
`project_p3_decision_logging.md` need P5-class data (per-agent
exploitability, modular policy access). Naturally sequences after #2.

## Open questions

- Does any consumer outside this repo import `_FULL_NUM_ACTIONS`? (grep
  before the change to be safe.)
- Should the HH resolver clear `_current_bid` on resolve, the way
  `_resolve_call` does implicitly via `_game_over`? (Cosmetic — game is
  terminal either way; default to "no, leave the standing bid visible for
  trace inspection.")
