# Training Pipeline — Test Scaffold

Tracks the test cases that should accompany each module of the refactored
training pipeline. This is a living checklist; contributors add cases as
modules land.

## P3 — Decision Logging + Reflect v1

### `src/training/logging.py`

- [ ] `DecisionRecord.to_json_line()` round-trips through `json.loads` /
      `pandas.read_json(..., lines=True)` with no field loss.
- [ ] Optional fields (`outcome`, `extras`) are dropped from the JSONL line
      when empty so logs stay compact.
- [ ] `action_to_str()` covers all three engine action classes:
      `0..NUM_BIDS-1` (bids), `CALL_ACTION`, `HH_ACTION`.
- [ ] `card_to_str()` matches the card encoding doc (rank*4+suit, suit order
      C/D/H/S) and is the inverse of the encoder for all 52 cards.
- [ ] `DecisionLogger` truncates the file on open (a run owns its log).

### `src/training/decision_capture.py`

- [ ] `LoggingAgentWrapper` forwards the chosen action unchanged
      (`inner.choose_action(state) == wrapper.choose_action(state)` for a
      deterministic agent).
- [ ] Wrapper proxies arbitrary attributes via `__getattr__` (e.g. attached
      `state`, `seat` on heuristic agents).
- [ ] Wrapper increments `turn_counter_ref[0]` exactly once per call.
- [ ] `_snapshot_state` produces a stable shape for both pre-bid (no standing
      bid) and mid-round (standing bid present) states.

### `src/training/benchmark.py`

- [ ] `--log-decisions` writes `decisions.jsonl` to
      `data/runs/<run_id>/decisions.jsonl` and `metrics.json` alongside.
- [ ] Without `--log-decisions`, behavior is unchanged from the P1 path
      (writes `data/runs/benchmark/benchmark_results.json`).
- [ ] `_make_run_id` is deterministic for the same config and changes when
      any config field changes.
- [ ] Game IDs do not collide across pairs in a multi-pair run.

### `src/training/reflect.py`

- [ ] **Schema validation:** `_iter_records` rejects malformed JSON lines
      with a clear error and skips blank lines silently.
- [ ] **Infeasible-bid tripwire:** synthetic record with `feasible=False` on
      `chosen` is flagged exactly once.
- [ ] **Infeasible-bid tripwire (clean run):** all records with
      `feasible=True` produce zero flags — this is the acceptance criterion
      for the 2026-04-24 feasibility-fix sanity check.
- [ ] **Stale-bid repetition:** synthetic bucket with 12 opening turns,
      11 of them identical, is flagged with count = 10.
- [ ] **Stale-bid repetition:** bucket with 9 openings is *not* flagged
      (below the ≥10 threshold).
- [ ] **Bucket key:** `(agent, opponent, ruleset)` collisions never merge
      across rulesets.
- [ ] **CLI smoke:** `python -m training.reflect <run_id>` on a real run
      produces `summary.md` in <30s for ≥500 games (design N3).
- [ ] **Deferred-rules surface:** the deferred-rules table is always
      rendered, with a `deferred: needs p/eu` annotation.

### W&B integration

- [ ] `--wandb` populates `entity`, `project`, run name == run_id.
- [ ] `git_sha` and `config_hash` appear in run config.
- [ ] `flaws` dict (infeasible_bid, stale_bid_repetition, total_decisions,
      total_buckets) is logged and reflected in `run.summary`.

## P4 — OpenSpiel Adapter + Exploitability + Small-Game Oracle

Specifies the test scaffold landed in [tests/interop/](../tests/interop/) and
[tests/oracles/](../tests/oracles/).

### Adapter round-trip ([tests/interop/test_openspiel_roundtrip.py](../tests/interop/test_openspiel_roundtrip.py))

- [x] `pyspiel.load_game("python_liars_poker_kuhn")` loads, `num_distinct_actions == 4`.
- [x] `pyspiel.load_game("python_liars_poker_exact", {hand_size: 5})` loads,
      action space == `NUM_BIDS + 1 = 111`.
- [x] **1000 random rollouts** through the project engine
      (`game.engine.MatchState`, single round, hand_size 5, 2 players, exact
      rules) and the OpenSpiel adapter must agree at every step on:
  - the current player,
  - the set of legal action indices,
  - the terminal returns.
- [x] Kuhn-variant truth bid (HC A held by a player whose card *is* A,
      followed by an opponent call) yields ±1 in the bidder's favor.
- [x] Kuhn-variant lie bid (HC A bid by Q-holder, opponent calls) yields ±1
      in the caller's favor.

### Solver oracle ([tests/oracles/test_kuhn_convergence.py](../tests/oracles/test_kuhn_convergence.py))

- [x] CFR on `python_liars_poker_kuhn` converges to **< 1e-3 exploitability**
      within 1000 iterations (exit criterion).
- [x] More iterations ⇒ strictly lower exploitability on Kuhn (regression
      guard on the solver itself).

### Benchmark integration

- [x] `benchmark.py --exploitability` populates
      `output["oracle_exploitability"]` in `metrics.json`.
- [x] Per-pair `exploitability_a` / `exploitability_b` slots are present
      (set to `null` until P5 supplies the projection layer).

### Future Phases

Test scaffolds for P5–P6 will be appended here as those phases land.
