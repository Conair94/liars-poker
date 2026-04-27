# ADR-005: OpenSpiel game ID and state encoding

- **Status:** Accepted
- **Date:** 2026-04-26
- **Supersedes:** —
- **Implements:** P4 of [TRAINING_PIPELINE_PLAN.md](../../../Liars%20poker/TRAINING_PIPELINE_PLAN.md)

## Context

ADR-002 committed us to exposing Liar's Poker through OpenSpiel as an interop
layer rather than rewriting the engine on top of OpenSpiel. That ADR did not
pin down the canonical OpenSpiel game ID, action layout, chance-node design, or
information-state encoding. Without those, future agents (PSRO, NFSP,
continual-resolving variants) would each re-derive an encoding and the
small-game oracle policy would be non-portable.

This ADR records the canonical encoding so future work does not have to
re-litigate it.

## Decision

We register two games. Their wire formats are stable and are the contract for
all P4+ tooling (exploitability metric, oracle policy, round-trip tests).

### `python_liars_poker_kuhn` — small-game oracle

| Field | Value |
| --- | --- |
| Players | 2 |
| Deck | 3 cards, ranks `(Q=10, K=11, A=12)` in the project's 0..12 rank encoding |
| Hand size | 1 card per player |
| Pool best | High Card of `max(p0_card, p1_card)` |
| Bid space | High Card bids only — `{HC Q, HC K, HC A}` |
| Action layout | `0=HC Q, 1=HC K, 2=HC A, 3=CALL` (4 distinct actions) |
| Resolution | **Exact rules**: bid holds iff `pool_best_rank == bid_rank` |
| Reward | ±1 (zero-sum) on terminal call |
| `max_game_length` | 4 (HC Q, HC K, HC A, CALL) |

Chance: deal P0's card, then P1's, both via `EXPLICIT_STOCHASTIC` chance nodes
(uniform over remaining ranks).

Rationale: this is the smallest reduction that preserves the project's
exact-rules semantics — it has private hidden information, a totally-ordered
bid space, and a non-trivial bluff/call decision. With 24 information states it
is solvable exactly to <1e-3 exploitability in ~1k CFR iterations on CPU.

### `python_liars_poker_exact` — single-round 52-card adapter

| Field | Value |
| --- | --- |
| Players | configurable via `num_players` param (default 2; range [2, 5]) |
| Deck | 52 cards, project `card_index = rank * 4 + suit` |
| Hand size | configurable via `hand_size` param (default 5; range [1, 5]) |
| Bid space | full project bid space (`NUM_BIDS = 110`) |
| Action layout | `0..109 = bids in ascending order (game.bids.index_to_bid)`, `110 = CALL` |
| HH action | **disabled** in the adapter (the project engine's HH=111 is excluded). **TODO(P5):** all future games use HH as a standard rule (project decision 2026-04-26) — extend `_FULL_NUM_ACTIONS` to `NUM_BIDS + 2`, mirror `MatchState._resolve_high_hand` in a single-round resolver, and amend this ADR. |
| Resolution | **Exact rules**: some 5-card subset of the pool evaluates exactly to the standing bid |
| Reward | ±1 (zero-sum) on terminal call |
| `max_game_length` | `NUM_BIDS + 1 = 111` (worst case: every bid in order, then CALL) |

Chance: deal `num_players * hand_size` cards in player-major order (P0's hand
first, then P1's, ...) via uniform chance nodes over remaining cards.

Rationale: tabular exploitability is **not** tractable on this game (chance
fanout alone is C(52, num_players * hand_size)). The adapter exists for:

1. Round-trip parity tests against the in-house engine (legal actions and
   terminal returns).
2. Sampled / projection-based exploitability proxies (P5+).
3. Plug-in compatibility with OpenSpiel's PSRO / continual-resolving libraries
   that operate on `pyspiel.Game` directly.

### Information state

For both games we provide an information-state string. The Kuhn variant also
provides a tensor (one-hot player + one-hot private card + per-turn one-hot
action history). The 52-card adapter provides only the string — neural agents
must encode hands from the engine's native `info_state(seat)` dict, not from
the OpenSpiel observer.

This split keeps the OpenSpiel adapter minimal: tensor encoding for the small
game where exact-solve algorithms need it; string-only for the full game where
neural agents already use the project's encoder.

## Consequences

**Positive:**

- Stable game IDs unblock P5 (modular agent contract) and P6 (publish-grade
  exploitability tracking) without further design questions.
- Single round of 52-card play is tractable to expose, side-stepping the
  multi-round match progression that would explode action history.
- Disabling HH in the adapter keeps the action space small and means the same
  adapter works whether or not the project engine has HH enabled.

**Negative:**

- The 52-card adapter is single-round; the engine's full match progression
  (count-up / count-down with hand-size growth) is **not** exposed via
  OpenSpiel. Multi-round PSRO would need a separate adapter or, more
  realistically, a project-side league trainer.
- Per-agent exploitability on the 52-card variant is intentionally not
  computed — `metrics.json` carries `exploitability_a` / `exploitability_b`
  slots set to `null` until P5 supplies the small-game projection layer.

## Alternatives considered

- **Multi-round adapter.** Rejected for P4: the action-history encoding
  required for OpenSpiel's perfect-recall observers grows unboundedly across
  rounds, and the chance fanout per deal is already prohibitive.
- **Bid-rank-only Kuhn variant.** Considered tying ranks to suit-bearing cards
  to test flushes/straights at minimal scale. Rejected: a 1-card pool can
  never form a flush or straight, so suits would be dead information; making
  the pool larger reintroduces the chance-fanout blow-up.

## References

- [src/interop/openspiel_adapter.py](../../../src/interop/openspiel_adapter.py)
- [src/training/metrics/exploitability.py](../../../src/training/metrics/exploitability.py)
- [tests/interop/test_openspiel_roundtrip.py](../../../tests/interop/test_openspiel_roundtrip.py)
- [tests/oracles/test_kuhn_convergence.py](../../../tests/oracles/test_kuhn_convergence.py)
- [data/oracles/liars_poker_kuhn_policy.npz](../../../data/oracles/liars_poker_kuhn_policy.npz)
