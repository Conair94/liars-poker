# Small-game variants

This document specifies the reduced Liar's Poker variants used as solver
oracles. Their canonical encoding is fixed by [ADR-005](adr/005-openspiel-game-id-and-encoding.md).

## `python_liars_poker_kuhn`

A Kuhn-poker-sized reduction that retains the essential exact-rules dynamic:
private information, a totally-ordered bid space, and a bluff/call decision.

### Rules

- 2 players.
- Deck = 3 cards, project rank encoding (Q=10, K=11, A=12).
- Each player is dealt 1 private card.
- Pool size = 2 → best 5-card hand = High Card of `max(p0, p1)` (no other hand
  type can form).
- Bidding: P0 bids first; P1 may raise or call; play continues until a CALL.
- Bid space: HC Q, HC K, HC A.
- Resolution (**exact rules**): the bid holds iff `pool_best_rank == bid_rank`.
  Bidder wins if it holds, caller wins if it doesn't. ±1 zero-sum reward.

### Information state

Each player observes their own private card and the public bid history. The
adapter exposes both an info-state string and a tensor (player one-hot +
private-card one-hot + per-turn one-hot history).

### Reference policy behaviors

After CFR convergence to <1e-3 exploitability:

- **Holding A.** Always bids HC A — uniquely truthful and call-resistant.
- **Holding K.** Mostly bids HC K (truth) with a small probability of bluffing
  HC A; calls a HC A bid more often than the prior alone would warrant.
- **Holding Q.** Mostly bids HC Q (truth); bluffs HC K or HC A with
  small mixing probability; calls almost any HC A bid.

These match Kuhn poker's qualitative equilibrium structure (low cards bluff
sometimes, high cards always truth-bid, mid cards play mixed defense).

### Use

- Validate the project's CFR / CFR+ implementations: solve the same game with
  both, compare best-response gaps and policy distance.
- Validate any future neural agent: project the agent's policy onto this
  variant via the helpers in `interop.openspiel_adapter` and measure
  exploitability.

### Reference artifact

`data/oracles/liars_poker_kuhn_policy.npz` contains the converged
average-policy table from `training.metrics.build_kuhn_oracle`. Fields:

| Field | Type | Description |
| --- | --- | --- |
| `infoset_keys` | `np.ndarray[str]` | OpenSpiel information-state strings, in row order |
| `action_probs` | `np.ndarray[float32]` | per-infoset action distribution `(n_infosets, num_actions)` |
| `exploitability` | `float32` | converged exploitability |
| `iterations` | `int64` | CFR iterations used to produce the policy |

## Future variants

The following are placeholders; implement only when an agent migration needs
them.

- **`python_liars_poker_pair2`** — 2 players × 2 cards each from a 12-card
  deck (3 ranks × 4 suits). Adds Pair to the pool's hand-type space and tests
  rank-leakage strategies. Currently *not implemented*.
- **`python_liars_poker_kuhn_3p`** — 3-player Kuhn variant for testing PSRO /
  multiplayer exploitability proxies. Currently *not implemented*.
