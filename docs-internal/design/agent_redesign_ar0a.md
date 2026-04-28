# AR-0a Sub-design — Shared Types, Protocols, Checkpoints

- **Status:** Draft (design-first gate; implementation follows in this same session per user direction)
- **Date:** 2026-04-28
- **Owner:** main
- **Parent plan:** [agent_redesign_plan.md](agent_redesign_plan.md)
- **Parent design:** [agent_redesign.md](agent_redesign.md) §8

This sub-design fixes the on-the-wire details that the parent design left
abstract: dtypes, validation timing, JSON canonical form, checkpoint key
order, and the back-compat strategy for existing logging code. Once landed,
AR-1 imports `agents.contracts` and never re-derives any of these shapes.

## Goals

- Land a single Python module (`src/agents/contracts.py`) that defines every
  dataclass and Protocol from parent §8.1–8.5.
- Pin all numeric dtypes and validation invariants. No surprise float64.
- Provide one canonical JSON round-trip for `Infostate` and the per-decision
  trace fields `LoggingAgentWrapper` will write.
- Provide a single `save_component` / `load_component` pair (parent §8.6) so
  every subsequent phase produces interoperable checkpoints.
- Extract `is_bid_feasible` from
  [src/agents/registry.py](../../src/agents/registry.py) into a module that
  has no dependency on the heuristic-agent stack
  ([src/game/feasibility.py](../../src/game/feasibility.py)) so
  contracts.py can import it without dragging the registry along.
- Be **strictly additive** to existing code. No behavior change for any
  current agent, benchmark, or test.

## Non-goals

- Sweep / benchmark harness. That is AR-0b.
- Any HandModel / CallPolicy / BidPolicy implementation. That is AR-1+.
- Refactoring `LiarsPokerNet` or `LoggingAgentWrapper` beyond the minimum
  needed to accept an optional `AgentDecision`.

---

## 1. Dtypes and validation timing

### Dtypes

| Field | dtype | Reason |
| --- | --- | --- |
| `q`, `q_logits`, `pi`, `pi_logits` | `np.float32` | Match torch default; halves memory vs float64; sufficient for log-prob arithmetic at our scale |
| `feasible_mask` (in beliefs/dists) | `np.bool_` | Set/clear semantics, one byte per slot |
| `Infostate.feasible_mask` | `tuple[bool, ...]` | Frozen dataclass member, immutable, hashable |
| Action indices (`int64` interface) | Python `int` in dataclasses, `np.int64` in arrays | We never need int8 economy here |
| `pool_size`, `hand_sizes` | Python `int` / `tuple[int, ...]` | Engine returns ints |

### Invariant validation

We chose **always-on `__post_init__` validation** for `HandBelief`,
`CallDecision`, and `BidDistribution`. Reasoning:

- These objects are constructed once per decision (not per inner-loop
  step), so the cost is negligible relative to a forward pass.
- Catching `q.sum() != 1` *at the source* is much cheaper than
  catching it 200 turns downstream in a benchmark report.
- If profiling shows it matters during large-scale rollouts (AR-1
  Phase A), we add an `unsafe_construct(...)` classmethod then.
  Profiling first, optimization second.

`Infostate` does **not** validate in `__post_init__` — it is constructed
once per turn from a known-good `MatchState`, and validating it on
construction would mean re-running engine-internal consistency checks.
Instead, `Infostate.from_match_state(state)` is the only public
constructor, and that method does the (cheap) checks the engine already
guarantees: `feasible_mask` length is `NUM_ACTIONS`, `legal_actions ⊆
range(NUM_ACTIONS)`, `bid_history` entries are well-formed.

Tolerances (used in invariant assertions):

- `abs(q.sum() - 1.0) < 1e-5`
- `abs(pi.sum() - 1.0) < 1e-5`
- `q[~feasible_mask].sum() < 1e-6`
- `pi[~feasible_mask | ~legal_mask].sum() < 1e-6`

We do not use stricter tolerances because float32 softmax over 110
elements has ~`110 * eps_f32` ≈ 1.3e-5 worst-case rounding error.

## 2. `Infostate` canonical form

The dataclass schema:

```python
@dataclass(frozen=True)
class Infostate:
    own_hand:        tuple[int, ...]            # sorted card ids
    pool_size:       int
    hand_sizes:      tuple[int, ...]
    own_seat:        int
    current_player:  int
    standing_bid:    int | None
    bid_history:     tuple[tuple[int, int], ...]
    legal_actions:   tuple[int, ...]
    feasible_mask:   tuple[bool, ...]           # len NUM_ACTIONS
    exact_rules:     bool
    high_hand:       bool
    five_kings:      bool
```

### Factory

`Infostate.from_match_state(state) -> Infostate` reads exactly the fields
populated by `MatchState.info_state(seat)` plus `state.legal_actions()`
plus the ruleset flags. It computes `feasible_mask` as

```
feasible_mask[a] = (a in legal_actions) and (
    a >= NUM_BIDS                        # CALL/HH always feasible-given-legal
    or is_bid_feasible(a, pool_size)     # bids: hand-type feasibility
)
```

### JSON form

```json
{
  "own_hand":       [3, 17, 42],
  "pool_size":      10,
  "hand_sizes":     [5, 5],
  "own_seat":       0,
  "current_player": 1,
  "standing_bid":   42,
  "bid_history":    [[0, 4], [1, 17]],
  "legal_actions":  [18, 19, 20, 110, 111],
  "feasible_mask":  [false, false, ..., true, true],
  "exact_rules":    true,
  "high_hand":      true,
  "five_kings":     false
}
```

Round-trip rule: `Infostate.from_json(state.to_json()) == state` for any
`Infostate` constructed via `from_match_state`. Tested with 1000-sample
property test in AR-0a tests.

The `feasible_mask` as an array of 112 bools is wasteful on disk (12 KB
per turn at 100k turns) but it is the trace's single load-bearing field
for downstream "was this bid feasible at the time?" analyses; we keep
it. Compression at the JSONL level (gzip) can recover this when we
care.

## 3. Beliefs / decisions / distributions

```python
@dataclass(frozen=True)
class HandBelief:
    q:             np.ndarray   # (NUM_BIDS,) float32
    q_logits:      np.ndarray   # (NUM_BIDS,) float32; -inf on infeasible
    feasible_mask: np.ndarray   # (NUM_BIDS,) bool
    n:             int          # pool size at the call site

    def __post_init__(self):
        # Shape, dtype, sum-to-one, mask invariant. See §1 tolerances.
        ...

@dataclass(frozen=True)
class CallDecision:
    p_call:  float
    inputs:  Mapping[str, float]   # diagnostics; schema open

@dataclass(frozen=True)
class BidDistribution:
    pi:            np.ndarray   # (NUM_ACTIONS,) float32
    pi_logits:     np.ndarray   # (NUM_ACTIONS,) float32
    legal_mask:    np.ndarray   # (NUM_ACTIONS,) bool
    feasible_mask: np.ndarray   # (NUM_ACTIONS,) bool
    entropy:       float
    support_size:  int

    def __post_init__(self): ...

@dataclass(frozen=True)
class AgentDecision:
    action_probs: dict[int, float]
    chosen:       int | None
    belief:       HandBelief | None
    call:         CallDecision | None
    bid:          BidDistribution | None
    hh_fired:     bool
```

`HandBelief.feasible_mask` is the **bid-only** mask of length `NUM_BIDS`.
`BidDistribution.feasible_mask` and `legal_mask` are full `NUM_ACTIONS`
length and apply jointly: the support is `legal_mask & feasible_mask`.
This shape difference is intentional: `q` is over pool best-hands (no
CALL/HH), `pi` is over actions (CALL/HH included).

### Trace JSON

To keep JSONL rows small, the trace serializes top-K entries instead of
full vectors:

```json
{
  "belief":   {"q_top5": [[42, 0.31], [55, 0.28], ...], "entropy": 2.7, "n": 10},
  "call":     {"p_call": 0.18, "inputs": {"q_at_bid": 0.12, "peak_q": 0.31}},
  "bid":      {"support_size": 4, "entropy": 1.4, "pi_top5": [[55, 0.4], ...]},
  "hh_fired": false
}
```

`top5` chosen empirically: the `Mixed`/`Adaptive` ladder has support ≤ 4
in practice, so 5 entries always cover the full support. AR-1 may bump
this to top10 if learned BidPolicy spreads more.

## 4. Protocols

```python
class HandModel(Protocol):
    def belief(self, info: Infostate) -> HandBelief: ...
    def belief_batch(self, infos: list[Infostate]) -> list[HandBelief]: ...

class CallPolicy(Protocol):
    def call_prob(self, info: Infostate, q: HandBelief) -> CallDecision: ...

class BidPolicy(Protocol):
    def bid_dist(
        self, info: Infostate, q: HandBelief, *, hh_fired: bool
    ) -> BidDistribution: ...
```

`belief_batch` is mandatory (not default). A vacuous default would
silently degrade Phase A pretrain throughput. AR-1's `LearnedHandModel`
provides a real batched path; baseline adapters provide a `[belief(i)
for i in infos]` loop that is honest about its serial nature.

## 5. Checkpoint schema

```python
def save_component(
    path: str,                       # *.pt
    *,
    component: Literal["handmodel", "callpolicy", "bidpolicy", "unified"],
    config: dict,                    # JSON-serializable
    state_dict: Mapping[str, "torch.Tensor"],
    iter: int,
    parent_run: str | None = None,
) -> None: ...

def load_component(path: str) -> dict: ...
```

Saved blob (a single `torch.save` payload):

```python
{
    "schema_version": 1,
    "component":      "handmodel",
    "config":         {...},
    "state_dict":     {...},
    "iter":           5000,
    "git_sha":        "89e3443",
    "parent_run":     None,
    "saved_at":       "2026-04-28T12:34:56Z",
}
```

- `git_sha` is captured automatically via `git rev-parse --short HEAD`,
  with `?` if not in a git tree (e.g. test environments).
- `schema_version=1` lets us version migrations later without breaking
  existing files.
- Optimizer state goes in a *separate* file `<path>.opt.pt` per parent
  §8.6; this module does not handle it.
- We do not capture `python_version` / `torch_version`; we will only
  start needing those across major version bumps and can add the field
  in v2.

## 6. Decision-capture extension

[src/training/decision_capture.py](../../src/training/decision_capture.py)
currently captures `action_probs` from any agent. We extend
`LoggingAgentWrapper` to:

- Look for an optional `agent.decision(state) -> AgentDecision` method.
- If present: use the AgentDecision to populate the new optional
  `belief` / `call` / `bid` / `hh_fired` fields on the
  `DecisionRecord`.
- If absent: behavior is identical to today (only `choices` and
  `chosen` are populated; the new fields default to `None`).

We add the extra fields to `DecisionRecord` (in
[src/training/logging.py](../../src/training/logging.py)) as `Optional`
with defaults, so every existing call site continues to compile.

The extension is mechanical and additive; no existing test changes.

## 7. policy.py — additive helper

[src/agents/policy.py](../../src/agents/policy.py) gets one new helper:

```python
def decision(agent, state: MatchState) -> AgentDecision:
    """Return the agent's full AgentDecision at `state`.

    Falls back to `action_probs(agent, state)` when the agent does not
    implement `.decision()`; in that case the trace fields are None.
    """
    fn = getattr(agent, "decision", None)
    if callable(fn):
        return fn(state)
    probs = action_probs(agent, state)
    return AgentDecision(action_probs=probs, chosen=None,
                         belief=None, call=None, bid=None, hh_fired=False)
```

The existing `action_probs(agent, state)` function is **untouched**.
This is a pure addition, opt-in for callers who want the richer return.

## 8. Feasibility helper extraction

`src/game/feasibility.py` (new file):

```python
from game.bids import (
    HIGH_CARD, PAIR, TWO_PAIR, THREE_OF_A_KIND,
    STRAIGHT, FLUSH, FULL_HOUSE, FOUR_OF_A_KIND, STRAIGHT_FLUSH,
    NUM_BIDS, NUM_ACTIONS, index_to_bid,
)

_MIN_CARDS_FOR_HAND = {
    HIGH_CARD: 1, PAIR: 2, TWO_PAIR: 4, THREE_OF_A_KIND: 3,
    STRAIGHT: 5, FLUSH: 5, FULL_HOUSE: 5,
    FOUR_OF_A_KIND: 4, STRAIGHT_FLUSH: 5,
}

def is_bid_feasible(action: int, pool_size: int) -> bool:
    if action >= NUM_BIDS:
        return True
    return pool_size >= _MIN_CARDS_FOR_HAND[index_to_bid(action).hand_type]

def feasible_action_mask(pool_size: int) -> np.ndarray:
    """Length-NUM_ACTIONS bool mask (CALL/HH always True)."""
    mask = np.empty(NUM_ACTIONS, dtype=np.bool_)
    for a in range(NUM_ACTIONS):
        mask[a] = is_bid_feasible(a, pool_size)
    return mask
```

`registry.py` keeps its private `_is_bid_feasible` for now — we do not
refactor it in this phase (would touch every Exact* agent and risk
behavior drift). A separate cleanup ticket can dedupe later. The two
implementations are bit-identical; we add a regression test that
asserts they agree on every `(action, pool_size ∈ [1, 25])`.

## 9. Testing

New tests under `tests/agents/`:

- **`test_contracts.py`**:
  - 1000 random `MatchState`s → `Infostate.from_match_state` →
    JSON round-trip → equality.
  - `feasible_mask` agrees with `legal_actions ∩ is_bid_feasible`.
  - `HandBelief` invariants hold for handcrafted q (uniform, peaked,
    mask-violating). Mask-violating constructions raise.
  - `BidDistribution` same.
  - `Protocol` runtime check: an `IdentityHandModel` (uniform-over-
    feasible) satisfies the `HandModel` Protocol via `isinstance` with
    `runtime_checkable`.
- **`test_checkpoints.py`**:
  - Save → load → state_dict key parity for a small `nn.Linear`.
  - Schema version, component name, parent_run all round-trip.
  - `load_component(path)` on a non-existent path raises `FileNotFoundError`.
- **`test_feasibility.py`** (under `tests/game/`):
  - `is_bid_feasible` agrees with `registry._is_bid_feasible` on every
    `(action, pool_size)` in `range(NUM_ACTIONS) × range(1, 26)`.
  - `feasible_action_mask(n)` is the vectorized form of `is_bid_feasible`.

No test introduces or modifies a slow marker; AR-0a tests are all <1s.

## 10. Acceptance for AR-0a

(Re-stating the parent plan's gate.)

- All new tests pass; existing `tests/` suite has no new failures vs.
  the pre-flight baseline.
- Scratch script: build any registry agent → wrap with
  `policy.decision()` → `Infostate.from_match_state` → round-trip
  through JSON → recover the same `action_probs`. End-to-end works
  for at least `RandomAgent`, `ExactRulesConditionalAgent`,
  `CFRNashAgent`. (Implemented as a single integration test rather
  than a scratch script.)

## 11. Open questions punted to AR-0b / AR-1

- **Compression of `feasible_mask` in trace JSONL.** Defer until trace
  size is actually a problem; gzip-at-rotate is the dumb fix.
- **Whether `decision()` should be the new mandatory contract** for
  registry agents. We keep it optional: the heuristic ladder can adopt
  it incrementally during AR-1's baseline-adapter work.
- **Should `BidDistribution.feasible_mask` and `legal_mask` be one
  field?** They are conceptually different (one is engine, one is
  game-theory). Keep separate; the cost is one extra 112-byte array.
