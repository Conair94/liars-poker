# Agent Redesign — Design Doc

- **Status:** Draft (design-first gate; no code this session)
- **Date:** 2026-04-27
- **Owner:** main
- **Predecessor:** P5 (commit `89e3443`) — training infrastructure (HH adapter,
  LBR + sampled-subgame exploitability, decision logging, reflect rules)
  is now in place. This doc plans the *agents* that infrastructure exists to
  measure and train.
- **Supersedes:** the heuristic `Exact*` ladder in
  [src/agents/registry.py](../../src/agents/registry.py) (kept as heuristic
  baseline), `CFRNashAgent` (mb3 / mb4 checkpoints — **defunct**, not a
  baseline for new testing), and the existing `RNaD` trainer
  (`src/agents/learned/rnad/`) — **defunct**, superseded by AR-3's modular
  R-NaD trainer. No new tests should reference `CFRNashAgent` or `RNaD`.

---

## 1. Motivation and current weakness

Per the latest benchmark snapshot (memory, 2026-04-24):

| Agent | Aggregate win rate | Notes |
| --- | --- | --- |
| `ExactRulesConditional` | 85.8% | Strongest current agent; pure heuristic peak-prob with hand-conditioning |
| `CFRNashAgent` (mb3) | 49.2% | **Defunct.** Plays Nash on a *restricted* bid space (HC+Pair, 26 of 110 bids); near-random against the heuristic ladder. Not used as baseline going forward. |
| `RNaD` (Stage A, 5k iters) | 0.59 vs random under exact rules | **Defunct.** Pre-dates HH wiring and eval-fix; cannot be trusted today. Superseded by AR-3. |

The "good" agent is a hand-coded peak-probability heuristic with
opponent-rank bumps and HH declaration glued on; the "Nash" agent is a
solver running on a 26-bid abstraction. Neither is a self-play-trained
agent on the real 110-bid HH-enabled game with full pool sizes. P5 gives
us LBR + sampled-subgame exploitability so we can finally **measure** how
exploitable any candidate is on the real game; this doc is what to put on
the other side of that measurement.

### Empirically observed shortcomings (carried forward as design constraints)

1. **Implausible-bid pathology.** Past learned agents put non-trivial
   policy mass on bids whose hand type is physically impossible at the
   current pool size (e.g. a Straight bid at `n=4`). A purely
   probabilistic baseline (no learning) trivially never does this — see
   `_is_bid_feasible` in [registry.py:75](../../src/agents/registry.py#L75).
   The new agent must do *no worse than* this baseline on plausibility,
   because rejecting infeasible bids is free information.
2. **Single-network "everything at once" learning** confounds three
   distinct judgement calls. The agent has to (a) infer the pool's
   hand distribution from public information, (b) decide whether the
   standing bid is honest, and (c) decide what to bid — and gradient
   signal on the third washes out the first two.
3. **Rank leakage.** A deterministic opener leaks the bidder's rank in a
   single bid. The current `Mixed`/`OpponentModel` ladder addresses this
   by structured 4-way mixing; a learned agent must **also** mix and the
   mixing must be derived from regret/optimality, not from a literal
   `random.randint(4)`.
4. **Restricted-game Nash plateaus.** Past CFR runs solved subgames at
   `max_bids=4` on a 26-bid space; the resulting policy is locally
   correct but globally limited. We should not ship a Nash agent that is
   only Nash on a pruned tree.

## 2. Goals and non-goals

### Goals

- Define the **architecture** and **training plan** for a new generation
  of agents that beat `ExactRulesConditional` decisively on the full
  real game (110 bids, exact rules, HH enabled), measured by both
  pairwise win rate and the LBR / sampled-subgame exploitability
  metrics shipped in P5.
- Sequence the work as a curriculum: **1v1 with up to 5 cards each →
  5-player with 5 cards each**. (User constraint.)
- Decompose the policy into three components — hand model, call/no-call,
  bid choice — with a clean enough interface that any one can be swapped
  for an exact / oracle / heuristic / learned implementation.
- Be hard-coded to never assign nonzero policy mass to physically
  infeasible bids. Plausibility is a constraint, not a learned objective.
- Reuse P5 infrastructure end-to-end: the policy contract in
  [src/agents/policy.py](../../src/agents/policy.py), the OpenSpiel
  adapter in [src/interop/openspiel_adapter.py](../../src/interop/openspiel_adapter.py),
  and the LBR / subgame metrics in `src/training/metrics/`.
- **Train and evaluate many agents in parallel.** The pipeline must
  support sweeping over architectural and training hyperparameters
  (HandModel depth, BidPolicy entropy floor, R-NaD η, population mix,
  call-threshold prior, etc.) as concurrent runs, with **granular,
  structured I/O at every component boundary** so any one head can be
  swapped, frozen, or replaced with an oracle and the whole stack still
  composes. Automated research — sweep, benchmark, ablate, regress —
  is a first-class requirement, not a tooling afterthought.

### Non-goals

- Multi-round (with elimination) training. M4 / full-match work is
  explicitly out of scope; we train and evaluate on **single-round**
  matches first, exactly as P5 did.
- A new game engine. Engine work is closed.
- Five-Kings rules. Out of scope per ADR-005 status quo.
- Web frontend integration. Frontend is retired (ADR-001/004).
- Beating a hand-tuned human; the bar is "decisively beat the heuristic
  ladder *and* score low on LBR + sampled-subgame exploitability."

## 3. Why "1v1 first, then 5p" is principled

The user's intuition — that liar's poker with `k > 2` players is, at
each turn, a **local 1v1 decision** between the actor and the standing
bidder — is almost exactly correct, with one caveat we have to handle:

- The **call decision** at turn `t` for actor `i` is genuinely 1v1
  against the standing bidder `j`: payoff is ±1, only `j`'s policy
  matters, and the only uncertainty is the pool composition given
  everyone's hands. The other `k − 2` players' policies are irrelevant
  to *this* call.
- The **bid choice** is *almost* 1v1 — actor `i` is choosing what to put
  in front of the next player `i + 1`, who will be the next caller.
  Once `i + 1` calls or raises, players `i + 2 …` are again irrelevant
  to that interaction.
- The caveat: **pool composition** depends on everyone's hand. When
  hand sizes are uneven across seats, the actor's belief about the pool
  is conditioned on its own hand *and* the public bid history of every
  player. The bid history part *is* genuinely multi-player information
  — each prior bid is a (possibly mixed) signal about the bidder's
  private hand.

The principled translation:

- **Hand-model component must be multi-player aware** — it conditions
  on every public bid, not just the most recent one.
- **Call and bid components can be trained 1v1** and lifted to 5p with
  little change, because at the moment of decision the actor only faces
  one opponent (the standing bidder for calls; the next actor for bids).
- The 5p curriculum step is mostly about (a) the hand-model component
  generalising from "one opponent's bid history" to "k − 1 opponents'
  bid histories," and (b) re-tuning whatever mixing temperature we end
  up with so leakage stays bounded under more eyes.

This means the 1v1 → 5p jump is a **fine-tuning** step, not a fresh
training run, provided the architecture treats opponents permutation-
invariantly.

## 4. Three-component architecture

We split the policy into three pieces with explicit interfaces.
Diagram (one decision):

```
  public state ┐
               │
  own hand ────┼──►  [HandModel]  ────►  q(pool | info)        (NUM_BIDS,)
               │                              │
  bid history ─┘                              │
                                              ▼
                          [CallPolicy]  ──► P(call) ∈ [0, 1]
                                              │
                                              ▼
                          [BidPolicy]   ──►  π(action | info, q, ¬call)
                                                  ─►  bid distribution
                                                  ─►  HH special-cased
```

The three components are **independently trainable** but share a
forward pass at inference (`q` is computed once and consumed twice).

### 4.1 HandModel — `q(pool | info)`

**Output.** A distribution over the pool's best 5-card hand, in the
same 110-bid index space we already use. (Or, equivalently, over the
`(hand_type, primary_rank)` lattice; the bid index is a bijection.)

**Input.** The full public infostate at decision time:

- Own hand (private; permutation-invariant over cards — DeepSet over
  card embeddings, like the existing `RNaD` net in
  [network.py:1](../../src/agents/learned/rnad/network.py)).
- Pool size `n` and per-seat hand sizes (used as a positional feature).
- Bid history as a sequence of `(seat, bid_index)` tokens. **Crucially,
  encoded as a transformer / RNN over the *full* sequence**, not a
  bag — order and identity matter (each player reveals different
  information).
- The current player's seat (for symmetry-breaking when seats matter,
  e.g. who acts next).

**Why not "just compute the conditional probability table"?** Because
the project already has those tables — they are what `ExactRulesCondi-
tional` uses. They condition only on `(own_hand, n)`. The new HandModel
must additionally condition on **the bid history**, which the analytic
tables cannot do without combinatorial blowup. That is precisely the
informational gap the new agent is meant to close.

**Training signal (options, ranked).**

1. **Supervised on rolled-out games (preferred for v1):** generate
   self-play rollouts under any policy, log the actual final
   `(pool_type, pool_primary)` for each decision, train HandModel as a
   classifier with cross-entropy. This is purely an inference task —
   no game-theoretic subtlety — and gets us a strong prior.
2. **Bayesian-consistent training:** at each rollout step, compute the
   posterior over `q` given the *opponent's mixed bid policy* and
   distill HandModel toward that posterior. Cleaner game-theoretically
   but expensive; defer until v1 baseline exists.
3. **Auxiliary head on a unified policy net:** what the existing R-NaD
   trainer attempts, with predictable confounding. We keep this
   *option* available but it is no longer the primary path.

### 4.2 CallPolicy — `P(call | info, q)`

**Output.** Scalar in `[0, 1]`. (`HH` is split off; see §4.4.)

**Input.** Same infostate features as HandModel, **plus** the
HandModel's output `q` and the standing bid `b`.

**Why split it out.** The call decision is the cleanest piece of the
game — payoff is exactly ±1, only depends on whether the pool actually
satisfies `b`, and `q` is exactly the relevant sufficient statistic.
Given a calibrated `q`, the *optimal* call rule is

```
call iff E_q[ pool ⊨ b ]  <  0.5
```

— literally the threshold the heuristic agent already uses. Two reasons
to learn it anyway:

1. `q` from HandModel will be calibrated *for the pool distribution
   only*; it does not know the **opponent's bluff propensity**, which
   shifts `P(b honest | b made)` away from `q`. CallPolicy's job is
   precisely to learn that shift from data.
2. Symmetry-breaking under HH: the optimal rule changes near the peak
   because HH is the better response there.

**Training signal.** CFR-style regret on the call/no-call binary
choice given `q` and `b`. With `q` frozen during this phase, the
update reduces to a one-action regret update against the current
opponent — a textbook two-action Nash on each infoset. Tabular CFR+
on the call substate is feasible because the substate has very low
dimension once `q` is fixed.

### 4.3 BidPolicy — `π(action | info, q, ¬call)`

**Output.** A distribution over **legal, feasible** bid actions.

**Mandatory hard mask.** Before softmax: zero out (a) every action not
in `state.legal_actions()`, (b) every bid index `i` with
`!_is_bid_feasible(i, n)`. This is non-negotiable and subsumes the
plausibility-baseline requirement from §1. The mask is applied at the
logit level (`-inf`) so gradients respect it.

**Soft prior.** Initialize the BidPolicy logits as `log q` (the
HandModel's pool posterior). This is a strong, principled, almost-free
warm start — bidding the most-likely-true hand is a natural anchor —
and it guarantees the agent never starts in the implausible-bid regime
that bit past R-NaD runs.

**Mixing requirement.** A pure (deterministic) BidPolicy leaks rank
(see [feedback_mixed_strategy_insight.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_mixed_strategy_insight.md)).
We do not hand-mix; we let the training objective produce the mixing.
Two viable objectives:

1. **CFR+ on the bid substate** with `q` frozen and CallPolicy fixed
   as the response. Works whenever the bid substate is small (low pool
   sizes, deep into a round). Gives an exact mixed strategy; this is
   the same machinery we already use in
   [src/training/metrics/subgame_exploitability.py](../../src/training/metrics/subgame_exploitability.py).
2. **NeuRD / R-NaD policy gradient** with explicit entropy regularization
   *and* the masked-logit prior from `log q`. R-NaD is what we already
   have a trainer for; we are not throwing that away, just removing its
   responsibility for HandModel and CallPolicy.

### 4.4 HH as a special case

HH is not really a 110th bid; it is "the standing bid is exactly the
pool's best hand." Its optimal rule under the ±1 payoff model is

```
declare HH iff   argmax_q == bid_to_index(b)   AND   q[bid_to_index(b)] is high
```

— i.e. it is purely a function of `q` and the standing bid, with **no
mixing required** at the decision-theoretic optimum. We therefore
implement HH as a small deterministic policy that *consumes* `q` and
the standing bid and outputs HH whenever the rule fires. CallPolicy
and BidPolicy are then conditioned on "HH did not fire."

This is exactly what the heuristic `ExactRulesConditional` does today
(`hh_band` parameter) — we keep the structure but `q` becomes the
learned posterior instead of the analytic conditional table.

## 5. Curriculum

### 5.1 Stage 1 — 1v1, hand sizes ∈ {2, 3, 4, 5}

- **Engine config:** `exact_rules=True`, `high_hand=True`, two seats,
  single round.
- **Training set:** mixture over hand sizes — 25% each of 2, 3, 4, 5
  cards per side. This is *crucial*: training only at `n=10` has been
  the historical failure mode. Per-step pool size is a feature, so one
  net handles all four sizes.
- **Adversaries (population):**
  - `ExactRulesConditional`, `ExactRulesMixed`, `ExactRulesAdaptive`
    (the heuristic ladder).
  - Self-play (current iterate).
  - A frozen historical iterate from 10 k steps ago (avoids forgetting,
    cheap defense against cycling).
  - `RandomAgent` and `BiasedRandom{30,70}` as a sanity floor.
- **Stop criterion (no fewer than two of these have to clear):**
  - Pairwise win rate ≥ 60% vs. every heuristic ladder agent at every
    hand size (currently `ExactRulesConditional` is at ~85.8%; we want
    the new agent to top that).
  - LBR exploitability lower than `ExactRulesConditional`'s LBR.
  - Sampled-subgame exploitability lower than `ExactRulesConditional`'s.

### 5.2 Stage 2 — 5 players, 5 cards each (`n = 25`)

- **Architecture diff:** the bid-history transformer extends to a
  longer sequence and wider seat embedding. HandModel learns to
  condition on `k − 1 = 4` opponents' bid histories.
- **Warm start:** initialize from the Stage-1 checkpoint. Per the
  locality argument in §3, the call and bid components should
  fine-tune cheaply; HandModel is the piece that does most of the
  work in this stage.
- **Adversaries:** the lifted Stage-1 agent (against itself, frozen
  historicals), plus a 5p version of the heuristic ladder.
- **Stop criterion:** same metrics, lifted to 5p tournaments
  (round-robin win rate + 5p LBR — note 5p LBR is more expensive but
  no more conceptually involved than 1v1 LBR). Sampled-subgame
  exploitability is well-defined for `k > 2` zero-sum-team games but
  needs a small extension to the existing solver; flagged in §9.

## 6. Detailed training plan (Stage 1)

The plan is **bootstrap-then-RL**, three phases:

### Phase A — Pretrain HandModel (supervised)

- Generate ~10⁶ self-play rollouts under
  `ExactRulesAdaptive` (it is the strongest current agent and
  produces realistic distributions).
- For every decision point, log `(infostate, true_pool_best_bid_index)`.
- Train HandModel on this dataset to convergence (cross-entropy).
- Sanity check: HandModel's marginal at `n=5..25` ≈ the existing
  `extended_conditional_exact_probs.json` cache (this is the test that
  HandModel did not regress vs. the analytic baseline).
- **Reuse:** the existing card-embedding + DeepSet + bid-history
  encoder from
  [src/agents/learned/rnad/network.py](../../src/agents/learned/rnad/network.py)
  — the architecture is right, only the training loss changes.

### Phase B — CFR+ on call and bid substates given HandModel

- Freeze HandModel.
- For each of a few thousand sampled deals, run CFR+ on the (small)
  bidding subgame using `kuhn_cfr_plus_solve`-style infrastructure
  (this is exactly what the sampled-subgame exploitability metric in
  P5 already does — we are reusing the solver, not the metric).
- Distill the average CFR+ policy into BidPolicy + CallPolicy heads
  (cross-entropy from action probs).
- Output: a strong starting policy that is locally Nash on every
  sampled deal, with HH special-cased.

### Phase C — R-NaD fine-tune end-to-end

- Unfreeze all heads. Continue with R-NaD on full 1v1 self-play.
- Two losses:
  - The standard regret-matched policy loss on the masked logit space.
  - An auxiliary HandModel loss against ground-truth pool outcomes
    (keeps HandModel calibrated against the new, slightly different
    policy distribution induced by self-play).
- Population schedule: 80% self-play, 10% past iterate, 10% heuristic
  ladder.
- Compute target: CPU per
  [feedback_mps_cpu_speed.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_mps_cpu_speed.md);
  MPS is 21× slower for this workload.

### Why not skip A and B?

- Pure end-to-end R-NaD has been the existing path and is the path that
  produced agents currently considered useless. The bootleneck has not
  been compute; it has been gradient noise on a multi-objective task.
- Phases A + B cost ~hours of compute for a checkpoint that is already
  competitive with the heuristic ladder. Phase C then has the easier
  job of polishing rather than discovering structure.

## 7. Other suggestions to fold in

These are the items the user invited under "let me know any other
suggestions"; in priority order:

1. **Action mask in the network, not in the dispatcher.** Plausibility
   and legality masks must be applied at the logit level so gradients
   respect them. This is a known fix for the implausible-bid pathology.
2. **Population-based self-play with a small zoo of frozen historicals**
   (Lanctot et al., StarCraft-style). Cheap insurance against the
   exploitability cycles R-NaD on its own can fall into. The
   exploitability metrics from P5 give us a principled trigger for
   adding a checkpoint to the zoo (e.g. "freeze whenever LBR drops by
   ≥ 10%").
3. **Permutation-invariant opponent encoding** in the bid-history
   transformer for the 5p stage. Specifically: per-seat embeddings
   should be relative to the actor (`seat_offset = (other_seat −
   own_seat) mod k`), not absolute. Generalizes 1v1 → 5p without
   re-learning seat identity.
4. **Calibration audits as a first-class metric.** Once HandModel
   exists, log `Brier(q, true_outcome)` per pool size on a held-out
   set; treat regressions as a test failure. This catches "policy
   improved but inference got worse" silent regressions.
5. **Counterfactual decision tracing.** P3's
   `decision_capture.py` already logs per-turn decisions; extend the
   schema to log `q` and `P(call)` separately so the reflect rules can
   ask "did the agent call when `q` said honest? did it bid against
   `q`?" — this generalizes the existing rank-leak / missed-call rules
   to the modular agent.
6. **Anti-tilt / entropy floor.** At low pool sizes (`n=2..4`) the
   exact analytical strategies are very thin and a learned agent can
   collapse to a deterministic rule that has 0 entropy at the opener.
   Maintain a minimum entropy floor at the BidPolicy logits when
   `n ≤ 4`. (This addresses
   [test_opening_mix flaky](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_test_opening_mix_flaky.md)
   structurally rather than as a test marker.)
7. **Don't kill the heuristic ladder.** The Exact* agents are the
   strongest single-policy adversaries we have. Keep them in the
   benchmark, in the population mix, and in the docs as the bar to
   beat. They are not the future, but they are the floor.
8. **Prefer one shared trunk + three small heads** over three
   independent networks. Cuts inference cost ~3×, ensures `q`,
   `P(call)`, and bid logits see the same encoded state, and matches
   Phase A/B/C training where heads are frozen/unfrozen independently.

## 8. I/O contracts (component-level)

This is the load-bearing section for "automated research." Every
component boundary is a typed function with a stable schema; every
forward pass writes a structured trace; every checkpoint is
component-addressable. With these in place, "swap HandModel for the
analytic oracle and re-run benchmarks" becomes a config change, not a
code change.

All schemas use `dataclasses` (or `TypedDict` at the JSON boundary).
Concrete dtypes are pinned — `np.float32` everywhere except action
indices (`int64`) — so on-disk traces are reproducible across machines.

### 8.1 `Infostate` — the only thing the agent ever sees

Defined once, consumed by all three heads:

```python
@dataclass(frozen=True)
class Infostate:
    own_hand:        tuple[int, ...]      # card ids, sorted, len 1..5
    pool_size:       int                  # n
    hand_sizes:      tuple[int, ...]      # per seat, len = num_seats
    own_seat:        int
    current_player:  int
    standing_bid:    int | None           # bid index 0..NUM_BIDS-1 or None
    bid_history:     tuple[tuple[int, int], ...]  # ((seat, bid_idx), ...)
    legal_actions:   tuple[int, ...]      # mask materialized from engine
    feasible_mask:   tuple[bool, ...]     # len NUM_ACTIONS; product of legal ∩ _is_bid_feasible
    # ruleset-as-data; never inferred from globals
    exact_rules:     bool
    high_hand:       bool
    five_kings:      bool                 # always False under this doc; reserved
```

`Infostate.from_match_state(state) -> Infostate` is the canonical
adapter. **Every** training-time and inference-time consumer goes
through this; no head ever touches `MatchState` directly. This is what
lets us serialize an entire decision trace to JSONL (§9.3) and replay
it offline.

### 8.2 `HandBelief` — output of HandModel

```python
@dataclass(frozen=True)
class HandBelief:
    q:             np.ndarray  # shape (NUM_BIDS,), dtype float32, sums to 1
    q_logits:      np.ndarray  # shape (NUM_BIDS,), pre-softmax (for distillation)
    feasible_mask: np.ndarray  # shape (NUM_BIDS,), bool — applied before softmax
    n:             int         # pool size at the call site (sanity assert)
```

**Invariants** (asserted in tests, not in production):

- `q[~feasible_mask].sum() < 1e-6` — zero mass on infeasible bids.
- `abs(q.sum() - 1.0) < 1e-5`.
- `q_logits[~feasible_mask] == -np.inf`.

**HandModel interface:**

```python
class HandModel(Protocol):
    def belief(self, info: Infostate) -> HandBelief: ...
    def belief_batch(self, infos: list[Infostate]) -> list[HandBelief]: ...
```

`belief_batch` is mandatory — Phase A pretraining and LBR / subgame
evaluation are batched workloads and unbatched inference is the
historical bottleneck.

### 8.3 `CallDecision` — output of CallPolicy

```python
@dataclass(frozen=True)
class CallDecision:
    p_call:  float                  # P(call) ∈ [0, 1]
    inputs:  dict[str, float]       # diagnostics: q_at_bid, peak_q, n_unique_bidders, ...
```

`p_call` is exposed as the scalar; `inputs` is a small bag of named
floats that downstream reflect rules can predicate over without the
network needing a public API for each. Schema is open — adding a new
diagnostic does not break old logs.

```python
class CallPolicy(Protocol):
    def call_prob(self, info: Infostate, q: HandBelief) -> CallDecision: ...
```

### 8.4 `BidDistribution` — output of BidPolicy

```python
@dataclass(frozen=True)
class BidDistribution:
    pi:            np.ndarray  # shape (NUM_ACTIONS,), float32, sums to 1
    pi_logits:     np.ndarray  # shape (NUM_ACTIONS,), masked
    entropy:       float       # H(pi)
    support_size:  int         # |{a : pi[a] > 1e-6}|
```

Same masking invariants as `HandBelief`: `pi[~legal] = 0`,
`pi[~feasible] = 0`, `pi_logits[mask] = -inf`. `support_size` and
`entropy` are precomputed because the entropy floor (§4.3) is a hot
path and we do not want to recompute it from `pi` every time.

```python
class BidPolicy(Protocol):
    def bid_dist(
        self, info: Infostate, q: HandBelief, *, hh_fired: bool
    ) -> BidDistribution: ...
```

`hh_fired=True` short-circuits to the all-zeros distribution because HH
preempted the bid step; we still return a `BidDistribution` (with
`support_size=0`) so calling code is uniform.

### 8.5 `AgentDecision` — what `Policy.action_probs` returns

This is the **only** public surface the rest of the codebase
(benchmark, LBR, subgame solver, decision logger) sees:

```python
@dataclass(frozen=True)
class AgentDecision:
    action_probs: dict[int, float]   # {action_idx: prob}; matches policy.py contract
    chosen:       int | None         # one-shot sample if the caller wants it; None otherwise

    # Granular trace — None when the agent is monolithic (e.g. RandomAgent)
    belief:       HandBelief | None
    call:         CallDecision | None
    bid:          BidDistribution | None
    hh_fired:     bool
```

A `ModularNashAgent` always populates the trace fields; the legacy
`Exact*` ladder and `RandomAgent` populate `action_probs` only and set
the rest to `None`. The existing `agents.policy.action_probs(agent,
state)` helper continues to work unchanged — it sees only the
`action_probs` field.

### 8.6 Training-side I/O (datasets and gradients)

| Phase | Reads | Writes | Format |
| --- | --- | --- | --- |
| A — HandModel pretrain | rollout JSONL (Infostate, true `(hand_type, primary_rank)`) | HandModel checkpoint | `data/runs/<run_id>/handmodel/iter_*.pt` |
| B — CFR+ distillation | HandModel checkpoint, sampled-deal cache | per-deal CFR+ policy + distilled CallPolicy / BidPolicy checkpoints | `data/runs/<run_id>/cfr_deals/*.npz`, `<run_id>/{call,bid}_policy/iter_*.pt` |
| C — R-NaD fine-tune | all three checkpoints | unified checkpoint + decision JSONL | `data/runs/<run_id>/unified/iter_*.pt`, `<run_id>/decisions/*.jsonl` |

**Checkpoint schema** (single rule that makes everything else cheap):
each checkpoint file is a single Python dict with exactly these keys:
`{"component": str, "config": dict, "state_dict": dict, "iter": int,
"git_sha": str, "parent_run": str | None}`. `component` is one of
`{"handmodel", "callpolicy", "bidpolicy", "unified"}`. **No bundled
optimizer state in the same file** — optimizer goes alongside as
`*.opt.pt`. This is the only way "load HandModel from run X, CallPolicy
from run Y, BidPolicy fresh" composes without code changes.

### 8.7 Evaluation-side I/O (decision traces)

Every decision the agent makes during evaluation is serialized to a
JSONL row using the existing
[src/training/decision_capture.py](../../src/training/decision_capture.py)
schema, **extended** with the modular fields:

```json
{
  "run_id": "...", "agent_id": "...", "match_id": "...", "turn": 7,
  "infostate": { ... Infostate as JSON ... },
  "action": 42, "action_probs": {"42": 0.4, "55": 0.3, "110": 0.3},
  "belief": {"q_top5": [[42, 0.31], [55, 0.28], ...], "entropy": 2.7, "n": 10},
  "call":   {"p_call": 0.18, "inputs": {"q_at_bid": 0.12, "peak_q": 0.31}},
  "bid":    {"support_size": 4, "entropy": 1.4, "pi_top5": [[55, 0.4], ...]},
  "hh_fired": false
}
```

`q_top5` / `pi_top5` keep the row size bounded (full 110-vectors per
turn would balloon traces); the full vectors are recoverable from a
seed-pinned re-run if a later analysis needs them. This is the
**single** schema every reflect rule, every exploitability comparison,
and every ablation-vs-ablation post-hoc reads from.

## 9. Parallel training and automated research

The point of §8's discipline is that **everything below this line is
config**, not code.

### 9.1 Run unit and identity

A *run* is a single training trajectory under a fixed config. Every
run has:

- `run_id`: `<phase>-<YYYYMMDDTHHMMSS>-<slug>-<git_sha8>`. Unique.
- `data/runs/<run_id>/config.yaml`: the resolved config (after CLI
  overrides), including the `git_sha`, `random_seed`, `device`, and the
  exact P5 metric configs the run will be evaluated under.
- `data/runs/<run_id>/manifest.json`: declares which checkpoints are
  loaded for which components (e.g. `{"handmodel": "ar1-…/iter_5000",
  "callpolicy": null, "bidpolicy": null}`). Lets a Phase B run reuse
  any prior Phase A checkpoint without copying weights around.

This is the same shape as existing benchmark / training runs under
`data/runs/`; we do not invent a new directory layout.

### 9.2 Parallel sweep driver

A new top-level entry point:

```bash
python -m training.sweep \
  configs/sweeps/handmodel_depth_sweep.yaml \
  --max-parallel 8 --device cpu
```

`configs/sweeps/*.yaml` declares an axis-product or explicit list of
configs. The driver spawns one subprocess per run (no in-process
multi-train; we want full isolation for crash recovery). Each
subprocess writes to its own `data/runs/<run_id>/`. The driver writes
a `data/sweeps/<sweep_id>/index.json` mapping config → `run_id` so
later analysis is a pure read.

**Concurrency controls**: `--max-parallel` and per-run `--cpu-quota`
(integer cores; defaults to `cpu_count() // max_parallel`). MPS and
CUDA are supported but per
[feedback_mps_cpu_speed.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_mps_cpu_speed.md)
CPU is the default and is what sweeps assume.

**Crash recovery**: a run's `manifest.json` tracks `last_completed_iter`
and is written atomically. Re-running `python -m training.sweep …`
resumes any incomplete runs from their last checkpoint and skips
completed ones. Failed runs surface in the sweep summary; we do not
silently retry.

### 9.3 Automated benchmark stage

After every sweep, an automatic post-stage:

```bash
python -m training.bench_sweep <sweep_id> \
  --opponents heuristic_ladder,self,random \
  --metrics winrate,lbr,subgame \
  --games-per-pair 200
```

Reads `<sweep_id>/index.json`, builds every run's final
checkpoint into a `ModularNashAgent`, and runs the existing
`benchmark.py` pipeline pairwise. Output:
`data/sweeps/<sweep_id>/bench/results.parquet` — one row per
`(run_id, opponent, metric, value, ci_low, ci_high)`. Parquet because
typical sweeps produce ~10⁴ rows and we want pandas/polars analysis
without a custom loader.

### 9.4 Component-level ablation matrix

The `ModularNashAgent` constructor accepts each component
independently. To ablate "is HandModel actually buying us anything?":

```python
agent = ModularNashAgent(
    handmodel=AnalyticHandModel(),       # the existing conditional table
    callpolicy=load("ar2-…/callpolicy/iter_final"),
    bidpolicy=load("ar2-…/bidpolicy/iter_final"),
)
```

Three "flavors" ship with the codebase, all implementing the
`HandModel` / `CallPolicy` / `BidPolicy` protocols defined in §8:

- `Analytic*`: closed-form / table-driven. Fast, exact under the model
  it was derived from. Used as oracle baselines.
- `Heuristic*`: extract the relevant logic from the existing `Exact*`
  ladder behind the new interface. No new training; just a refactor.
  Lets the existing strong agents participate in the ablation matrix
  without special-casing.
- `Learned*`: the trained networks.

Ablation runs are just additional rows in a sweep config:
`{handmodel: analytic, callpolicy: learned-ar2-X, bidpolicy: learned-ar2-X}`
is one config; swapping `handmodel: learned-ar1-Y` is another. The
sweep driver does not know or care what the components are made of.

### 9.5 Comparator and reporting

`python -m training.compare <sweep_id> [<sweep_id> ...]` produces:

- **Pareto plots** of (LBR exploitability, win-rate vs. heuristic
  ladder) across runs in the sweep — direct visual answer to "which
  ablation Pareto-dominates."
- **Calibration plots** of HandModel: empirical vs. predicted pool
  distribution, per pool size, on a held-out deal set. Standalone
  metric per §7.4.
- **Decision-log diff:** for two named runs, sample N matches, log
  every turn where they choose different actions, and surface the
  Infostate + both action distributions side-by-side. This is the
  qualitative tool for "why did run A actually do better than run B."

Output: `data/sweeps/<sweep_id>/report/` — markdown + parquet + PNGs.
Generated, never hand-edited.

### 9.6 Reproducibility floor

The hard rules — every run satisfies them or the sweep driver refuses
to start it:

- `random_seed` is set; all stochastic dependencies (numpy, torch,
  python) read it.
- `git_sha` recorded; `git status --porcelain` must be empty for any
  run flagged `production: true` in its config.
- All inputs (parent checkpoints, dataset hashes) recorded in
  `manifest.json`.
- Output dir is content-addressable in the sense that re-running with
  the same `run_id` is an error (forces explicit `--resume` or new id).

This is what makes "automated research" honest: the sweep summary is
not a screenshot, it's a reproducible artifact.

### 9.7 What this gives you on day one

With §8 + §9 in place — before any new agent has been trained — the
following are one-liners:

| Question | Command |
| --- | --- |
| Is the AnalyticHandModel still as good as a trained one? | `bench_sweep` with one row per HandModel flavor, others fixed |
| Which Phase B CFR+ deal count is enough? | sweep over `cfr_deals ∈ {1k, 5k, 10k, 50k}`, plot loss-vs-count |
| Does the entropy floor at n≤4 actually fix opener leakage? | sweep over `entropy_floor ∈ {0, 0.1, 0.5}`, compare LBR + qualitative opener mix |
| Stage 1 → Stage 2 transfer: what fine-tunes fastest? | sweep over `freeze ∈ {none, handmodel, all-but-bid}` for the 5p step |

Every one of these is an `axis × axis` over the §8 component
interface; none of them needs a new code path.

## 10. Component summary

| Component | Output | Trained how (Phase) | Mandatory hard rules |
| --- | --- | --- | --- |
| HandModel | `q ∈ Δ^110` over pool best-hand | Supervised on ground-truth pool outcomes (A); auxiliary loss in (C) | `q[i] = 0` if `_is_bid_feasible(i, n) == False` |
| CallPolicy | `P(call) ∈ [0, 1]` | CFR+ distillation (B); R-NaD (C) | Suppressed when CALL not in legal_actions |
| BidPolicy | `π ∈ Δ^{legal feasible bids}` | CFR+ distillation (B); R-NaD (C) | Logit mask from `legal_actions ∩ feasible`; entropy floor at `n ≤ 4` |
| HH gate | Deterministic `q`-driven rule | None (rule, not learned) | HH only fires when `argmax q == standing bid` |

## 11. Open questions

1. **Joint vs. independent CallPolicy and BidPolicy training.** Phase B
   alternates "solve assuming opponent's old policy" — does this
   converge cleanly when the *opponent's* HandModel/BidPolicy/CallPolicy
   are also evolving? This is essentially the standard R-NaD question
   but with three heads instead of one. Resolve before Phase C.
2. **Sampled-subgame exploitability for `k > 2`.** Today's metric is
   1v1; lifting to 5p is a small but real extension to
   `subgame_exploitability.py` (need to define "team" in zero-sum-team
   for liar's poker — likely "actor vs. all others" with the others
   playing the agent's policy). Scope this with a follow-up doc before
   Stage 2.
3. **HH in 5p.** With 5 potential HH-declarers, the optimal rule is
   no longer "argmax q == bid_to_index(b)" — there is correlated
   incentive to be the *first* to declare. Whether the current rule
   is acceptable as a 5p approximation needs an empirical check.
4. **Compute budget.** Phase A is cheap (~hours). Phase B's CFR+ step
   per sampled deal is the hot path; how many deals (~10⁴? 10⁵?) for
   a stable BidPolicy distillation? Probably empirical — set up a
   loss-vs-deal-count plot in the design follow-up.
5. **Naming.** Per
   [feedback_agent_naming.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_agent_naming.md)
   names must be descriptive, no nicknames. Working name for the new
   agent class: `ModularNashAgent`. Open to alternatives —
   `ComponentwiseAgent`, `ThreeHeadAgent`, `BeliefAgent` are all
   acceptable; pick one before code lands.

## 12. Phasing

This doc is the umbrella. Each numbered phase below gets its own
follow-up design doc + plan + implementation session, per the
design-first gate
([feedback_design_first_gate.md](../../../.claude/projects/-Users-connorlockhart-Documents-GitHub-liars-poker/memory/feedback_design_first_gate.md)).

| Phase | Scope | Predecessor | Design doc |
| --- | --- | --- | --- |
| AR-0 | This doc — agreement on architecture and curriculum | P5 | this file |
| AR-1 | HandModel architecture + Phase-A pretrain pipeline (1v1) | AR-0 | TBD |
| AR-2 | CallPolicy + BidPolicy heads + Phase-B CFR+ distillation | AR-1 | TBD |
| AR-3 | Phase-C R-NaD fine-tune + population mixing | AR-2 | TBD |
| AR-4 | Stage-1 (1v1) acceptance: pairwise + LBR + sampled-subgame | AR-3 | reuse P5 metrics |
| AR-5 | Stage-2 lift to 5-player | AR-4 | TBD (5p metrics) |

## 13. Acceptance criteria for this design doc

This doc is **accepted** when the user signs off on:

- The three-component split in §4 (vs. monolithic policy).
- The bootstrap-then-RL phasing in §6 (vs. pure end-to-end).
- The 1v1-first curriculum in §5 (vs. starting at 5p).
- The hard-mask + plausibility constraint in §4.3 (vs. learned).
- The component-level I/O contracts in §8 and the parallel-sweep
  pipeline in §9 (vs. ad-hoc per-run scripts).

Anything else above is a defensible default and can be redirected by
the user before any AR-1+ design doc is written.

## 14. References

- **Engine:** [src/game/engine.py](../../src/game/engine.py),
  [src/game/bids.py](../../src/game/bids.py)
- **Existing agents:** [src/agents/registry.py](../../src/agents/registry.py),
  [src/agents/heuristic/](../../src/agents/heuristic/),
  [src/agents/learned/rnad/](../../src/agents/learned/rnad/)
- **Policy contract:** [src/agents/policy.py](../../src/agents/policy.py)
- **OpenSpiel adapter (HH wired):** [src/interop/openspiel_adapter.py](../../src/interop/openspiel_adapter.py)
- **Exploitability metrics:** [src/training/metrics/lbr.py](../../src/training/metrics/lbr.py),
  [src/training/metrics/subgame_exploitability.py](../../src/training/metrics/subgame_exploitability.py)
- **Decision logging / reflect:** [src/training/decision_capture.py](../../src/training/decision_capture.py),
  [src/training/reflect.py](../../src/training/reflect.py)
- **Predecessor design docs:** [p5_design.md](p5_design.md),
  [p5_2_exploitability.md](p5_2_exploitability.md),
  [small_games.md](small_games.md)
- **ADRs:** [adr/](adr/) (esp. 002 OpenSpiel adoption, 005 game-id and
  encoding)
