# Liar's Poker RL Agent — Design & Planning Document

**Status:** Stage 2 of the Liar's Poker project. Stage 1 (combinatorial/probability analysis) is complete and lives in `Liars-poker.tex`. This document scopes a reinforcement-learning agent that plays card-based Liar's Poker at a high level.

**Goal:** Train an agent capable of near-Nash-equilibrium play in card-based Liar's Poker using **Regularized Nash Dynamics (R-NaD)**, starting from the blind (no-private-info) variant and scaling to the full game. Provide a web interface for human vs. agent play.

---

## 1. Scope & Milestones

| # | Milestone | Deliverable | Status |
|---|-----------|-------------|--------|
| M0 | Literature survey on R-NaD and related CFR-free equilibrium methods | `LITERATURE_SURVEY.md` populated; Shi et al. 2022 (APU/CTDE) integrated | ✅ Done |
| M1 | Game engine (pure Python) | `agent/game/engine.py` + `agent/game/bids.py`; unit tests passing | ✅ Done |
| M2 | Blind-game baseline strategy | `agent/baseline/blind_equilibrium.py`; backward induction for N=2; 8/8 tests | ✅ Done |
| M2b | Warm-start lookup | `agent/rnad/warm_start.py`; marginal + conditional vectors for R-NaD network; 16/16 tests | ✅ Done |
| M6a | Minimal web interface | FastAPI + HTMX; playable vs. random and blind baseline; `agent/web/` | ✅ Done |
| M3 | R-NaD trainer (self-play) | `agent/rnad/config.py`, `network.py`, `trainer.py`, `eval.py`; 10/10 tests; Stage A (fixed hand size) complete | ✅ Done (Stage A) |
| M6b | Full web interface | Trained agent, stats recording, session replay, match history | 🔲 After M3 |
| M4 | Full-game R-NaD agent with private information | Trained network checkpoint + evaluation vs. blind baseline and random | 🔲 Planned |
| M5 | Test-time compute extension | Search-augmented inference (MCTS-style or iterative best-response) | 🔲 Planned |

**Priority rationale:** M6a (minimal web UI) is placed before M3 so that the game engine can be validated interactively and agents can be observed while training. M6b (polished UI with stats) follows M3 once there is a trained agent to serve.

**Note on M1:** The original M1 called for OpenSpiel integration; the pure-Python engine in `agent/game/engine.py` satisfies the milestone requirements without requiring OpenSpiel as a dependency.

---

## 2. Game Specification (for OpenSpiel)

### 2.1 Parameters
- `num_players`: **2..5**, variable per game. Formal exploitability claims are made only for N=2; N=3..5 is reported as empirical results (see §5.3 and §9).
- `deck`: standard 52-card deck, no jokers.
- **No fixed `cards_per_player`.** Hand sizes are dynamic and grow as players lose rounds (see §2.2).

### 2.2 Match structure (multi-round elimination)
A *match* is a sequence of *rounds*. Within a round, the card-based Liar's Poker bidding game is played once.

- **Initial state.** Every player starts with `hand_size = 1`.
- **Round.**
  1. **Deal.** Each player is dealt `hand_size_i` fresh private cards from a freshly shuffled 52-card deck. The remainder of the deck is dead for the round.
  2. **Bidding.** Starting from a designated first bidder (loser of the previous round, or seat 0 in round 1), players alternate in seat order. Each turn the acting player either announces a strictly stronger bid than the current standing bid, or calls the previous bid a lie ("call").
  3. **Resolution.** On a call, all hands are revealed and concatenated into the pool. If the pool contains the bid hand, the caller loses the round; otherwise the bidder loses.
- **Between rounds.** The loser's `hand_size` is incremented by 1.
- **Elimination.** A player who loses a round while already holding 5 cards is eliminated from the match. (Hand sizes therefore range over 1..5; no player ever holds 6.)
- **Match terminal.** The match ends when only one player remains. Terminal reward: +1 to the survivor, −1 to any player eliminated, 0 to players who are still active but not the winner (should not occur — matches run to a single winner). Zero-sum.

Pool size `n = sum(hand_sizes of active players in the current round)` ranges over **2..25**, matching Stage 1's computed probability tables exactly. This is the structural hook that makes Stage 1 a direct warm-start for Stage 2 (see §4 and §5.4).

### 2.3 Pairwise decision structure (core insight)
Although a round can involve up to 5 seated players, *every individual decision point is effectively a pairwise interaction*:
- The acting player compares the standing bid (set by the previous bidder) against their private information and the public pool prior.
- They choose between **call** (binary, resolves the round against the previous bidder) or **raise** (continuous over the bid space, exposing themselves to exactly one next decision-maker).
- The identity of the "previous" and "next" player rotates round-robin, but the local structure is always 1-vs-1 against the previous bidder.

This means the policy network needs only to encode (a) my private cards, (b) the standing bid, (c) a compact summary of the public state (pool size, hand-size profile, bid history), and (d) the identity of the previous and next bidders. It does *not* need to reason jointly about all N−1 opponents at every decision. The architecture in §5.4 reflects this.

### 2.4 Bid space
Ordered list of (hand_type, primary_rank) pairs per `_evaluate_ranked` in `poker_math_exact.py`. Exact ordering matches the threshold tables in the paper. Total bid space size: 10 hand types × up to 13 primary ranks, minus invalid combos ≈ 100 bids. The bid ordering is independent of pool size n; what changes with n is the *prior probability* of each bid being truthful.

### 2.5 Information state (from player i's view)
- Player i's private cards (variable length, 1..5)
- Hand-size vector for all active players (public)
- Active-player mask (who has been eliminated)
- Full bid history for the current round
- Match history summary (round outcomes, optional — may be compressed to per-player lose counts)
- Current player to act and previous bidder's seat

### 2.6 Tests against ground truth
For each (n, round-size-profile) configuration with n ∈ {2..7}, enumerate all deals and verify:
- Legal bid space size matches spec
- Terminal payoff computation matches `_evaluate()` on the concatenated pool
- Chance-node marginals match `get_hand_probabilities(n)`
- Round-to-round hand-size transitions respect the elimination rule (nobody ever reaches 6 cards; the 5-card loser is removed and the match continues with N−1 seats)

---

## 3. Literature Survey (M0 — to populate)

### 3.1 Core R-NaD references
- **Perolat et al. 2022** — "Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning" (arXiv:2206.05825). Defines R-NaD: replicator dynamics + entropy regularization + policy iteration. Stratego as flagship imperfect-information large-state benchmark.
- **Dewey et al. 2025** — "Mastering Liar's Poker via Self-Play and RL" (arXiv:2511.03724). Dice-variant Liar's Poker via R-NaD. Provides architecture and hyperparameter reference points; does **not** use test-time compute — we will extend.

### 3.2 To review (search queries for Explore agent)
- R-NaD / NeuRD / magnetic mirror descent convergence theory
- Regret minimization vs. policy-gradient equivalences in imperfect-information games
- Neural Fictitious Self-Play (NFSP) — predecessor baseline
- Deep CFR / ESCHER — alternative equilibrium solvers (comparison baselines)
- Test-time search in imperfect-information games: ReBeL (Brown et al.), Player of Games (Schmid et al.), Search in Matrix Games (Sokota et al.)

### 3.3 Architecture reference points (see §5.4 for committed design)
Design is now specified in §5.4. Literature survey serves to validate hyperparameters and identify test-time compute method (§6).

---

## 4. Blind Baseline & Probability Warm-Start (M2)

Stage 1 produced blind marginal tables (`hand_probabilities.json`) and conditional tables (`conditional_probs_data.json`) for pool sizes n=5..25. Stage 2 uses these in two distinct but complementary ways:

### 4.1 Blind baseline (equilibrium reference)
The **blind variant** removes private cards — players bid over the shared public pool prior only. This is a family of small extensive-form games, one per `(N, hand_size_profile)`, each with a known common prior.

- Ground-truth equilibrium computable by backward induction / LP on the small EFG.
- Each configuration produces an equilibrium bid-frequency table, cached to `baseline_blind_{N}_{profile}.json`.
- Serves as: (a) R-NaD sanity check (trainer must recover the cached policy within ε on the blind variant), and (b) a curriculum stage (pretrain on blind, fine-tune on full game).

### 4.2 Probability tables as network warm-start
The Stage 1 marginal and conditional probability tables feed directly into the policy/value network as **fixed auxiliary features**, so the agent starts from Bayesian-optimal beliefs rather than learning them from scratch:

- **Marginal prior feature.** For the current pool size n, look up `get_hand_rank_counts(n)` and pass the full 100-dim distribution over (hand_type, primary_rank) as a state feature. The network sees what an opponent *with no private info* should bid truthfully.
- **Conditional posterior feature.** For each observed private-hand condition (pair, trips, suited, etc.), look up the matching conditional distribution and pass that as an additional feature. The network sees the Bayes-updated pool distribution given its own hand.
- **Auxiliary loss (optional, empirical).** Add a supervised auxiliary head that must predict the pool hand distribution given the private cards; target is the tabulated conditional probability. This regularizes the representation toward probabilistic ground truth and typically speeds convergence substantially.

Expected benefit: the network never has to rediscover "a pair of 2s in my hand means the pool is more likely to contain at least a pair" from scratch — that fact is baked into the input features.

### 4.3 Extended conditional probability tables (M2 compute task)
Stage 1's conditional tables are coarse: `pair` aggregates all pair ranks, `trips` aggregates all trip ranks, etc. For warm-start to carry maximum information, we need **rank-specific conditions**:

| Current (coarse) | Extended (rank-specific) |
|---|---|
| `pair` (one per n) | `pair_2, pair_3, ..., pair_A` (13 per n) |
| `trips` | `trips_2, ..., trips_A` (13 per n) |
| `suited` (any 2 same-suit) | `suited_{low_rank}_{high_rank}` or at minimum `suited_high_{rank}` (~13 per n) |
| `3suited` | `3suited_high_{rank}` (~13 per n) |
| `adjacent` | `adjacent_low_{rank}` (~12 per n) — "12-A" etc. |
| `3range` | `3range_low_{rank}` (~10 per n) |

For each extended condition × each pool size n ∈ {5..25}, run Monte Carlo with ≥1M samples and record both the type-level distribution and the rank-level distribution via `_evaluate_ranked`.

**Compute profile.** ~80 conditions × 21 pool sizes × 1M samples ≈ 1.7B hand evaluations. At ~500k/s single-threaded Python (current `poker_math_exact.py` throughput), this is a multi-day overnight run. Options to accelerate:
- Parallelize across conditions using `multiprocessing` (trivial; near-linear speedup on 8+ cores — brings the run to a single overnight).
- Share one MC sweep per n across all compatible conditions (reject-sample by condition post-hoc from a single large draw — ~10× speedup because drawing cards is cheaper than evaluation).
- Optional: rewrite `_evaluate` in Cython/Numba for another 5-10× if needed; revisit only if pure Python is too slow after parallelization.

**Deliverable.** `compute_extended_conditional_probs.py` script producing `figures/extended_conditional_probs.json`, cached and reloaded on rerun per the project-wide MC caching rule. The existing coarse tables remain for the paper's Stage 1 figures; the extended tables are additional and live alongside them.

### 4.4 R-NaD convergence sanity check
The R-NaD trainer, when run against the blind variant with warm-started features disabled, should converge to the §4.1 equilibrium within ε. This is a precondition before turning on warm-start features and moving to the full game.

---

## 5. R-NaD Trainer (M3–M4)

### 5.1 Algorithm sketch
Reward-transformed self-play with entropy regularization:
- Maintain a running policy π and a regularization anchor π_reg
- At each outer iteration:
  1. Collect self-play trajectories under π
  2. Transform rewards: r̃ = r − η·(log π − log π_reg)
  3. Policy-gradient update toward the transformed-reward best response
  4. Periodically update π_reg ← π (or Polyak average)
- Convergence: the sequence of anchors traces out a path to a Nash equilibrium of the unregularized game.

### 5.2 Infrastructure choices
- **Framework:** OpenSpiel's `python/algorithms/` for the self-play loop; PyTorch for the network (better ecosystem for transformers than TF1 JAX in OpenSpiel). An alternative is to use the JAX reference R-NaD in DeepMind's Stratego release if it's open-sourced.
- **Compute:** Start on a single GPU (RTX-class). Blind variant should converge on CPU.
- **Logging:** Weights & Biases or TensorBoard; track exploitability against best-response computed via `exploitability.py` from OpenSpiel.

### 5.3 Evaluation
- **Exploitability** (OpenSpiel `exploitability.nash_conv`) — primary metric for N=2. For N≥3, report approximate best-response win rate as an empirical proxy.
- **Multiplayer training stability (Stage C).** When N>2, simultaneous policy updates across all N seats cause oscillation. Apply **APU (Asynchronous Policy Update)** at each outer iteration: designate one seat as the active learner; all other seats play under the current anchor π_reg (frozen). This maps naturally onto R-NaD's existing π_reg mechanism. Reference: Shi et al. 2022 (§4, Multi-Agent Generalization section of LITERATURE_SURVEY.md).
- **Head-to-head win rate** vs.:
  - Uniform random
  - Blind baseline (M2, §4.1)
  - Previous training checkpoints (to verify monotonic improvement)
- **Calibration**: does the agent's bid distribution match the theoretical marginals/conditionals when it has no private info beyond what is tabulated?
- **N-generalization**: train at variable N ∈ {2..5}, evaluate at each N separately, report per-N exploitability (N=2) and per-N win rates (N=3..5).

### 5.4 Network architecture (committed)
Single network, shared weights across all (N, hand-size) configurations.

- **Private-card encoder.** DeepSet or small self-attention over the agent's 1..5 private cards. Each card embedded as `(rank_embedding, suit_embedding)`; summary is a fixed-dim vector independent of hand size. Padding masked.
- **Stage 1 prior features (warm-start, §4.2).** For the current pool size n, concatenate:
  - Marginal hand-type distribution (10-dim) and rank-level distribution (~100-dim) from `get_hand_rank_counts(n)`.
  - Matched conditional distribution from the extended conditional tables (§4.3), keyed on the agent's private hand's most specific matching condition (e.g. "pair of 2s" if the agent holds one).
  These features are computed from static lookup tables, not learned — they are deterministic state features.
- **Opponent encoder.** Transformer over N−1 opponent descriptors `(hand_size, seat_offset_from_self, eliminated_flag)`. Variable length; handles N ∈ {2..5} with the same weights.
- **Bid history encoder.** Small causal transformer or GRU over the sequence of bids in the current round, with positional encoding for seat and turn index. The previous bidder is marked with a distinguished flag (reflecting the pairwise structure from §2.3).
- **Trunk.** Concatenate (private-card summary, prior features, opponent summary, bid-history summary, scalar state features like `n`, `round_index`, `match_hand_size_profile`). Pass through 3–6 transformer/MLP layers.
- **Heads.**
  - Policy head: categorical over the bid space (~100 bids) + 1 "call" action. Illegal-action mask applied.
  - Value head: scalar ∈ [−1, 1] for match-level reward.
  - (Optional) Auxiliary prior-prediction head (§4.2): predicts the pool hand distribution given private cards; supervised against the tabulated conditional probabilities. Auxiliary loss, not used at inference.

### 5.5 Curriculum
1. **Stage A — fixed hand size, N=2.** Freeze everyone at hand_size=1 (then 2, 3, ...). Verify R-NaD convergence against the blind baseline equilibrium on each small variant. Sanity check only.
2. **Stage B — dynamic hand size, N=2.** Turn on the full match structure with elimination. This is the first "real" game.
3. **Stage C — variable N ∈ {2..5}.** Randomize N per episode. Train the same network to generality across table sizes.
4. **Stage D — test-time compute (M5).** Wrap the trained network in a search procedure.

---

## 6. Test-Time Compute (M5)

The Dewey et al. paper trains a static policy with no search at inference. We extend with test-time compute.

### 6.1 Candidate methods
1. **ReBeL-style subgame solving** — reconstruct the public belief state at the current decision point and solve the subgame to depth d using the trained value network as a leaf evaluator.
2. **Iterative best response** — starting from the network policy, run a few steps of CFR on the current infostate and play the refined policy.
3. **Monte Carlo tree search over public states** — treat the opponent's hidden cards as chance and sample rollouts, weighted by the network's belief over opponent holdings.

The choice will be driven by the literature survey in §3.2. ReBeL is the most principled but most engineering-heavy; iterative best response is the cheapest to prototype.

### 6.2 Evaluation
- Does test-time compute reduce exploitability vs. the raw network?
- At what compute budget does it plateau?
- Head-to-head vs. the raw network (should strictly dominate).

---

## 7. Web Interface (M6)

### 7.1 Stack (proposed)
- **Backend:** FastAPI (Python) serving:
  - Game state endpoints
  - Agent move endpoint (loads trained checkpoint, runs inference ± test-time compute)
  - Stats persistence (SQLite is sufficient for personal use)
- **Frontend:** Vanilla JS + a light framework (HTMX or plain React). Card rendering via SVG.
- **Hosting:** Local first; optionally deploy to a small VPS or Hugging Face Spaces.

### 7.2 Features
- New game setup: players, cards per player, agent strength (raw network / +search)
- Hand-by-hand play with move history
- Per-session stats: win rate, exploitability proxy (avg loss vs. agent), bid-accuracy vs. equilibrium frequencies
- Results summary page aggregating across sessions — doubles as a public-facing summary of the paper's findings

### 7.3 Stats recorded
- Anonymized session ID, timestamp, game config
- Full move log (for later replay / analysis)
- Outcome + per-decision Bayesian regret estimate against the agent's policy as oracle

---

## 8. Repository Layout (proposed)

```
papers/Liars poker/
├── Liars-poker.tex              # Stage 1 paper
├── CLAUDE.md                    # project bootstrap
├── AGENT_DESIGN.md              # THIS FILE
├── poker_math_exact.py          # Stage 1 library (shared)
├── generate_prob_tables.py      # Stage 1
├── compute_conditional_probs.py # Stage 1
├── figures/                     # Stage 1 outputs
└── agent/                       # Stage 2 — to be created
    ├── game/
    │   ├── liars_poker_cards.py # OpenSpiel game implementation
    │   └── tests/
    ├── baseline/
    │   └── blind_equilibrium.py # M2
    ├── rnad/
    │   ├── network.py
    │   ├── trainer.py
    │   └── eval.py
    ├── search/                  # M5 test-time compute
    ├── web/                     # M6 interface
    │   ├── backend/
    │   └── frontend/
    └── checkpoints/             # trained models (gitignored)
```

Per workspace convention, all Stage 2 code lives inside the paper folder — no code in a top-level `scripts/`.

---

## 9. Open Questions

**Resolved (2026-04-08):**
- ~~Player count.~~ **N ∈ {2..5}, variable per game.** Formal exploitability claims restricted to N=2; N=3..5 reported empirically. See §2.1, §5.3.
- ~~Cards per player.~~ **Dynamic hand sizes 1..5**, match played as a multi-round elimination game. Loser of a round holding 5 cards is eliminated. See §2.2.
- ~~Warm-starting.~~ **Yes, via fixed tabulated prior features** (marginal + rank-specific conditional probabilities) passed as deterministic state features, not as network initialization. See §4.2, §5.4.

**Still open:**
1. **Game length.** Long bidding sequences inflate the state space. Do we cap the number of raises per round, or rely on natural termination?
2. **Public state reconstruction for ReBeL.** How expensive is belief propagation over card Liar's Poker public states? Dice are exchangeable; cards are not (suits break symmetry). Affects M5 method choice.
3. **Auxiliary loss weight.** How aggressively should the conditional-prediction auxiliary head (§5.4) be weighted? Too high → overfits to the tabulated prior; too low → no benefit. Sweep empirically.
4. **Extended conditional granularity ceiling.** Do we stop at "pair of 2s" level, or push to "pair of 2s + adjacent suited kicker"-type compound conditions? Each level of granularity multiplies MC compute; diminishing returns are likely past single-feature conditioning.
5. **Paper vs. product.** Is the web interface part of the academic artifact (reproducible demo) or a separate side project? Affects polish level.

These should be resolved as the literature survey (§3.2) completes and before M3 starts in earnest.
