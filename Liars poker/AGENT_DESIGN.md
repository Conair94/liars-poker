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
| M3 | R-NaD trainer (self-play) | `agent/rnad/config.py`, `network.py`, `trainer.py`, `eval.py`; 14/14 tests; Stage A + Stage B (full match/elimination) complete | ✅ Done (Stage A + B) |
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

### 2.6 Exact-Rules Information Structure (key strategic shift)

Under **at-least rules** a bid B resolves to "does the pool's best 5-card hand ≥ B?" — a pure threshold question over the marginal pool distribution. Private cards shift the prior slightly; the blind equilibrium (bid at the 50% threshold) captures most of the strategy. This is **blackjack-like**: optimize against a known distributional quantity.

Under **exact rules** a bid B resolves to "does any 5-card subset of the opponents' hands evaluate to exactly B?" This changes the problem in three fundamental ways:

1. **Bids become claims about opponent holdings, not pool statistics.** A bid of "Full House Aces" claims that at least one opponent holds the specific cards needed to complete that hand. This is only plausible if the bidder believes the opponents' ranges contain those cards. The prior probability tables (marginal/conditional) become secondary; opponent range estimates become primary.

2. **Bid history carries range-narrowing information.** Each bid in the sequence signals which hand categories the bidder believes the opponents can hold. This is structurally analogous to betting in Texas Hold'em: each action updates the Bayesian posterior over opponent holdings. The information content of the history therefore grows faster under exact rules than at-least rules.

3. **Call/fold decisions require opponent range models.** Under at-least rules, the caller asks: "is P(pool ≥ standing bid) < 0.5?" Under exact rules, the caller asks: "can I construct a 5-card subset of plausible opponent holdings that completes the bid?" The second question requires an explicit model of what opponents might hold, weighted by the bid history.

**Consequence for the agent:** The at-least agent can get by with good probability tables and a minimal history encoder. The exact-rules agent needs an explicit Bayesian range tracker that updates a per-opponent range estimate from each observed bid. The range tracker output — not just the marginal pool probabilities — becomes the primary input for both the call/raise decision and the value function.

**Critical limitation of the blind equilibrium under exact rules:**

The blind equilibrium computes the correct Nash for the *blind game* (no private info). Under exact rules, private cards create a **massive conditional shift** that the blind equilibrium cannot see:

| Situation | P(HC A holds \| n=2) | Call EV |
|---|---|---|
| Blind (no info) | 0.145 | +0.710 |
| You hold an Ace | **0.941** | −0.883 ← never call |
| You hold a King | **0.078** | +0.843 ← always call |

The ratio is **12x**. The blind equilibrium's recommendation to "bid HC A" is correct only for a player who holds an Ace. A non-Ace holder bidding HC A will be called almost every time and lose. Similarly, the rational first bid for a player holding rank r is approximately HC r (not HC A), since P(HC r \| hold rank r) ≈ 0.63 — the player commits to the exact hand they know they can support.

**Implication:** Under exact rules, the blind equilibrium is a **convergence sanity check only** — not a strategy guide for the full game. The real strategic signal is the private-card-conditioned probability P_exact(b \| my cards), which is the dominant input to any sensible exact-rules policy. The Bayesian range tracker (§5.4) and conditional probability warm-start (§4.2) become primary, not secondary.

Under at-least rules the private-info shift is small (a pair in hand shifts the ~50% threshold by a few bids). Under exact rules the shift is enormous and the blind equilibrium is misleading if applied to the full game.

**Blind equilibrium comparison results (n=2..10, cached):**

| n | At-least 1st bid | p(holds) | EV₀ | Exact 1st bid | p(holds) | EV₀ |
|---|---|---|---|---|---|---|
| 2 | HC J | 0.566 | +0.085 | HC A | 0.145 | **−0.709** |
| 5 | Pair 2 | 0.500 | +0.000 | HC A | 0.194 | **−0.612** |
| 8 | Two Pair 10 | 0.503 | +0.006 | HC A | 0.496 | **−0.009** |
| 9 | Two Pair K | 0.561 | +0.005 | HC A | 0.543 | **+0.087** |
| 10 | Straight 9 | 0.503 | +0.006 | HC A | 0.586 | **+0.173** |

Key findings:

1. **Exact first bid = HC A for all n=2..10.** Under at-least rules the first bid tracks the ~50% probability threshold across hand types (HC J, Pair 2, Two Pair 10, Straight 9 as n grows). Under exact rules the first bid is always High Card Ace, because HC A is the most probable single exact hand outcome for any pool size — an Ace is the most common max-rank card and High Card is the dominant hand type when pools are small.

2. **First-mover DISADVANTAGE under exact rules at small n.** At n=2 (Stage A hand sizes), EV₀=−0.709 — whoever is forced to bid first has a severe structural disadvantage. This inverts around n≈8 and the first bidder gains advantage at n=9..10. Under at-least rules EV₀≈0 for all n. This is a critical meta-game consequence for the elimination match: losing a round (which forces you to bid first next round at a larger hand size) transitions from a penalty → neutral → bonus as hand sizes grow.

3. **The forcing mechanism.** Bidding HC A forces the opponent to either (a) call a low-probability bid (good for first bidder when n is small and P_exact(HC A) is low) or (b) raise to Pair or higher, committing to a harder-to-hold exact subset. As n grows and P_exact(HC A) approaches and exceeds 0.5, the first bid becomes more credible and the first bidder gains power.

**Note on rule standardization:** Neither "at-least" nor "exact-subset" resolution is formally standardized in Liar's Poker. The game has no canonical rule set across variants and cultures. Both are folk rules; this project defines them precisely in §2 and treats them as two distinct games.

### 2.7 Tests against ground truth
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
- Each configuration produces an equilibrium bid-frequency table, cached to `agent/data/blind_equilibrium.json` (keys `"{n}"` for at-least, `"exact_{n}"` for exact).
- **Under at-least rules:** serves as both (a) R-NaD convergence sanity check and (b) a reasonable first-approximation strategy (private cards shift the threshold only slightly).
- **Under exact rules:** serves as (a) R-NaD convergence sanity check **only**. The blind equilibrium's strategy (bid HC A, call everything) is NOT a useful approximation for the full game. Private cards shift the effective probability by up to 12x (see §2.6). Do not use the blind equilibrium as a warm-start strategy target under exact rules — use the conditional probability tables (§4.2) instead.

### 4.2 Probability tables as network warm-start
The Stage 1 marginal and conditional probability tables feed directly into the policy/value network as **fixed auxiliary features**, so the agent starts from Bayesian-optimal beliefs rather than learning them from scratch:

- **Marginal prior feature.** For the current pool size n, look up `get_hand_rank_counts(n)` and pass the full 100-dim distribution over (hand_type, primary_rank) as a state feature. The network sees what an opponent *with no private info* should bid truthfully.
- **Conditional posterior feature.** For each observed private-hand condition (pair, trips, suited, etc.), look up the matching conditional distribution and pass that as an additional feature. The network sees the Bayes-updated pool distribution given its own hand.
- **Auxiliary loss (optional, empirical).** Add a supervised auxiliary head that must predict the pool hand distribution given the private cards; target is the tabulated conditional probability. This regularizes the representation toward probabilistic ground truth and typically speeds convergence substantially.

Expected benefit: the network never has to rediscover "a pair of 2s in my hand means the pool is more likely to contain at least a pair" from scratch — that fact is baked into the input features.

**Under exact rules**, the probability tables remain useful but play a secondary role. The primary warm-start feature shifts to the **opponent range tracker** described in §5.4. The marginal/conditional tables are still valid as a uniform-prior baseline (before any bid history is observed), but the range tracker quickly dominates once bids reveal information about opponent holdings. The auxiliary loss target should therefore be changed: instead of predicting the pool distribution from private cards alone, predict the updated opponent range distribution from (private cards + bid history). See §5.4 for the full architecture.

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
- **[Exact-rules addition] Bayesian range tracker.** A deterministic module (not learned weights) that processes the bid history and maintains a per-opponent range distribution — a probability vector over plausible hand categories the opponent might hold, conditioned on all bids observed so far. Updated via Bayes' rule: each bid shifts probability mass toward hand categories consistent with rationally making that bid. Implementation:
  - Per-opponent range vector: dim = (num_hand_types × num_primary_ranks) ≈ 100-dim, initialized to the marginal prior from Stage 1 tables.
  - At each observed bid b by opponent o: multiply the range vector by the likelihood L(b | range) — the probability a player with that hand distribution would make bid b — then normalize. Likelihood L can be bootstrapped from the uniform R-NaD policy and refined during training.
  - Output: N−1 range vectors, each ~100-dim, concatenated as a fixed-dim block fed into the trunk.
  - Reference: Ganzfried & Sandholm 2015 (joint distribution via Bayes from blueprint strategy); Johanson et al. 2007 (Restricted Nash Response — range-conditioned exploitation).
- **Trunk.** Concatenate (private-card summary, prior features, opponent summary, bid-history summary, **opponent range vectors**, scalar state features like `n`, `round_index`, `match_hand_size_profile`). Pass through 3–6 transformer/MLP layers.
- **Heads.**
  - Policy head: categorical over the bid space (~100 bids) + 1 "call" action. Illegal-action mask applied.
  - Value head: scalar ∈ [−1, 1] for match-level reward. Receives opponent range vectors as additional input (range advantage affects value directly per Ganzfried & Chiswick 2019).
  - **[Exact-rules addition] Auxiliary range-prediction head:** Predicts the final revealed opponent hand from the bid history at episode end; supervised against true holdings. Replaces (or augments) the pool-distribution auxiliary head from the at-least design. This trains the representation to extract range information from bids — the core skill for exact-rules play.

### 5.5 Curriculum
1. **Stage A — fixed hand size=1, N=2.** With one card per player the pool is 2 cards total; only High Card and the rare Pair can appear. The exact-rules Nash equilibrium is **tractable to compute exactly** (tiny extensive-form game, LP-solvable) and serves as a hard ground-truth sanity check for R-NaD convergence. The range tracker has almost no work to do here — bids carry little range information with 2-card pools. Stage A validates that the trainer converges and produces correct call frequencies before any complexity is added.
2. **Stage B — dynamic hand size, N=2.** Full match structure with elimination (hand sizes 1→5). The blind equilibrium under exact rules must be recomputed (different from the at-least equilibrium; see §4.1 and §2.6). Range tracking becomes meaningful at hand_size ≥ 2 (pool ≥ 4 cards). This is the first "real" game; the range tracker begins to matter.
3. **Stage C — variable N ∈ {2..5}.** This is the **critical leap in difficulty**: multiple opponents means the range tracker must maintain N−1 simultaneous per-opponent posteriors, joint distribution complexity grows, and APU (Shi et al. 2022) must be applied for training stability. Randomize N per episode. Train the same network to generality across table sizes. Formal exploitability claims remain N=2 only.
4. **Stage D — test-time compute (M5).** Wrap the trained network in a search procedure. The range tracker provides the belief state needed for ReBeL-style subgame solving; Stage C's trained range representations are the prerequisite.

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
3. **Auxiliary loss weight (range prediction).** Under exact rules, the auxiliary head predicts opponent range from bid history (§5.4). How aggressively to weight this loss vs. the main R-NaD objective? Too high → overfits to short bid histories; too low → the range representation doesn't develop. Sweep empirically.
4. **Extended conditional granularity ceiling.** Do we stop at "pair of 2s" level, or push to "pair of 2s + adjacent suited kicker"-type compound conditions? Each level of granularity multiplies MC compute; diminishing returns are likely past single-feature conditioning.
5. **Paper vs. product.** Is the web interface part of the academic artifact (reproducible demo) or a separate side project? Affects polish level.
6. **Range tracker likelihood initialization.** The Bayesian range tracker (§5.4) needs a likelihood function L(bid | range) to compute updates. Before the network is trained, what prior to use? Options: (a) uniform over legal bids, (b) blind-equilibrium bid frequencies per pool size, (c) learned jointly with the policy. Option (b) is the natural warm-start: the blind equilibrium tells us what a rational player with no private info would bid, giving a plausible default likelihood.
7. ~~**Exact-rules blind equilibrium.**~~ **RESOLVED.** Computed and cached for n=2..10 via `get_blind_equilibrium_exact()`. First bid = HC A for all n; see §2.6 results table. Cache key prefix: `"exact_{n}"` in `agent/data/blind_equilibrium.json`.

These should be resolved as training progresses through Stages A–C.
