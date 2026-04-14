# Liar's Poker — Nash Equilibrium Research

**"Strategic Analysis of Card-Based Liar's Poker: Combinatorial Foundations and Nash Equilibrium Approximation"**
*Connor M. Lockhart — University of Maryland*

A research project studying a card-based variant of Liar's Poker played with a standard 52-card deck. Players bid on the quality of the combined pool of all players' hidden hands using standard poker hand rankings. The project has two stages: a combinatorial/probability analysis (Stage 1, written up as a paper) and a reinforcement-learning agent trained to play at near-Nash-equilibrium (Stage 2).

---

## Play Online

A playable web client is hosted on GitHub Pages, built entirely in JavaScript — no server required.

**[Play Liar's Poker](https://conair94.github.io/liars-poker/)**

Supports 2–5 players, spectator mode, and two AI opponents:
- **Random** — picks any legal action uniformly at random
- **Blind Baseline** — plays the N=2 backward-induction Nash equilibrium (ignores private cards; calls at the ~50% probability threshold)

---

## Game Rules

- 2–5 players, each starting with 1 face-down card. The count grows when you lose rounds.
- The designated first bidder names a **poker hand** (e.g. "Pair 9", "Straight A"). Each subsequent player must name a **strictly stronger hand** or **Call Bluff**.
- On a call, all hands are revealed and combined into a **pool**. The pool's best 5-card hand is evaluated.
  - Pool ≥ standing bid → **caller loses**
  - Pool < standing bid → **bidder loses**
- The loser gets +1 card. A player holding 5 cards who loses a round is **eliminated**.
- Last player standing wins.

---

## Repository Layout

```
liars-poker/
├── Liars poker/
│   ├── Liars-poker.tex              # Stage 1 paper (LaTeX)
│   ├── AGENT_DESIGN.md              # Stage 2 design & milestone tracker
│   ├── LITERATURE_SURVEY.md
│   ├── poker_math_exact.py          # hand evaluator + probability engine
│   ├── generate_prob_tables.py      # marginal figure generation
│   ├── compute_conditional_probs.py # conditional probability figures
│   ├── figures/                     # pre-compiled figures + JSON caches
│   └── agent/
│       ├── game/
│       │   ├── engine.py            # pure-Python match engine
│       │   └── bids.py              # bid space (110 bids)
│       ├── baseline/
│       │   └── blind_equilibrium.py # N=2 backward-induction equilibrium solver
│       ├── rnad/
│       │   ├── config.py            # RNaDConfig hyperparameters
│       │   ├── network.py           # LiarsPokerNet (418K params)
│       │   ├── trainer.py           # R-NaD self-play trainer (Stage A + B)
│       │   ├── eval.py              # evaluation suite
│       │   └── warm_start.py        # probability warm-start lookup
│       ├── web/
│       │   └── backend/             # FastAPI server (local play)
│       ├── data/                    # Monte Carlo caches + warm-start tables
│       └── checkpoints/             # trained model checkpoints (gitignored)
└── docs/
    ├── index.html                   # self-contained GitHub Pages client
    └── .nojekyll
```

---

## Milestone Progress

| # | Milestone | Status |
|---|-----------|--------|
| M0 | Literature survey (R-NaD, CFR-free methods, Shi et al. APU) | ✅ Done |
| M1 | Game engine — `engine.py` + `bids.py`, unit tests | ✅ Done |
| M2 | Blind baseline equilibrium — backward induction N=2, cached for n=2..10 | ✅ Done |
| M2b | Warm-start lookup — marginal + conditional probability vectors for the network | ✅ Done |
| M3 | R-NaD trainer — Stage A (fixed hand size) + Stage B (full match/elimination), 14/14 tests | ✅ Done |
| M6a | Web interface — FastAPI + HTMX local server; GitHub Pages JS client | ✅ Done |
| M4 | Full-game R-NaD agent — trained checkpoint + evaluation vs baselines | 🔲 Planned |
| M5 | Test-time compute — search-augmented inference (ReBeL-style or iterative BR) | 🔲 Planned |
| M6b | Full web interface — trained agent, match history, win stats | 🔲 Planned |

---

## Stage 1 — Combinatorial Analysis

The paper computes exact and Monte Carlo hand-type probabilities for pool sizes n=5..25, and conditional probabilities given six private-hand conditions (pair, trips, suited, 3-suited, adjacent, 3-range). These tables are used directly as warm-start features for the Stage 2 network.

Key result: the N=2 blind equilibrium first bid tracks the 50%-threshold bid across all pool sizes — e.g. "Pair 2" at n=5, "Straight 9" at n=10.

---

## Stage 2 — R-NaD Agent

The agent is trained via **Regularized Nash Dynamics (R-NaD)** — reward-transformed self-play that provably converges to Nash equilibrium in two-player zero-sum games (Perolat et al. 2022).

### Network

`LiarsPokerNet` (418K parameters):
- **Card encoder** — DeepSet sum-pool over private cards (1–5 cards)
- **Warm-start features** — marginal + conditional probability vectors from Stage 1 tables (static, no gradient)
- **Bid history encoder** — embedding lookup over last 6 actions
- **Trunk** — 4-layer MLP with LayerNorm + ReLU
- **Heads** — policy (110 bids + call), value (scalar ∈ (−1, 1)), auxiliary conditional-distribution predictor

### Training

```bash
# Stage A — fixed hand size, single-round episodes
cd "Liars poker/"
python3 -m agent.rnad.trainer --stage A --hand-size 1 --iterations 20000

# Stage B — full match with elimination
python3 -m agent.rnad.trainer --stage B --iterations 20000

# Resume from checkpoint
python3 -m agent.rnad.trainer --resume agent/checkpoints/rnad_final.pt
```

### Local Web Server

```bash
cd "Liars poker/"
python3 agent/web/run.py
# Open http://localhost:8000
```

---

## References

- Perolat et al. (2022) — *Mastering the Game of Stratego with Model-Free Multiagent RL* (R-NaD). arXiv:2206.05825
- Dewey et al. (2025) — *Mastering Liar's Poker via Self-Play and RL* (dice variant). arXiv:2511.03724
- Shi et al. (2022) — APU/Dual-APU for multiplayer training stability
- Wu & Wu (2024) — Exact formulas for Straight/Flush/Full House in n-card poker. arXiv:2309.00011
- Lanctot et al. (2019) — OpenSpiel. arXiv:1908.09453
