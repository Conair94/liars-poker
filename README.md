# Liar's Poker — Nash Equilibrium Research

**"Strategic Analysis of Card-Based Liar's Poker: Combinatorial Foundations and Nash Equilibrium Approximation"**  
*Connor M. Lockhart — University of Maryland*

A research project studying a card-based variant of Liar's Poker played with a standard 52-card deck.
Players bid on the quality of the combined pool of all players' hidden hands using standard poker hand
rankings. The project has two stages:

1. **Stage 1** — combinatorial and probability analysis (see `paper/`)
2. **Stage 2** — reinforcement-learning agent trained via R-NaD self-play (local-dev only until
   agents are strong enough to re-host)

---

## Repository Layout

```text
liars-poker/
├── src/                 — Python source packages (game engine, agents, training)
│   ├── game/            — pure-Python match engine + bid space
│   ├── agents/          — heuristic + learned agents, registry
│   └── training/        — training scripts, benchmark runner, prob tables
├── tests/               — pytest test suite (mirrors src/)
├── data/
│   ├── probs/           — probability JSON caches
│   ├── runs/            — training run outputs + checkpoints
│   └── checkpoints/     — trained model checkpoints (gitignored)
├── docs/                — GitHub Pages JS client (local-dev only)
├── docs-internal/       — design docs, ADRs, milestone tracking
│   └── design/
├── paper/               — LaTeX source for Stage 1 paper
└── Liars poker/         — legacy directory (archived in P6)
```

See [src/README.md](src/README.md) for a detailed description of each subpackage.

---

## Running Tests

```bash
pytest tests/
```

One pre-existing failure: `test_bid_count` — tracks a known open question about
the High Card bid count (does not affect any agent behaviour).

---

## Running the Benchmark

```bash
python -m training.benchmark
```

Results are written to `data/runs/benchmark/`. See `data/runs/benchmark/README.md`
for the result format.

---

## Training an R-NaD Agent

```bash
# Stage A — fixed hand size
python -m agents.learned.rnad.trainer --stage A --hand-size 1 --iterations 20000

# Stage B — full match with elimination
python -m agents.learned.rnad.trainer --stage B --iterations 20000

# Resume from checkpoint
python -m agents.learned.rnad.trainer --resume data/checkpoints/rnad_final.pt
```

---

## Milestone Progress

| # | Milestone | Status |
| --- | --------- | ------ |
| M0 | Literature survey | Done |
| M1 | Game engine — engine.py + bids.py, unit tests | Done |
| M2 | Blind baseline equilibrium — backward induction N=2, cached n=2..10 | Done |
| M2b | Warm-start lookup — marginal + conditional probability vectors | Done |
| M3 | R-NaD trainer — Stage A + Stage B, 14/14 tests | Done |
| M6a | Web interface — GitHub Pages JS client | Done (local-dev only) |
| M4 | Full-game R-NaD agent — trained checkpoint + evaluation | Planned |
| M5 | Test-time compute — search-augmented inference | Planned |
| M6b | Full web interface — trained agent, match history, win stats | Planned |

---

## References

- Perolat et al. (2022) — *Mastering the Game of Stratego with Model-Free Multiagent RL* (R-NaD). arXiv:2206.05825
- Dewey et al. (2025) — *Mastering Liar's Poker via Self-Play and RL* (dice variant). arXiv:2511.03724
- Wu & Wu (2024) — Exact formulas for Straight/Flush/Full House in n-card poker. arXiv:2309.00011
- Lanctot et al. (2019) — OpenSpiel. arXiv:1908.09453
