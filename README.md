# Liar's Poker — Nash Equilibrium Research

**"Strategic Analysis of Card-Based Liar's Poker: Combinatorial Foundations and Nash Equilibrium Approximation"**
*Connor M. Lockhart — University of Maryland*

A research project on a card-based variant of Liar's Poker played with a
standard 52-card deck. Players bid on the quality of the combined pool of
all players' hidden hands using standard poker hand rankings. The work
spans (1) combinatorial / probability analysis of the game and (2)
self-play training of an agent toward Nash equilibrium.

---

## Project Status (2026-04-27)

The project is mid-refactor. The pre-refactor monolith was split into a
phased plan (P0–P6); P0–P5 are complete, P6 (markdown consolidation +
acceptance validation) is in progress.

| Phase | Scope | Status |
| --- | --- | --- |
| P0 | Preflight — ADRs, deps, tracking stack | ✅ |
| P1 | `src/` skeleton, game package move, unified `tests/` tree | ✅ |
| P2 | Frontend retirement, `archive/web-2026-04/` | ✅ |
| P3 | Decision logging + reflect v1 | ✅ |
| P4 | OpenSpiel adapter + Kuhn-oracle exploitability + benchmark wiring | ✅ |
| **P5** | **HH adapter, honest per-agent exploitability (LBR + sampled subgame), reflect rules** | **✅ (2026-04-27)** |
| P6 | Markdown consolidation, acceptance validation, doc polish | 🟡 in progress |

See [`CHANGELOG.md`](CHANGELOG.md) for the full per-phase change log and
[`docs-internal/design/`](docs-internal/design/) for active design docs
and ADRs.

### What works today

- **Pure-Python game engine** with exact-rules + High-Hand declarations,
  fully tested (`src/game/`).
- **Heuristic agent zoo** (random, blind, conditional, mixed, opponent
  model, adaptive, biased) registered in `src/agents/registry.py` and run
  pairwise via `src/training/benchmark.py`.
- **CFR+ reference solver** (`src/agents/heuristic/cfr_1v1*.py`) — Nash on
  a restricted bid space; useful as a baseline.
- **R-NaD trainer** (`src/agents/learned/rnad/`) — currently weak on the
  full game; slated for a rewrite (see Next Steps).
- **OpenSpiel adapter** for both a Kuhn-sized oracle and the full
  single-round 52-card game with HH wired (`src/interop/`).
- **Honest exploitability metrics** — Local Best Response (LBR) and
  sampled subgame exploitability, per-agent, wired into the benchmark CLI
  (`src/training/metrics/`).
- **Decision logging + reflection** — JSONL per-turn capture and a
  rule-based summary writer that flags infeasible bids, stale openings,
  missed calls, and rank leaks (`src/training/decision_capture.py`,
  `src/training/reflect.py`).

### Known weaknesses

- The current `CFRNashAgent` plays the bid-restricted CFR Nash and is
  near-random against the heuristic ladder (~49% win rate). Item P5-2 was
  built so we can *measure* this honestly; the actual fix is a planned
  agent rewrite (see Next Steps).
- The R-NaD checkpoint pre-dates the eval-fix and HH-enabled era and
  should not be used until retrained.
- One pre-existing test failure (`tests/game/test_bids.py::test_bid_count`)
  is a stale assertion; harmless.

---

## Repository Layout

```text
liars-poker/
├── src/                 — Python source packages
│   ├── game/            — pure-Python match engine + bid space
│   ├── agents/          — heuristic + learned agents, registry, policy contract
│   ├── interop/         — OpenSpiel adapter + adapter↔engine state bridge
│   └── training/        — benchmark runner, decision logger, reflect, metrics, prob tables
├── tests/               — pytest suite (mirrors src/)
├── data/
│   ├── probs/           — probability JSON caches
│   ├── runs/            — benchmark + training run outputs
│   ├── checkpoints/     — trained model checkpoints (gitignored)
│   └── oracles/         — Kuhn reference policy
├── docs-internal/
│   └── design/
│       ├── adr/         — Architecture Decision Records (irreversible decisions)
│       ├── *.md         — active design docs (P5, small games, …)
│       └── legacy/      — pre-refactor design / planning docs (read-only archive)
├── paper/               — LaTeX source for the Stage 1 paper
├── archive/             — frozen retired code (web demo)
├── CHANGELOG.md         — single linear change log (release tags resume after P6)
└── README.md            — this file
```

See [`src/README.md`](src/README.md) for a per-module description.

---

## How to Use It

All commands are run from the repository root with the project installed
via `pip install -e ".[dev]"` (one-time setup).

### Run tests

```bash
pytest tests/
```

Expect `124 passed, 1 pre-existing failure` (the stale `test_bid_count`).
R-NaD tests are skipped unless PyTorch is installed.

### Run the agent benchmark

```bash
# Pairwise win-rate matrix on the heuristic zoo (default 100 games/pair)
python -m training.benchmark

# With per-turn decision logging + reflection summary
python -m training.benchmark --games 500 --groups exact --log-decisions --run-name <name>
python -m training.reflect <run_id>   # writes summary.md to data/runs/<run_id>/
```

Output lands in `data/runs/<run_id>/metrics.json` (with `--log-decisions`)
or `data/runs/benchmark/benchmark_results.json` (without).

### Measure exploitability

```bash
# Kuhn-oracle exploitability (the small-game variant)
python -m training.benchmark --exploitability

# Per-agent honest exploitability via Local Best Response and sampled subgame
python -m training.benchmark --groups exact --lbr --subgame --exploitability-deals 50

# Paper-grade run (slower)
python -m training.benchmark --groups exact --lbr --subgame \
    --exploitability-deals 200 --lbr-depth 3
```

Per-agent results land under `output["agent_exploitability"][<key>]`
in the run's `metrics.json`. Schema is pinned in
[`docs-internal/design/p5_2_exploitability.md`](docs-internal/design/p5_2_exploitability.md) §2c.

### Train an R-NaD agent (legacy — pending rewrite)

```bash
# Stage A — fixed hand size
python -m agents.learned.rnad.trainer --stage A --hand-size 1 --iterations 20000 --device cpu

# Stage B — full match with elimination
python -m agents.learned.rnad.trainer --stage B --iterations 20000 --device cpu

# Resume
python -m agents.learned.rnad.trainer --resume data/checkpoints/rnad_final.pt
```

Always pass `--device cpu` — MPS is ~21× slower for this trainer (sequential
data collection swamps the GPU win).

### Compile the paper

```bash
cd paper/
latexmk -pdf Liars-poker.tex
```

Probability tables and figures are pre-rendered to PDF and included via
`\includegraphics`; the `.tex` file does not load Python data directly.
To regenerate them, run the scripts in `src/training/probs/` first.

---

## What's Next

P5 closed the **measurement** half of the project. The next major chunk
is the **agent half** — the current Nash / CFR / R-NaD agents are weak
on the full game and need to be rebuilt. See `New-features.md` §3.4–3.5
in the legacy archive for the original framing.

Roadmap (in order):

1. **P6 — markdown consolidation + acceptance validation.** This README
   and the legacy archive are the consolidation step; acceptance is a
   500-game benchmark run with decision logging + W&B push, plus
   verification that §12 acceptance criteria are met.
2. **Agent rewrite design doc.** Per the design-first gate, before any
   code: scope a modular agent stack (hand-model + bid-policy +
   call-policy modules) with the new exploitability metrics in the loop
   from day one. Lift relevant sections from
   [`docs-internal/design/legacy/AGENT_DESIGN.md`](docs-internal/design/legacy/AGENT_DESIGN.md)
   and [`LITERATURE_SURVEY.md`](docs-internal/design/legacy/LITERATURE_SURVEY.md).
3. **New agent implementation** — multi-session, scope-dependent. The
   target is an agent whose LBR exploitability is materially below the
   heuristic ladder's.
4. **Stage 2 paper section** — once a strong agent exists, write up the
   training story alongside the existing combinatorial analysis.

---

## References

- Perolat et al. (2022) — *Mastering the Game of Stratego with Model-Free
  Multiagent RL* (R-NaD). arXiv:2206.05825
- Dewey et al. (2025) — *Mastering Liar's Poker via Self-Play and RL*
  (dice variant). arXiv:2511.03724
- Wu & Wu (2024) — Exact formulas for Straight / Flush / Full House in
  n-card poker. arXiv:2309.00011
- Lanctot et al. (2019) — OpenSpiel. arXiv:1908.09453
- Lisý & Bowling (2017) — Local Best Response (LBR) — used in the
  per-agent exploitability metric.
